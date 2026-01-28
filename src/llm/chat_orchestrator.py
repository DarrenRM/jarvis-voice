"""Chat orchestrator for LLM tool use coordination.

Extracts the tool loop logic from groq_client.py (lines 292-567).
Manages:
- Tool loop (up to max_turns)
- Search limit enforcement (3 max)
- Counter tracking
- Error recovery
- Streaming coordination
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol

from ..tools.executor import ToolExecutor, ACTION_TOOLS

logger = logging.getLogger(__name__)


# Maximum number of search tool calls before forcing an answer
MAX_SEARCH_CALLS = 3

# Discord action tools that should have brief acknowledgments (no follow-up conversation)
DISCORD_ACTION_TOOLS = {"tag_user", "post_to_chat", "post_table"}


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that the orchestrator can use.

    Defines the interface the orchestrator needs from an LLM client.
    """

    async def call_api(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> "APIResponse":
        """Make a non-streaming API call.

        Args:
            messages: Conversation messages
            tools: Tool definitions (None to disable tools)
            tool_choice: Tool choice mode ("auto", "none")

        Returns:
            APIResponse with message and optional tool calls
        """
        ...

    async def stream_response(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Stream a response (tools disabled).

        Args:
            messages: Conversation messages

        Yields:
            Response text chunks
        """
        ...


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class APIResponse:
    """Response from LLM API call."""

    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class OrchestratorState:
    """State tracked during a chat_stream request."""

    tool_calls_this_request: list[str] = field(default_factory=list)
    search_calls_this_request: int = 0
    last_action_success: Optional[str] = None

    @property
    def search_limit_reached(self) -> bool:
        """Check if search limit has been reached."""
        return self.search_calls_this_request >= MAX_SEARCH_CALLS


class ChatOrchestrator:
    """Orchestrates LLM chat with tool use support.

    Extracted from groq_client.py. Manages:
    - Tool loop (up to max_turns)
    - Search limit enforcement (3 max)
    - Counter tracking
    - Error recovery
    - Streaming coordination

    Uses ToolExecutor's is_search_tool() and is_action_tool()
    for limit tracking (constants already defined in executor.py).

    Example:
        orchestrator = ChatOrchestrator(llm_client, tool_executor)

        async for chunk in orchestrator.chat_stream(message, messages):
            print(chunk, end="")
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        tool_executor: ToolExecutor,
        max_turns: int = 5,
    ):
        """Initialize chat orchestrator.

        Args:
            llm_client: LLM client implementing LLMClientProtocol
            tool_executor: Tool executor for running tools
            max_turns: Maximum tool loop turns before forcing answer
        """
        self._llm = llm_client
        self._executor = tool_executor
        self._max_turns = max_turns

    async def chat_stream(
        self,
        message: str,
        messages: list[dict],
        on_message_update: Optional[callable] = None,
    ) -> AsyncIterator[str]:
        """Stream response with tool use support.

        Args:
            message: User message (for tracking/graceful failure)
            messages: Full message history (including system prompt)
            on_message_update: Optional callback when messages are modified

        Yields:
            Response text chunks

        Note:
            The messages list is modified in-place as tool calls are made.
            Use on_message_update to track these changes if needed.
        """
        # Reset state for this request
        state = OrchestratorState()
        self._executor.reset_counters()

        # Tool loop
        for turn in range(self._max_turns):
            logger.debug(f"Orchestrator turn {turn + 1}/{self._max_turns}")

            # Get available tools based on search limit
            available_tools = self._executor.get_definitions(
                search_limit_reached=state.search_limit_reached
            )

            if state.search_limit_reached:
                logger.info(
                    f"Search limit reached ({state.search_calls_this_request}), "
                    "action tools only"
                )
                # Inject force-answer message
                self._inject_force_answer_message(messages)

            # Make API call with tools
            try:
                response = await self._llm.call_api(
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    tool_choice="auto" if available_tools else "none",
                )
            except Exception as e:
                # Handle API errors
                async for chunk in self._handle_api_error(e, state, message, messages):
                    yield chunk
                return

            # Check for tool calls
            if response.has_tool_calls:
                logger.debug(f"Tool calls detected: {len(response.tool_calls)}")

                # Yield pre-tool content if any
                # Pattern from groq_client.py:432-435
                if response.content:
                    yield response.content
                    if not response.content.endswith("\n"):
                        yield "\n"

                # Add assistant message with tool calls to history
                messages.append(self._build_assistant_tool_message(response))
                if on_message_update:
                    on_message_update()

                # Execute tools
                for tool_call in response.tool_calls:
                    async for chunk in self._execute_tool(
                        tool_call, state, messages, on_message_update
                    ):
                        yield chunk

                continue  # Next turn to process tool results

            # No tool calls - stream the final response
            async for chunk in self._llm.stream_response(messages):
                yield chunk

            return  # Exit after streaming response

        # Max turns exhausted - graceful failure
        logger.warning(
            f"Max tool turns ({self._max_turns}) exhausted. "
            f"Tool calls: {state.tool_calls_this_request}"
        )
        yield self._build_graceful_failure()

    def _inject_force_answer_message(self, messages: list[dict]) -> None:
        """Inject system message forcing model to answer with gathered info.

        Pattern from groq_client.py:352-357
        """
        force_answer_msg = (
            "You have gathered enough search information. DO NOT search again. "
            "Either: (1) Answer the user's question using the info you gathered, OR "
            "(2) If the user requested an image/infographic, call generate_image NOW "
            "with a detailed prompt based on what you learned. "
            "Give a brief 1-2 sentence voice response. "
            "NO MARKDOWN - no asterisks, bold, headers, or formatting."
        )

        # Only inject once
        if not any(m.get("content") == force_answer_msg for m in messages):
            messages.append({"role": "system", "content": force_answer_msg})

    def _build_assistant_tool_message(self, response: APIResponse) -> dict:
        """Build assistant message with tool calls for history.

        Pattern from groq_client.py:438-451
        """
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in response.tool_calls
            ],
        }

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        state: OrchestratorState,
        messages: list[dict],
        on_message_update: Optional[callable],
    ) -> AsyncIterator[str]:
        """Execute a single tool call and update state.

        Patterns from groq_client.py:454-493
        """
        tool_name = self._executor.normalize_name(tool_call.name)

        # Special notification for deep research (before execution)
        # Pattern from groq_client.py:460-461
        if tool_name == "deep_research":
            yield "[thoughtful] This one needs some deeper digging, give me a moment.\n"

        # Track search tools toward limit
        if self._executor.is_search_tool(tool_name):
            state.search_calls_this_request += 1

        # Track tool call for graceful failure message
        try:
            args = json.loads(tool_call.arguments)
            if tool_name == "web_search":
                state.tool_calls_this_request.append(f"searched '{args.get('query', '')}'")
            elif tool_name == "fetch_url":
                state.tool_calls_this_request.append(f"fetched {args.get('url', '')}")
            else:
                state.tool_calls_this_request.append(tool_name)
        except json.JSONDecodeError:
            state.tool_calls_this_request.append(tool_name)

        # Execute tool
        try:
            args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            result_output = "Error: Invalid arguments"
        else:
            result = await self._executor.execute(tool_name, args)
            result_output = result.output if result.success else f"Error: {result.error}"

        # Add tool result to messages
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.name,
            "content": result_output,
        })
        if on_message_update:
            on_message_update()

        # Track successful action tool results for fallback
        # Pattern from groq_client.py:488-492
        if self._executor.is_action_tool(tool_name) and result_output.startswith("Done!"):
            # Extract just the success message (before "You can now edit...")
            success_msg = result_output.split(".")[0] + "."
            state.last_action_success = success_msg

        # Inject brief ack guidance for Discord action tools
        # These are side-effect tools that don't need conversational follow-up
        if tool_name in DISCORD_ACTION_TOOLS and "Done" in result_output:
            messages.append({
                "role": "system",
                "content": (
                    "Action completed. Give only a brief 1-2 word acknowledgment "
                    "like 'Done' or 'Tagged'. Do not elaborate or answer other "
                    "questions from the original message."
                ),
            })

    async def _handle_api_error(
        self,
        error: Exception,
        state: OrchestratorState,
        message: str,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Handle API errors with recovery strategies.

        Patterns from groq_client.py:374-423, 515-558
        """
        error_str = str(error)

        # Handle output parsing error
        # Pattern from groq_client.py:518-525
        if "output_parse_failed" in error_str or "Parsing failed" in error_str:
            logger.warning(f"Output parsing failed: {error}")
            if state.last_action_success:
                logger.info(f"Using last action success as fallback: {state.last_action_success}")
                yield state.last_action_success
            else:
                yield "I couldn't find specific information about that. Try asking in a different way."
            return

        # Handle model trying to call tools when disabled
        # Pattern from groq_client.py:527-555
        if "tool_use_failed" in error_str or "model called a tool" in error_str:
            logger.warning(f"Model tried to call tool after limit: {error}")

            # Try to summarize gathered tool results
            tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
            if tool_results:
                logger.info(f"Attempting summary of {len(tool_results)} tool results")
                summary = await self._attempt_summary(tool_results[-3:], message)
                if summary:
                    yield summary
                    return

            yield self._build_graceful_failure()
            return

        # Other errors
        logger.error(f"API error: {error}")
        yield "Sorry, I encountered an error processing your request."

    async def _attempt_summary(
        self,
        tool_results: list[str],
        original_message: str,
    ) -> Optional[str]:
        """Attempt to summarize tool results when API errors occur.

        Pattern from groq_client.py:530-551
        """
        try:
            summary_context = "\n\n".join(tool_results)
            summary_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful voice assistant. "
                        "Extract and answer the question from the search results. "
                        "Be brief (1-2 sentences). Use emotion tags like [happy], [surprised]. "
                        "NO MARKDOWN - no asterisks, bold, headers, or formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on this search data, answer: {original_message}\n\n"
                        f"Search results:\n{summary_context[:6000]}"
                    ),
                },
            ]

            response = await self._llm.call_api(
                messages=summary_messages,
                tools=None,
                tool_choice="none",
            )

            if response.content:
                logger.info(f"Summary retry succeeded: {response.content[:100]}...")
                return response.content

        except Exception as e:
            logger.warning(f"Summary retry failed: {e}")

        return None

    def _build_graceful_failure(self) -> str:
        """Build a graceful failure message.

        Pattern from groq_client.py:560-567
        Keep it SHORT for voice - don't list what we searched.
        """
        return "[surprised] I searched but couldn't find a good answer. Could you try asking differently?"
