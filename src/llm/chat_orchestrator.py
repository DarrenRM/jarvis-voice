"""Chat orchestrator for LLM tool use coordination.

Manages:
- Tool loop (up to max_turns), one streaming LLM call per turn
- Tool execution with every tool call answered, even on bad arguments
- Error handling that never writes error text into history
"""

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Protocol, Union

from ..tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


# Discord action tools that should have brief acknowledgments (no follow-up conversation)
DISCORD_ACTION_TOOLS = {"tag_user", "post_to_chat", "post_table"}
DISCORD_ACTION_ACK_NOTE = (
    "Action completed. Give only a brief 1-2 word acknowledgment "
    "like 'Done' or 'Tagged'. Do not elaborate or answer other "
    "questions from the original message."
)

ERROR_REPLY = "Sorry, I encountered an error processing your request."
GRACEFUL_FAILURE_REPLY = (
    "[surprised] I couldn't find a good answer. Could you try asking differently?"
)

# Text before a tool call is filler ("Let me check that...") that the thinking
# phrase already covers. Short text is held until the turn ends (Cerebras
# finishes a turn ~30-70ms after its first text) and dropped if tool calls
# follow; anything longer is a real answer and streams straight away.
HOLD_PRE_TOOL_CHARS = 150


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class TurnResult:
    """Everything the model produced in one streamed turn."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class TurnOutcome:
    """Filled in by chat_stream so callers know whether to keep the turn in history."""

    ok: bool = False


class LLMClientProtocol(Protocol):
    """Interface the orchestrator needs from an LLM client."""

    def stream_turn(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[Union[str, TurnResult]]:
        """Stream text chunks, then yield exactly one TurnResult."""
        ...


class ChatOrchestrator:
    """Orchestrates LLM chat with tool use support.

    Example:
        orchestrator = ChatOrchestrator(llm_client, tool_executor)

        async for chunk in orchestrator.chat_stream(messages):
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
            max_turns: Maximum tool loop turns before giving up
        """
        self._llm = llm_client
        self._executor = tool_executor
        self._max_turns = max_turns

    async def chat_stream(
        self,
        messages: list[dict],
        outcome: Optional[TurnOutcome] = None,
    ) -> AsyncIterator[str]:
        """Stream response with tool use support.

        Args:
            messages: Full message history (system prompt first, user message last).
                Modified in place: assistant/tool messages are appended, ending
                with the final assistant reply when the turn succeeds.
            outcome: Optional; set to ok=True only if the turn completed normally.

        Yields:
            Response text chunks
        """
        outcome = outcome or TurnOutcome()
        tools = self._executor.get_definitions() or None

        for turn in range(self._max_turns):
            logger.debug(f"Orchestrator turn {turn + 1}/{self._max_turns}")

            result: Optional[TurnResult] = None
            held: list[str] = []
            streaming = False  # True once text is long enough to be a real answer
            try:
                async for event in self._llm.stream_turn(messages, tools):
                    if isinstance(event, TurnResult):
                        result = event
                    elif streaming:
                        yield event
                    else:
                        held.append(event)
                        if sum(len(h) for h in held) > HOLD_PRE_TOOL_CHARS:
                            streaming = True
                            yield "".join(held)
                            held = []
            except Exception as e:
                logger.error(f"LLM API error: {e}")
                if held:
                    yield "".join(held)
                yield ERROR_REPLY
                return

            if result is None:
                logger.error("LLM stream ended without a result")
                if held:
                    yield "".join(held)
                yield ERROR_REPLY
                return

            if not result.has_tool_calls:
                if held:
                    yield "".join(held)
                if not result.content.strip():
                    logger.warning(f"LLM returned no text (finish={result.finish_reason})")
                    yield GRACEFUL_FAILURE_REPLY
                    return
                messages.append({"role": "assistant", "content": result.content})
                outcome.ok = True
                return

            logger.debug(f"Tool calls detected: {len(result.tool_calls)}")

            if streaming:
                # Flush the spoken pre-tool text to TTS before the tool runs
                if not result.content.endswith("\n"):
                    yield "\n"
            elif result.content.strip():
                logger.info(f"Not speaking pre-tool filler: {result.content.strip()[:80]!r}")

            # Unspoken filler is left out of history so it isn't imitated later
            messages.append(self._build_assistant_tool_message(result, spoken=streaming))
            for tool_call in result.tool_calls:
                await self._execute_tool(tool_call, messages)

        logger.warning(f"Max tool turns ({self._max_turns}) exhausted")
        yield GRACEFUL_FAILURE_REPLY

    def _build_assistant_tool_message(self, result: TurnResult, spoken: bool = True) -> dict:
        """Build assistant message with tool calls for history."""
        return {
            "role": "assistant",
            "content": (result.content.strip() or None) if spoken else None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in result.tool_calls
            ],
        }

    async def _execute_tool(self, tool_call: ToolCall, messages: list[dict]) -> None:
        """Execute one tool call and always append its tool result message."""
        tool_name = tool_call.name

        try:
            args = json.loads(tool_call.arguments or "{}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse tool arguments for {tool_name}: {e}")
            args = None

        if not isinstance(args, dict):
            result_output = "Error: Invalid arguments"
        else:
            result = await self._executor.execute(tool_name, args)
            result_output = result.output if result.success else f"Error: {result.error}"

        # Discord actions are side effects that don't need conversational follow-up.
        # Carried in the tool result: Qwen's chat template rejects system messages
        # anywhere but the start of the conversation.
        if tool_name in DISCORD_ACTION_TOOLS and "Done" in result_output:
            result_output += f"\n\n{DISCORD_ACTION_ACK_NOTE}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_output,
        })
