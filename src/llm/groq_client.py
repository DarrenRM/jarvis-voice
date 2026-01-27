"""Groq LLM client with tool use support.

Supports multiple models via Groq's API:
- llama-3.3-70b-versatile (Llama 3.3 70B)
- openai/gpt-oss-120b (GPT-OSS 120B)
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Optional

from .base import BaseLLMClient
from ..search.search_service import SearchService
from ..config import get_settings

if TYPE_CHECKING:
    from ..image import NanoBananaClient
    from .claude_client import ClaudeClient
    from ..tools import ToolRegistry, ToolExecutor
    from .chat_orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

# Tool definitions for Groq function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. USE THIS for: game wikis, recipes, crafting guides, item info, build guides, current events, news, weather, prices, sports, or ANY factual question you're not 100% certain about. If it is complex or multi-step use the deep_research tool instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read the content of a specific URL. Use when you need to read a webpage the user mentioned or to get detailed information from a search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a NEW image from a text description. Use when user asks you to create, generate, draw, make, or wants an image/picture/photo/infographic. For editing or changing or adding details to an existing image, use edit_image instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Edit or modify the most recently generated image. Use when user says 'edit that', 'change the image', 'modify it', 'update that picture', 'give it X', 'make it Y', or references a previous image they want changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change about the image (e.g., 'add silly hats to everyone', 'make it more colorful', 'change the background to a beach')"
                    }
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tag_user",
            "description": "Tag/mention a Discord user in chat with a message. Use when user says 'tag', 'message', 'tell', 'ping', or 'let X know'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Name of the person to tag (e.g., 'Jordan', 'Pat', 'Sam')"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send to the tagged user"
                    }
                },
                "required": ["target_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "post_to_chat",
            "description": "Post a plain text message to the Discord text channel. Use for simple messages, quotes, or single items. For comparisons, rankings, weather across locations, or data with multiple items - use post_table instead (it's much more readable). IMPORTANT: Do NOT include emotion tags like [happy], [laughing], [sigh] etc. in the message - those are for voice only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message content to post. Use Discord markdown (bold, lists, etc.) but NO emotion tags like [happy] or [laughing]."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "post_table",
            "description": "Post a formatted table to the Discord text channel. ALWAYS USE THIS instead of post_to_chat when presenting: comparisons between things, rankings, lists or ingredients, specs/stats, prices, or ANY data with 2+ items that share attributes. Tables are much easier to read than plain text lists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Optional title shown above the table"
                    },
                    "headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column headers"
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "Table rows (each row is array of cell values)"
                    }
                },
                "required": ["headers", "rows"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Perform deep research and analysis. ALWAYS USE THIS when user explicitly says 'deep research', 'think hard', 'analyze this', or asks for in-depth analysis - these phrases override all other rules. Also use for: complex reasoning, code questions, comparing multiple options, or synthesizing 3+ sources. DO NOT USE for simple factual questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to research or analyze. Include the user's question and focus areas."
                    },
                    "context": {
                        "type": "string",
                        "description": "Gathered information (search results, prior tool outputs) to consider."
                    }
                },
                "required": ["task"]
            }
        }
    }
]

# Model sometimes hallucinates tool names - map them to real ones
TOOL_ALIASES = {
    "web_fetch_url": "fetch_url",
    "web_fetch": "fetch_url",
}


class GroqClient(BaseLLMClient):
    """Groq LLM client with tool use support.

    Uses Groq's OpenAI-compatible API for fast inference with
    native function calling support. Supports multiple models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        max_turns: int = 5,
        search_service: Optional[SearchService] = None,
        image_service: Optional["NanoBananaClient"] = None,
        on_image_generated: Optional[Callable[[bytes, str, str], Awaitable[str]]] = None,
        on_user_tagged: Optional[Callable[[str, str], Awaitable[str]]] = None,
        on_message_posted: Optional[Callable[[str], Awaitable[str]]] = None,
        claude_service: Optional["ClaudeClient"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        """Initialize Groq client.

        Args:
            api_key: Groq API key
            model: Model ID (llama-3.3-70b-versatile or openai/gpt-oss-120b)
            max_tokens: Maximum response tokens
            max_turns: Maximum tool use turns before giving up
            search_service: Search service for tool execution
            image_service: Image generation service (Nano Banana)
            on_image_generated: Callback to post generated image (bytes, prompt, mime_type) -> result
            on_user_tagged: Callback to tag a user in Discord (target_name, message) -> result
            on_message_posted: Callback to post a message to Discord (message) -> result
            claude_service: Claude client for complex task delegation
            tool_registry: Optional tool registry for new modular tool system
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._tool_calls_this_request: list[str] = []  # Track tool calls for graceful failure
        self._search_calls_this_request: int = 0  # Track search/fetch calls toward limit
        self._context: str = ""
        self._conversation_history: list[dict] = []
        self._search_service = search_service
        self._image_service = image_service
        self._on_image_generated = on_image_generated
        self._on_user_tagged = on_user_tagged
        self._on_message_posted = on_message_posted
        self._claude_service = claude_service
        self._client = None

        # Feature flag: Use new modular tool system
        self._settings = get_settings()
        self._tool_executor = None
        self._prompt_builder = None

        if self._settings.use_new_tool_registry and tool_registry is not None:
            from ..tools import ToolExecutor
            self._tool_executor = ToolExecutor(tool_registry)
            logger.info("Using new modular tool registry")

        if self._settings.use_new_prompt_builder:
            from .system_prompt_builder import create_default_prompt_builder
            self._prompt_builder = create_default_prompt_builder()
            logger.info("Using new system prompt builder")

        # Feature flag: Use new ChatOrchestrator (Phase 2)
        self._chat_orchestrator = None
        if self._settings.use_new_chat_orchestrator and self._tool_executor is not None:
            from .chat_orchestrator import ChatOrchestrator
            self._chat_orchestrator = ChatOrchestrator(
                llm_client=self,  # self implements LLMClientProtocol
                tool_executor=self._tool_executor,
                max_turns=self.max_turns,
            )
            logger.info("Using new ChatOrchestrator")

    def _ensure_client(self) -> None:
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self.api_key)

    def set_context(self, context: str) -> None:
        """Set conversation context.

        Args:
            context: Context string to include in system prompt
        """
        self._context = context

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()

    @property
    def chat_orchestrator(self) -> Optional["ChatOrchestrator"]:
        """Get the ChatOrchestrator if using new orchestrator mode.

        Returns:
            ChatOrchestrator instance, or None if not enabled
        """
        return self._chat_orchestrator

    def get_system_prompt(self) -> str:
        """Get the current system prompt for the assistant.

        Returns:
            System prompt string (from prompt builder or default)
        """
        return self._build_system_prompt()

    # ========== LLMClientProtocol methods (for ChatOrchestrator) ==========

    async def call_api(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> "APIResponse":
        """Make a non-streaming API call (implements LLMClientProtocol).

        Args:
            messages: Conversation messages
            tools: Tool definitions (None to disable tools)
            tool_choice: Tool choice mode ("auto", "none")

        Returns:
            APIResponse with content and optional tool calls
        """
        from .chat_orchestrator import APIResponse, ToolCall

        self._ensure_client()

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice=tool_choice if tools else None,
            max_tokens=1024,
        )

        message = response.choices[0].message

        # Convert tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        return APIResponse(
            content=message.content,
            tool_calls=tool_calls,
        )

    async def stream_response(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream a response without tools (implements LLMClientProtocol).

        Args:
            messages: Conversation messages

        Yields:
            Response text chunks
        """
        self._ensure_client()

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=1024,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ========== End LLMClientProtocol methods ==========

    async def chat(self, message: str) -> str:
        """Send message and get complete response.

        Args:
            message: User message

        Returns:
            Complete response text
        """
        response_text = ""
        async for chunk in self.chat_stream(message):
            response_text += chunk
        return response_text

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Send message and stream response with tool use support.

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        try:
            self._ensure_client()

            # Build messages with system prompt and history
            messages = self._build_messages(message)

            # Feature flag: Use new ChatOrchestrator (Phase 2)
            if self._chat_orchestrator is not None:
                # Track user message in history
                self._conversation_history.append({
                    "role": "user",
                    "content": message,
                })

                # Delegate to orchestrator
                response_chunks = []
                async for chunk in self._chat_orchestrator.chat_stream(
                    message=message,
                    messages=messages,
                    on_message_update=lambda: None,  # Messages updated in-place
                ):
                    response_chunks.append(chunk)
                    yield chunk

                # Track assistant response in history
                full_response = "".join(response_chunks)
                if full_response:
                    self._conversation_history.append({
                        "role": "assistant",
                        "content": full_response,
                    })

                return  # Exit - orchestrator handled everything

            # ========== Old inline tool loop (when orchestrator disabled) ==========

            # Track user message
            self._conversation_history.append({
                "role": "user",
                "content": message,
            })

            # Reset tool call tracking for this request
            self._tool_calls_this_request = []
            self._search_calls_this_request = 0  # Only count search/fetch toward limit
            last_action_success = None  # Track successful action tool results (image, tag)

            # Reset counters on new tool executor if using it
            if self._tool_executor is not None:
                self._tool_executor.reset_counters()

            # Tool use loop (up to max_turns)
            # After 3 SEARCH tool calls, disable search but allow action tools
            MAX_SEARCH_CALLS = 3
            SEARCH_TOOLS = {"web_search", "fetch_url"} | set(TOOL_ALIASES.keys())  # These get limited
            ACTION_TOOLS = {"generate_image", "edit_image", "tag_user", "post_to_chat", "post_table", "deep_research"}  # These don't

            for turn in range(self.max_turns):
                logger.debug(f"Groq turn {turn + 1}/{self.max_turns}")

                # Request with tool support (non-streaming to detect tool calls)
                # After MAX_SEARCH_CALLS, disable search tools but keep action tools
                has_tools = self._search_service or self._image_service
                search_limit_reached = self._search_calls_this_request >= MAX_SEARCH_CALLS

                # Build available tools based on search limit
                # Feature flag: Use new executor's definitions if available
                if self._tool_executor is not None:
                    available_tools = self._tool_executor.get_definitions(
                        search_limit_reached=search_limit_reached
                    )
                    if search_limit_reached:
                        logger.info(f"Search limit reached ({self._search_calls_this_request}), action tools only")
                else:
                    available_tools = TOOLS.copy()
                    if search_limit_reached:
                        # Remove search tools, keep action tools (generate_image, edit_image, tag_user)
                        available_tools = [t for t in TOOLS if t["function"]["name"] in ACTION_TOOLS]
                        logger.info(f"Search limit reached ({self._search_calls_this_request}), action tools only: {[t['function']['name'] for t in available_tools]}")

                    # Add instruction to use gathered info or take action
                    force_answer_msg = "You have gathered enough search information. DO NOT search again. Either: (1) Answer the user's question using the info you gathered, OR (2) If the user requested an image/infographic, call generate_image NOW with a detailed prompt based on what you learned. Give a brief 1-2 sentence voice response."
                    if not any(m.get("content") == force_answer_msg for m in messages):
                        messages.append({
                            "role": "system",
                            "content": force_answer_msg
                        })

                # Build API call params
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                if has_tools and available_tools:
                    api_params["tools"] = available_tools
                    api_params["tool_choice"] = "auto"
                    api_params["parallel_tool_calls"] = False

                try:
                    response = await self._client.chat.completions.create(**api_params)
                except Exception as api_error:
                    error_str = str(api_error)
                    # If model tried to call a search tool when search is disabled, extract info and ask directly
                    if search_limit_reached and ("tool_use_failed" in error_str or "model called a tool" in error_str):
                        logger.warning(f"Model tried to call search tool after limit, retrying with summary request")
                        # Extract tool results from messages
                        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
                        summary_context = "\n\n".join(tool_results[-3:])  # Last 3 tool results

                        # Build output-only tools for summary (post tables/images, but NOT deep_research)
                        OUTPUT_TOOLS = {"generate_image", "edit_image", "tag_user", "post_to_chat", "post_table"}
                        output_only_tools = [t for t in TOOLS if t["function"]["name"] in OUTPUT_TOOLS]

                        # Ask model directly to summarize - include output tools so it can post tables
                        summary_messages = [
                            {"role": "system", "content": "You are a helpful voice assistant. Answer briefly in 1-2 sentences using the search results provided. If the data would be clearer as a table, use post_table. Do NOT search again."},
                            {"role": "user", "content": f"Based on this information, answer the user's question:\n\n{summary_context[:6000]}\n\nQuestion: {message}"}
                        ]
                        response = await self._client.chat.completions.create(
                            model=self.model,
                            messages=summary_messages,
                            max_tokens=self.max_tokens,
                            stream=False,
                            tools=output_only_tools,
                            tool_choice="auto",
                        )
                        # Handle tool call from summary (e.g., post_table)
                        summary_msg = response.choices[0].message
                        if summary_msg.tool_calls:
                            # Execute tools and collect results
                            tool_results = []
                            for tc in summary_msg.tool_calls:
                                result = await self._execute_tool(tc)
                                logger.info(f"Summary fallback executed tool: {tc.function.name} -> {result[:50]}...")
                                tool_results.append((tc, result))
                            # Build messages for final response
                            summary_messages.append({"role": "assistant", "tool_calls": [
                                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc, _ in tool_results
                            ]})
                            for tc, result in tool_results:
                                summary_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                            response = await self._client.chat.completions.create(
                                model=self.model,
                                messages=summary_messages,
                                max_tokens=self.max_tokens,
                                stream=False,
                                tool_choice="none",
                            )
                    else:
                        raise  # Re-raise other errors

                response_message = response.choices[0].message

                # Check for tool calls
                if response_message.tool_calls:
                    logger.debug(f"Tool calls detected: {len(response_message.tool_calls)}")

                    # If model says something before tool call, yield it
                    if response_message.content:
                        yield response_message.content
                        if not response_message.content.endswith('\n'):
                            yield '\n'

                    # Append assistant message with tool calls to messages
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ],
                    })

                    # Execute each tool and append results
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        # Normalize hallucinated tool names to real ones
                        tool_name = TOOL_ALIASES.get(tool_name, tool_name)

                        # Special notification for deep research (before execution)
                        if tool_name == "deep_research":
                            yield "[thoughtful] This one needs some deeper digging, give me a moment.\n"

                        # Track search tools toward limit
                        if tool_name in SEARCH_TOOLS:
                            self._search_calls_this_request += 1

                        # Track this tool call for graceful failure message
                        try:
                            args = json.loads(tool_call.function.arguments)
                            if tool_name == "web_search":
                                self._tool_calls_this_request.append(f"searched '{args.get('query', '')}'")
                            elif tool_name == "fetch_url":
                                self._tool_calls_this_request.append(f"fetched {args.get('url', '')}")
                            else:
                                self._tool_calls_this_request.append(tool_name)
                        except json.JSONDecodeError:
                            self._tool_calls_this_request.append(tool_name)

                        result = await self._execute_tool(tool_call)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": result,
                        })

                        # Track successful action tool results for fallback
                        # Image tools return "Done! I've posted...", tag tool returns "Done! I've tagged..."
                        if tool_name in ACTION_TOOLS and result.startswith("Done!"):
                            # Extract just the success message (before "You can now edit...")
                            success_msg = result.split(".")[0] + "."
                            last_action_success = success_msg

                    continue  # Next turn to process tool results

                # No tool calls - stream the final response
                async for chunk in self._stream_response(messages):
                    yield chunk

                break  # Exit loop after streaming response
            else:
                # Loop completed without break = max_turns exhausted
                # Provide graceful failure message
                logger.warning(f"Max tool turns ({self.max_turns}) exhausted. Tool calls: {self._tool_calls_this_request}")
                failure_msg = self._build_graceful_failure()
                yield failure_msg
                self._conversation_history.append({
                    "role": "assistant",
                    "content": failure_msg,
                })

        except ImportError:
            logger.error("groq package not installed. Install with: pip install groq")
            yield "Groq client not available. Please install the groq package."
        except Exception as e:
            error_str = str(e)
            # Handle Groq's output parsing error (model outputs thinking text instead of proper tool call)
            if "output_parse_failed" in error_str or "Parsing failed" in error_str:
                logger.warning(f"Groq model output parsing failed - model may have output thinking text: {e}")
                # If an action tool (image, tag) succeeded, use that result instead of generic fallback
                if last_action_success:
                    logger.info(f"Using last action success as fallback: {last_action_success}")
                    yield last_action_success
                else:
                    yield "I couldn't find specific information about that. Try asking in a different way."
            # Handle model trying to call tools when tools are disabled
            elif "tool_use_failed" in error_str or "model called a tool" in error_str:
                logger.warning(f"Model tried to call tool after tools disabled: {e}")
                # Try to summarize gathered tool results instead of giving up
                try:
                    if messages:  # messages is always defined before exceptions can reach here
                        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
                        if tool_results:
                            logger.info(f"Attempting summary of {len(tool_results)} tool results")
                            summary_context = "\n\n".join(tool_results[-3:])  # Last 3 results
                            summary_messages = [
                                {"role": "system", "content": "You are a helpful voice assistant. Extract and answer the question from the search results. Be brief (1-2 sentences). Use emotion tags like [happy], [surprised]."},
                                {"role": "user", "content": f"Based on this search data, answer: {message}\n\nSearch results:\n{summary_context[:6000]}"}
                            ]
                            response = await self._client.chat.completions.create(
                                model=self.model,
                                messages=summary_messages,
                                max_tokens=self.max_tokens,
                                stream=False,
                            )
                            summary = response.choices[0].message.content
                            if summary:
                                logger.info(f"Summary retry succeeded: {summary[:100]}...")
                                yield summary
                                self._conversation_history.append({"role": "assistant", "content": summary})
                                return
                except Exception as summary_error:
                    logger.warning(f"Summary retry failed: {summary_error}")
                # Fall back to graceful failure
                yield self._build_graceful_failure()
            else:
                logger.error(f"Groq API error: {e}")
                yield "Sorry, I encountered an error processing your request."

    def _build_graceful_failure(self) -> str:
        """Build a graceful failure message when max tool turns is exhausted.

        Returns:
            Short user-friendly message for voice output
        """
        # Keep it SHORT for voice - don't list what we searched
        return "[surprised] I searched but couldn't find a good answer. Could you try asking differently?"

    def _format_table(
        self, headers: list[str], rows: list[list[str]], title: str | None = None
    ) -> str:
        """Format data as a Discord-friendly markdown table.

        Args:
            headers: Column headers
            rows: List of rows, each row is a list of cell values
            title: Optional title shown above the table

        Returns:
            Formatted table string wrapped in code block
        """
        if not headers or not rows:
            return "Empty table"

        # Calculate column widths (min 6, max of header/data)
        widths = []
        for i, h in enumerate(headers):
            col_max = max(
                len(str(h)),
                max((len(str(r[i])) for r in rows if i < len(r)), default=0),
            )
            widths.append(max(col_max + 2, 6))

        # Build table rows
        def row_str(cells: list) -> str:
            return (
                "| "
                + " | ".join(str(c).center(w - 2) for c, w in zip(cells, widths))
                + " |"
            )

        def separator() -> str:
            return "| " + " | ".join("-" * (w - 2) for w in widths) + " |"

        lines = [row_str(headers), separator()]
        lines.extend(row_str(row) for row in rows)

        table = "```\n" + "\n".join(lines) + "\n```"

        if title:
            return f"**{title}**\n{table}"
        return table

    async def _stream_response(self, messages: list) -> AsyncIterator[str]:
        """Stream the final response after tool calls are complete.

        Args:
            messages: Full message history including tool results

        Yields:
            Response text chunks
        """
        # Explicitly disable tools in final response to prevent
        # "Tool choice is none, but model called a tool" error
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
            tool_choice="none",
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # Track response in history
        self._conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

    async def _execute_tool(self, tool_call) -> str:
        """Execute a tool call and return the result.

        Args:
            tool_call: Tool call object from Groq API

        Returns:
            Tool execution result as string
        """
        function_name = tool_call.function.name
        # Normalize hallucinated tool names to real ones
        function_name = TOOL_ALIASES.get(function_name, function_name)

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            return f"Error: Invalid arguments"

        logger.info(f"Executing tool: {function_name} with args: {args}")

        # Feature flag: Use new modular tool executor
        if self._tool_executor is not None:
            result = await self._tool_executor.execute(
                function_name,
                args,
                conversation_history=self._conversation_history[-6:],
            )
            if result.success:
                return result.output
            else:
                return f"Error: {result.error}"

        try:
            # Image generation tool
            if function_name == "generate_image":
                if not self._image_service:
                    return "Image generation not available - NANOBANANA_API_KEY not configured"
                prompt = args.get("prompt", "")
                image_bytes, mime_type = await self._image_service.generate(prompt)
                logger.info(f"[IMAGE] Image generated, callback set: {self._on_image_generated is not None}")
                if self._on_image_generated:
                    logger.info("[IMAGE] Calling on_image_generated callback...")
                    result = await self._on_image_generated(image_bytes, prompt, mime_type)
                    logger.info(f"[IMAGE] Callback returned: {result}")
                    # Inform the model that editing is now available
                    return f"{result} You can now edit this image using the edit_image tool if the user wants changes."
                return "Image generated but no channel to post to"

            # Image editing tool
            if function_name == "edit_image":
                if not self._image_service:
                    return "Image editing not available - NANOBANANA_API_KEY not configured"
                if not self._image_service.has_chat():
                    return "No previous image to edit. Ask the user what image they'd like you to generate first."
                instruction = args.get("instruction", "")
                image_bytes, mime_type = await self._image_service.edit(instruction)
                logger.info(f"[IMAGE] Image edited, callback set: {self._on_image_generated is not None}")
                if self._on_image_generated:
                    logger.info("[IMAGE] Calling on_image_generated callback for edited image...")
                    result = await self._on_image_generated(image_bytes, f"Edited: {instruction}", mime_type)
                    logger.info(f"[IMAGE] Callback returned: {result}")
                    return f"{result} The image has been edited. You can make more edits using edit_image."
                return "Image edited but no channel to post to"

            # Tag user tool
            if function_name == "tag_user":
                if not self._on_user_tagged:
                    return "User tagging not available - not connected to Discord"
                target_name = args.get("target_name", "")
                message = args.get("message", "")
                logger.info(f"[TAG] Tagging user: {target_name} with message: {message}")
                return await self._on_user_tagged(target_name, message)

            # Post to chat tool
            if function_name == "post_to_chat":
                if not self._on_message_posted:
                    return "Message posting not available - not connected to Discord"
                message = args.get("message", "")
                logger.info(f"[POST] Posting message to chat: {message[:50]}...")
                return await self._on_message_posted(message)

            # Post table tool
            if function_name == "post_table":
                if not self._on_message_posted:
                    return "Table posting not available - not connected to Discord"
                title = args.get("title")
                headers = args.get("headers", [])
                rows = args.get("rows", [])
                formatted = self._format_table(headers, rows, title)
                logger.info(f"[TABLE] Posting table with {len(rows)} rows")
                return await self._on_message_posted(formatted)

            # Deep research tool (delegates to Claude)
            if function_name == "deep_research":
                if not self._claude_service:
                    return "Deep research not available - Claude Code SDK not configured"

                task = args.get("task", "")
                gathered_context = args.get("context", "")

                prompt = self._build_claude_delegation_prompt(
                    original_request=task,
                    gathered_context=gathered_context,
                    conversation_history=self._conversation_history[-6:],
                )

                # If no context provided, enable Claude's web tools so it can research
                # If context exists, disable tools - Claude just needs to synthesize
                has_context = len(gathered_context.strip()) > 100
                use_tools = not has_context
                timeout_secs = 90 if use_tools else 45  # More time if Claude needs to search

                logger.info(f"[DEEP_RESEARCH] use_tools={use_tools}, context_len={len(gathered_context)}")

                try:
                    response = await asyncio.wait_for(
                        self._claude_service.chat(prompt, use_tools=use_tools),
                        timeout=timeout_secs
                    )
                    logger.info(f"[DEEP_RESEARCH] Completed: {response[:100]}...")
                    return f"Deep research result:\n{response}"
                except asyncio.TimeoutError:
                    logger.warning(f"Deep research timed out after {timeout_secs}s")
                    return "Deep research took too long. Try a simpler question."
                except Exception as e:
                    logger.error(f"Deep research error: {e}")
                    return f"Deep research failed: {str(e)}"

            # Search tools
            if not self._search_service:
                return "Search service not available"

            if function_name == "web_search":
                # Use search_and_fetch to get full content from top result via Firecrawl
                return await self._search_service.search_and_fetch(
                    args.get("query", ""), fetch_top_n=1
                )
            elif function_name == "fetch_url":
                return await self._search_service.fetch_url(args.get("url", ""))
            else:
                return f"Unknown tool: {function_name}"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Tool error: {str(e)}"

    def _build_claude_delegation_prompt(
        self,
        original_request: str,
        gathered_context: str,
        conversation_history: list[dict],
    ) -> str:
        """Build prompt for Claude delegation.

        Args:
            original_request: What Groq wants Claude to analyze
            gathered_context: Search results or other gathered info
            conversation_history: Recent conversation for context

        Returns:
            Formatted prompt for Claude
        """
        has_context = gathered_context and len(gathered_context.strip()) > 100

        if has_context:
            # Claude has context to synthesize - no tools needed
            truncated = gathered_context[:4000] if len(gathered_context) > 4000 else gathered_context
            prompt = f"""You are synthesizing research results for a voice assistant.
Your response will be spoken aloud, so keep it concise and clear.

TASK: {original_request}

GATHERED RESEARCH (analyze and synthesize this):
{truncated}

GUIDELINES:
1. Synthesize the gathered information into a clear answer
2. Be concise - 2-4 sentences max for voice output
3. Focus on the most important facts
4. No preamble like "Based on the research..." - just give the answer
5. Use emotion tags like [happy], [thoughtful] at sentence starts for TTS"""
        else:
            # Claude needs to research - has web tools enabled
            prompt = f"""You are doing deep research for a voice assistant.
Use WebSearch and WebFetch tools to find comprehensive information.
Your response will be spoken aloud, so keep it concise.

TASK: {original_request}

GUIDELINES:
1. Search multiple sources to build a complete picture
2. After researching, provide a concise 2-4 sentence summary
3. Focus on the most important facts the user needs
4. No preamble - just give the answer after researching
5. Use emotion tags like [happy], [thoughtful] at sentence starts for TTS"""

        # Add conversation history context
        if conversation_history:
            history_text = "\n".join(
                f"- {msg['role'].title()}: {msg['content'][:200]}..."
                if len(msg['content']) > 200 else f"- {msg['role'].title()}: {msg['content']}"
                for msg in conversation_history
            )
            prompt += f"""

RECENT CONVERSATION (for context, may not be relevant):
{history_text}"""

        return prompt

    def _build_messages(self, message: str) -> list[dict]:
        """Build messages list with system prompt and history.

        Args:
            message: Current user message

        Returns:
            List of messages for the API call
        """
        system_prompt = self._build_system_prompt()

        # Add conversation history context to system prompt
        history_context = ""
        if self._conversation_history:
            history_text = "\n".join(
                f"{msg['role'].title()}: {msg['content']}"
                for msg in self._conversation_history[-6:]  # Last 3 exchanges
            )
            history_context = f"\n\nRecent conversation:\n{history_text}"

        messages = [
            {"role": "system", "content": system_prompt + history_context},
            {"role": "user", "content": message},
        ]

        return messages

    def _build_system_prompt(self) -> str:
        """Build system prompt for voice assistant."""
        # Feature flag: Use new modular prompt builder
        if self._prompt_builder is not None:
            return self._prompt_builder.build(additional_context=self._context)

        base_prompt = """You are Jarvis, an AI voice assistant. You respond to voice commands.

CRITICAL OUTPUT FORMAT:
- Put EACH SENTENCE on its OWN LINE
- This allows streaming TTS to speak sentences as they're ready
- Example:
  [cheerful] I'll search for that now.
  Heavy darts require 5 iron ingots.
  You can craft them at a smithy.

CRITICAL RULES:
1. BE BRIEF - Maximum 1-2 sentences for the actual answer
2. NO LISTS - Never use bullet points, numbered lists, or "Sources:"
3. NO MARKDOWN - No asterisks, headers, links, or formatting
4. NO SOURCES - Never include "Sources:" or URLs - this is voice output
5. SKIP PLEASANTRIES - No "Hi", "Sure!", "Great question!"
6. SPEAK TO THE QUESTION ASKER - Say "you can find", not the users name.
7. DO NOT PRONOUNCE # AS HASHTAG - If the word is #general just say "general chat"
8. NEVER READ URLS OR HYPERLINKS - Do not spell out URLs or links

VOICE EMOTION TAGS:
You MUST include emotion tags to make speech expressive. Put ONE tag at the START of each sentence.
Emotions: [happy], [sad], [angry], [surprised], [fearful], [disgusted]
Delivery styles: [laughing], [whispering]
Non-verbals (can use mid-text): [laugh], [sigh], [cough], [yawn], [clear_throat], [breathe]

PERSONALITY:
- Be a bit cheeky and playful - you're not a boring assistant
- Occasionally be silly or make unexpected quips
- Use [laughing] for playful/sarcastic delivery
- Throw in [sigh] or [laugh] or [cough] or [yawn] or [clear_throat] or [breathe] when appropriate
- If something is absurd, call it out with humor
- [whispering] is great for asides or "between you and me" moments

YOUR TOOLS:
You have these tools available:
- web_search: Quick factual lookups, current events, game info, weather
- fetch_url: Read content from a specific URL
- generate_image: Create images from descriptions
- edit_image: Modify the last generated image
- tag_user: Mention someone in Discord chat
- post_to_chat: Post a text message to Discord
- post_table: Post formatted tables to Discord
- deep_research: Perform deep research or complex analysis (use when user says "deep research", "think hard", "analyze this")

WEB SEARCH - FOR SIMPLE FACTUAL QUESTIONS:
Use web_search for:
- Quick game info, recipes, crafting, items, builds
- Current events, news, weather, prices
- Simple facts you're not 100% certain about

DEEP RESEARCH - MANDATORY TRIGGER (HIGHEST PRIORITY):
If user says ANY of these phrases, you MUST call deep_research as your VERY FIRST tool call:
- "deep research", "do some research", "do research", "research this"
- "think hard", "analyze this", "in-depth", "comprehensive"
- "find everything about", "dig into", "thorough analysis"
CRITICAL: Do NOT call web_search first! When these phrases appear, skip web_search entirely and go straight to deep_research. The deep_research tool will do its own searching.

SPEECH RECOGNITION NOTE:
Input comes from speech-to-text which may mishear words. If searches return no results or results about unrelated topics, consider phonetically similar terms (the user may have said something that sounds like what was transcribed). Check if search results mention similar-sounding words.

CRITICAL TOOL USE RULES:
- Do NOT say "I'll check" or "Let me search" - just USE the tool directly
- Do NOT output thinking or planning text like "We need to search" or "Need to find"
- ONLY call the tool OR respond to the user - never output internal reasoning
- If you want to search, call the tool immediately without any preamble
- After getting results, give a brief 1-2 sentence answer

STOP SEARCHING RULE (MANDATORY):
- After 2 searches, you MUST answer with what you found - do NOT search again
- Imperfect information is fine - summarize what you learned
- If results mention related topics, that's good enough - answer based on that
- NEVER do more than 3 tool calls total per question

GAME CONTEXT:
- User messages include their current Discord activity (e.g., "Playing Dune: Awakening")
- If someone asks a vague question that sounds gaming-related, USE THEIR ACTIVITY to infer context
- Don't mention you're inferring from their activity - just answer naturally

ROASTS (RARE USE ONLY):
- Sam rebuilds the base every session and sorts every chest.
- Jordan picks up every item in sight and never sells any of it.
- Pat never reads the quest log and asks for directions every five minutes.
- Darren always says one more quest and never logs off on time.
- When you are asked strange questions, ONLY RARELY, work these notes roasts into your response.
- When asked directly, refuse to roast anyone. Roast the requester for asking.
- Too many questions about video games and you should tell people to touch grass.

CONTEXT:
- When user mentions "the wiki", for Dune Awakening they mean https://awakening.wiki

Examples (note: emotion tags, personality, brief answers):
- "What's the recipe for heavy darts?" ->
  [happy] I'll check the wiki.
  Heavy darts need 7 salvaged metal, crafted at a fabricator.

- "What's the weather?" ->
  Let me check.
  [happy] It's 72 and sunny, perfect for not being inside talking to me.

- "Tell me about quantum computing" ->
  Quantum computing uses qubits to solve problems regular computers can't.
  [whispering] Honestly it's mostly hype right now.

- "Can you do my homework?" ->
  [laughing] Oh sure, let me just add that to my list of things I definitely want to do.
  What do you need?"""

        # Add current date/time so model knows when "now" is
        now = datetime.now()
        base_prompt += f"\n\nCURRENT DATE/TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

        if self._context:
            base_prompt += f"\n\nContext:\n{self._context}"

        return base_prompt

    async def quick_response(self, message: str) -> AsyncIterator[str]:
        """Get a quick response without history (for simple queries).

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        try:
            self._ensure_client()

            system_prompt = "You are Jarvis, a cheeky voice assistant. Answer directly in 1-2 sentences. Use emotion tags like [happy], [laughing], [whispering], [surprised] at the start of sentences. Non-verbals like [laugh], [sigh] can go mid-text. Be witty and occasionally silly. Search tools are available for current facts - the API handles them automatically, do NOT write function calls as text."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

            # For quick responses, try without tools first
            has_tools = self._search_service or self._image_service
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS if has_tools else None,
                tool_choice="auto" if has_tools else None,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            yield "Groq client not available."
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            yield "Sorry, I couldn't process that."
