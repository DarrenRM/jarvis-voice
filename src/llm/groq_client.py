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
            tool_registry: Tool registry for modular tool system (required)
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

        # Initialize modular tool system (required)
        if tool_registry is None:
            raise ValueError("tool_registry is required for GroqClient")

        from ..tools import ToolExecutor
        from .system_prompt_builder import create_default_prompt_builder
        from .chat_orchestrator import ChatOrchestrator

        self._tool_executor = ToolExecutor(tool_registry)
        self._prompt_builder = create_default_prompt_builder()
        self._chat_orchestrator = ChatOrchestrator(
            llm_client=self,
            tool_executor=self._tool_executor,
            max_turns=self.max_turns,
        )

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
    def chat_orchestrator(self) -> "ChatOrchestrator":
        """Get the ChatOrchestrator.

        Returns:
            ChatOrchestrator instance
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

        except ImportError:
            logger.error("groq package not installed. Install with: pip install groq")
            yield "Groq client not available. Please install the groq package."
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            yield "Sorry, I encountered an error processing your request."

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
        """Build system prompt for voice assistant.

        Returns:
            System prompt string from the prompt builder
        """
        return self._prompt_builder.build(additional_context=self._context)

    async def quick_response(self, message: str) -> AsyncIterator[str]:
        """Get a quick response without history (for simple queries).

        This method bypasses the tool loop for faster responses.
        Use chat_stream() for full tool support.

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        try:
            self._ensure_client()

            system_prompt = "You are Jarvis, a cheeky voice assistant. Answer directly in 1-2 sentences. Use emotion tags like [happy], [laughing], [whispering], [surprised] at the start of sentences. Non-verbals like [laugh], [sigh] can go mid-text. Be witty and occasionally silly."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
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
