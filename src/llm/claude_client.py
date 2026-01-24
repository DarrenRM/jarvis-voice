"""Claude Code SDK client with streaming support (headless mode)."""

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

# Project directory (for local settings that disable hooks)
PROJECT_DIR = Path(__file__).parent.parent.parent
SETTINGS_FILE = PROJECT_DIR / ".claude" / "settings.local.json"

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Claude Code SDK client for voice assistant.

    Uses headless Claude Code instead of direct API access.
    No API key required - uses your authenticated Claude Code session.

    Features:
    - Streaming responses for low latency
    - Conversation context management
    - Voice-optimized responses
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_tokens: int = 1024,
    ):
        """Initialize Claude client.

        Args:
            model: Model to use (sonnet, opus, haiku)
            max_tokens: Maximum response tokens
        """
        self.model = model
        self.max_tokens = max_tokens
        self._context: str = ""
        self._conversation_history: list[dict] = []

    def set_context(self, context: str) -> None:
        """Set conversation context (e.g., current game, channel info).

        Args:
            context: Context string to include in system prompt
        """
        self._context = context

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()

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
        """Send message and stream response using Claude Code SDK.

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        try:
            from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock

            # Build the prompt with context
            system_prompt = self._build_system_prompt()

            # Add conversation history context
            history_context = ""
            if self._conversation_history:
                history_text = "\n".join(
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in self._conversation_history[-6:]  # Last 3 exchanges
                )
                history_context = f"\n\nRecent conversation:\n{history_text}"

            # Track user message
            self._conversation_history.append({
                "role": "user",
                "content": message,
            })

            options = ClaudeCodeOptions(
                model=self.model,  # haiku for speed, sonnet for quality
                max_turns=10,  # Allow multiple tool uses + response
                system_prompt=system_prompt + history_context,
                cwd=PROJECT_DIR,
                settings=str(SETTINGS_FILE),  # Use local settings with no hooks
                allowed_tools=["WebSearch", "WebFetch"],  # Only allow web tools
            )

            full_response = ""
            logger.debug(f"Querying Claude with: {message[:100]}...")
            async for event in query(
                prompt=message,
                options=options,
            ):
                logger.debug(f"Received event: {type(event).__name__}")
                if isinstance(event, AssistantMessage):
                    for block in event.content:
                        if isinstance(block, TextBlock) and block.text:
                            full_response += block.text
                            yield block.text

            # Track assistant response
            self._conversation_history.append({
                "role": "assistant",
                "content": full_response,
            })

        except ImportError:
            logger.error(
                "claude-code-sdk not installed. Install with: pip install claude-code-sdk"
            )
            yield "Claude Code SDK not available. Please install it."
        except Exception as e:
            logger.error(f"Claude Code error: {e}")
            yield "Sorry, I encountered an error processing your request."

    def _build_system_prompt(self) -> str:
        """Build system prompt with context."""
        base_prompt = """You are Jarvis, an AI voice assistant. You respond to voice commands.

CRITICAL OUTPUT FORMAT:
- Put EACH SENTENCE on its OWN LINE
- This allows streaming TTS to speak sentences as they're ready
- Example:
  I'll search for that now.
  Heavy darts require 5 iron ingots.
  You can craft them at a smithy.

CRITICAL RULES:
1. BE BRIEF - Maximum 1-2 sentences for the actual answer
2. NO LISTS - Never use bullet points, numbered lists, or "Sources:"
3. NO MARKDOWN - No asterisks, headers, links, or formatting
4. NO SOURCES - Never include "Sources:" or URLs - this is voice output
5. SKIP PLEASANTRIES - No "Hi", "Sure!", "Great question!"
6. ONE ANNOUNCEMENT - Say "I'll check that" once, then just give the answer

RESPONSE STRATEGY:
- Web searches: Say "I'll look that up." ONCE, then give a 1-sentence answer
- Simple facts: One sentence answer
- Don't repeat or rephrase what you're doing

TOOLS:
- You have web search for current information
- Use it when questions need up-to-date facts

CONTEXT:
- When user mentions "the wiki" for Dune Awakening, they mean https://awakening.wiki

Examples (note: one announcement, then one short answer):
- "What's the recipe for heavy darts?" ->
  I'll check the wiki.
  Heavy darts need 7 salvaged metal, crafted at a fabricator.

- "What's the weather?" ->
  I'll look that up.
  It's 72 and sunny in Portland.

- "Tell me about quantum computing" ->
  Quantum computing uses qubits to solve problems regular computers can't.\""""

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
            from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock

            system_prompt = "You are Jarvis, a voice assistant. Answer immediately and directly. No greetings or filler. One or two sentences max. Use web search for current facts."

            options = ClaudeCodeOptions(
                model=self.model,
                max_turns=10,  # Allow multiple tool uses + response
                system_prompt=system_prompt,
                cwd=PROJECT_DIR,
                settings=str(SETTINGS_FILE),
                allowed_tools=["WebSearch", "WebFetch"],  # Only allow web tools
            )

            async for event in query(prompt=message, options=options):
                if isinstance(event, AssistantMessage):
                    for block in event.content:
                        if isinstance(block, TextBlock) and block.text:
                            yield block.text

        except ImportError:
            yield "Claude Code SDK not available."
        except Exception as e:
            logger.error(f"Claude Code error: {e}")
            yield "Sorry, I couldn't process that."
