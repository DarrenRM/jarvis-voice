"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    All LLM clients (Claude, Groq, etc.) must implement this interface
    to ensure compatibility with the voice assistant pipeline.
    """

    @abstractmethod
    def set_context(self, context: str) -> None:
        """Set conversation context (e.g., current game, channel info).

        Args:
            context: Context string to include in system prompt
        """
        pass

    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""
        pass

    @abstractmethod
    async def chat(self, message: str) -> str:
        """Send message and get complete response.

        Args:
            message: User message

        Returns:
            Complete response text
        """
        pass

    @abstractmethod
    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Send message and stream response chunks.

        This is the primary method used by the voice assistant for
        low-latency streaming TTS.

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        pass

    @abstractmethod
    async def quick_response(self, message: str) -> AsyncIterator[str]:
        """Get quick response without history (for simple queries).

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        pass
