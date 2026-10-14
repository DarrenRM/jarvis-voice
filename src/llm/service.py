"""LLM service: everything the voice session needs from the language model."""

import logging
from typing import TYPE_CHECKING, Optional

from .chat_orchestrator import ChatOrchestrator
from .system_prompt_builder import SystemPromptBuilder

if TYPE_CHECKING:
    from ..image import NanoBananaClient
    from ..tools import ToolExecutor
    from .cerebras_client import CerebrasClient

logger = logging.getLogger(__name__)


class LLMService:
    """Bundles the LLM client, tool loop, prompt builder and optional image service."""

    def __init__(
        self,
        client: "CerebrasClient",
        tool_executor: "ToolExecutor",
        prompt_builder: SystemPromptBuilder,
        image_service: Optional["NanoBananaClient"] = None,
        max_turns: int = 5,
    ):
        self._client = client
        self._prompt_builder = prompt_builder
        self.image_service = image_service
        self.chat_orchestrator = ChatOrchestrator(
            llm_client=client,
            tool_executor=tool_executor,
            max_turns=max_turns,
        )

    def build_system_prompt(
        self,
        speaker_name: Optional[str] = None,
        speaker_activity: Optional[str] = None,
    ) -> str:
        """Build the system prompt fresh (current date/time and speaker)."""
        return self._prompt_builder.build(
            speaker_name=speaker_name,
            speaker_activity=speaker_activity,
        )

    async def chat(self, message: str) -> str:
        """One-shot reply without tools or history (greetings, keep-alive).

        Raises on API errors so callers can skip speaking instead of voicing an error.
        """
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": message},
        ]
        return await self._client.complete(messages)
