"""Response generation stage using ChatOrchestrator.

Simple wrapper around ChatOrchestrator that:
- Builds prompts with speaker context
- Yields streaming response chunks
- Keeps a bounded conversation history
"""

import logging
from typing import AsyncIterator, Callable

from ..llm.chat_orchestrator import ChatOrchestrator, TurnOutcome
from .base import PipelineContext, StreamingPipelineStage

logger = logging.getLogger(__name__)


class ResponseStage(StreamingPipelineStage):
    """LLM response generation stage.

    Wraps ChatOrchestrator to provide streaming responses
    with speaker context from PipelineContext.

    Example:
        stage = ResponseStage(orchestrator, build_system_prompt)

        async for chunk in stage.process_stream(text, context):
            print(chunk, end="")
    """

    def __init__(
        self,
        orchestrator: ChatOrchestrator,
        build_system_prompt: Callable[[], str],
        max_history_messages: int = 60,
    ):
        """Initialize response stage.

        Args:
            orchestrator: Chat orchestrator for LLM interaction
            build_system_prompt: Returns the base system prompt. Called every turn
                so the date/time is current.
            max_history_messages: History cap; oldest whole exchanges are dropped
        """
        self._orchestrator = orchestrator
        self._build_base_prompt = build_system_prompt
        self._max_history = max_history_messages
        # Conversation without the system prompt (rebuilt each turn)
        self._history: list[dict] = []

    @property
    def name(self) -> str:
        """Stage name for logging and debugging."""
        return "response"

    def reset_conversation(self) -> None:
        """Reset conversation history to start fresh."""
        self._history = []
        logger.debug("Conversation history reset")

    def get_messages(self) -> list[dict]:
        """Get current message history (without system prompt)."""
        return list(self._history)

    async def process_stream(
        self,
        text: str,
        context: PipelineContext,
    ) -> AsyncIterator[str]:
        """Generate streaming response with speaker context.

        Args:
            text: User's transcribed text
            context: Pipeline context with speaker info

        Yields:
            Response text chunks
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt(context)},
            *self._history,
            {"role": "user", "content": self._build_user_message(text, context)},
        ]

        outcome = TurnOutcome()
        async for chunk in self._orchestrator.chat_stream(messages, outcome):
            yield chunk

        # Only keep turns that completed; a failed turn may end mid tool-loop
        if outcome.ok:
            self._history = self._trim(messages[1:])
        else:
            logger.info("Turn did not complete, not adding it to history")

    def _trim(self, history: list[dict]) -> list[dict]:
        """Drop oldest exchanges so history fits, cutting only at user messages.

        Cutting at a user message keeps each assistant tool_calls message
        together with its tool results.
        """
        if len(history) <= self._max_history:
            return history
        start = len(history) - self._max_history
        while start < len(history) and history[start].get("role") != "user":
            start += 1
        return history[start:]

    def _build_system_prompt(self, context: PipelineContext) -> str:
        """Build system prompt with speaker context.

        Args:
            context: Pipeline context with speaker info

        Returns:
            System prompt with any speaker-specific context
        """
        prompt = self._build_base_prompt()

        # Add speaker activity if available
        if context.speaker_activity:
            prompt += f"\n\nThe user is currently: {context.speaker_activity}"

        return prompt

    def _build_user_message(self, text: str, context: PipelineContext) -> str:
        """Build user message with speaker name prefix if available.

        Args:
            text: Transcribed user text
            context: Pipeline context with speaker info

        Returns:
            User message, optionally prefixed with speaker name
        """
        if context.speaker_name:
            return f"[{context.speaker_name}]: {text}"
        return text
