"""Response generation stage using ChatOrchestrator.

Simple wrapper around ChatOrchestrator that:
- Builds prompts with speaker context
- Yields streaming response chunks
"""

import logging
from typing import AsyncIterator, Optional

from ..llm.chat_orchestrator import ChatOrchestrator
from .base import PipelineContext, StreamingPipelineStage

logger = logging.getLogger(__name__)


class ResponseStage(StreamingPipelineStage):
    """LLM response generation stage.

    Wraps ChatOrchestrator to provide streaming responses
    with speaker context from PipelineContext.

    Example:
        stage = ResponseStage(orchestrator, system_prompt)

        async for chunk in stage.process_stream(text, context):
            print(chunk, end="")
    """

    def __init__(
        self,
        orchestrator: ChatOrchestrator,
        system_prompt: str,
    ):
        """Initialize response stage.

        Args:
            orchestrator: Chat orchestrator for LLM interaction
            system_prompt: Base system prompt for the assistant
        """
        self._orchestrator = orchestrator
        self._system_prompt = system_prompt
        self._messages: list[dict] = []
        self._on_message_update: Optional[callable] = None

    @property
    def name(self) -> str:
        """Stage name for logging and debugging."""
        return "response"

    def set_message_callback(self, callback: callable) -> None:
        """Set callback for message updates.

        Args:
            callback: Function called when messages list is modified
        """
        self._on_message_update = callback

    def reset_conversation(self) -> None:
        """Reset conversation history to start fresh."""
        self._messages = []
        logger.debug("Conversation history reset")

    def get_messages(self) -> list[dict]:
        """Get current message history.

        Returns:
            Copy of current messages list
        """
        return list(self._messages)

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
        # Build user message with speaker context
        user_message = self._build_user_message(text, context)

        # Initialize messages with system prompt if empty
        if not self._messages:
            self._messages.append({
                "role": "system",
                "content": self._build_system_prompt(context),
            })

        # Add user message
        self._messages.append({"role": "user", "content": user_message})

        # Stream response
        response_chunks: list[str] = []
        async for chunk in self._orchestrator.chat_stream(
            message=text,
            messages=self._messages,
            on_message_update=self._on_message_update,
        ):
            response_chunks.append(chunk)
            yield chunk

        # Add assistant response to history
        full_response = "".join(response_chunks)
        if full_response:
            self._messages.append({"role": "assistant", "content": full_response})

    def _build_system_prompt(self, context: PipelineContext) -> str:
        """Build system prompt with speaker context.

        Args:
            context: Pipeline context with speaker info

        Returns:
            System prompt with any speaker-specific context
        """
        prompt = self._system_prompt

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
