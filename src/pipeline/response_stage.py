"""Response generation stage using ChatOrchestrator.

Simple wrapper around ChatOrchestrator that:
- Builds prompts with speaker context
- Yields streaming response chunks
- Keeps a bounded conversation history
"""

import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

from ..llm.chat_orchestrator import ChatOrchestrator, TurnOutcome
from .base import PipelineContext, StreamingPipelineStage

logger = logging.getLogger(__name__)

# Tool results (web pages can be 15k chars each) stay whole in history for a
# short while so follow-ups ("what about tomorrow?") can use them, then are
# swapped for a note so they aren't re-sent with every later question.
TOOL_RESULT_TTL_SECONDS = 120.0
# Results this short (e.g. "Done! I've posted that") are cheap; keep them whole
SMALL_TOOL_RESULT_CHARS = 300


def _expired_note(tool_name: str, content: str) -> str:
    first_line = content.strip().splitlines()[0][:150] if content.strip() else ""
    return (f"[Earlier {tool_name} result removed to save space. It began: {first_line!r}. "
            f"Call {tool_name} again if you need the details.]")


@dataclass
class ResponseTurn:
    """What the latest process_stream call produced (read by the turn monitor)."""

    text: str = ""  # everything yielded to TTS, error replies included
    ok: bool = False  # the orchestrator finished the turn normally
    messages: list[dict] = field(default_factory=list)  # assistant/tool messages it added


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
        tool_result_ttl: float = TOOL_RESULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        game_notes=None,
    ):
        """Initialize response stage.

        Args:
            orchestrator: Chat orchestrator for LLM interaction
            build_system_prompt: Returns the base system prompt. Called every turn
                so the date/time is current.
            max_history_messages: History cap; oldest whole exchanges are dropped
            tool_result_ttl: Seconds a large tool result stays whole in history
            clock: Time source (tests)
            game_notes: Optional src.llm.game_notes.GameNotes; official notes for a
                game in play are added to the system prompt
        """
        self._orchestrator = orchestrator
        self._build_base_prompt = build_system_prompt
        self._max_history = max_history_messages
        # Conversation without the system prompt (rebuilt each turn)
        self._history: list[dict] = []
        self._history_before: list[dict] | None = None  # for forget_last_turn
        self._tool_result_ttl = tool_result_ttl
        self._clock = clock
        # tool_call_id -> when its (large) result entered history
        self._tool_result_times: dict[str, float] = {}
        self.last_turn: Optional[ResponseTurn] = None
        self._game_notes = game_notes

    @property
    def name(self) -> str:
        """Stage name for logging and debugging."""
        return "response"

    def reset_conversation(self) -> None:
        """Reset conversation history to start fresh."""
        self._history = []
        self._tool_result_times = {}
        logger.debug("Conversation history reset")

    def earlier_lookups(self) -> str:
        """Web lookup results still whole in history from turns before the latest one."""
        from ..llm.chat_post import lookup_results

        users = [i for i, m in enumerate(self._history) if m.get("role") == "user"]
        return lookup_results(self._history[:users[-1]]) if users else ""

    def forget_last_turn(self) -> bool:
        """Drop the latest exchange from history: it was cancelled before anyone heard
        the answer (the speaker kept talking). True if there was one to drop."""
        if self._history_before is None:
            return False
        self._history, self._history_before = self._history_before, None
        return True

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
        self._expire_tool_results()
        self._history_before = self._history
        messages = [
            {"role": "system", "content": self._build_system_prompt(context, text)},
            *self._history,
            {"role": "user", "content": self._build_user_message(text, context)},
        ]

        outcome = TurnOutcome()
        turn = self.last_turn = ResponseTurn()
        async for chunk in self._orchestrator.chat_stream(messages, outcome):
            turn.text += chunk
            yield chunk
        turn.ok = outcome.ok
        turn.messages = messages[len(self._history) + 2:]  # after system, history, user

        # Only keep turns that completed; a failed turn may end mid tool-loop
        if outcome.ok:
            now = self._clock()
            for msg in messages[len(self._history) + 1:]:
                big = len(msg.get("content") or "") > SMALL_TOOL_RESULT_CHARS
                if msg.get("role") == "tool" and big:
                    self._tool_result_times[msg["tool_call_id"]] = now
            self._history = self._trim(messages[1:])
        else:
            logger.info("Turn did not complete, not adding it to history")

    def _expire_tool_results(self) -> None:
        """Replace large tool results older than the TTL with a short note."""
        now = self._clock()
        expired = {cid for cid, t in self._tool_result_times.items()
                   if now - t >= self._tool_result_ttl}
        if not expired:
            return
        names = {tc["id"]: tc["function"]["name"]
                 for m in self._history for tc in (m.get("tool_calls") or [])}
        for i, msg in enumerate(self._history):
            cid = msg.get("tool_call_id")
            if msg.get("role") == "tool" and cid in expired:
                tool_name = names.get(cid, "the tool")
                self._history[i] = {**msg, "content": _expired_note(tool_name, msg["content"])}
        for cid in expired:
            del self._tool_result_times[cid]
        logger.debug(f"Expired {len(expired)} tool result(s) from history")

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

    def _build_system_prompt(self, context: PipelineContext, text: str = "") -> str:
        """Build system prompt with speaker context.

        Args:
            context: Pipeline context with speaker info
            text: The user's words (a named game brings in its notes)

        Returns:
            System prompt with any speaker-specific context
        """
        prompt = self._build_base_prompt()

        # Add speaker activity if available
        if context.speaker_activity:
            prompt += f"\n\nThe user is currently: {context.speaker_activity}"

        if self._game_notes is not None:
            notes = self._game_notes.prompt_block(text, context.speaker_activity,
                                                  context.channel_activities)
            if notes:
                prompt += "\n\n" + notes

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
