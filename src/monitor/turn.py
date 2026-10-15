"""Per-turn records from the live voice session.

VoiceSession and VoicePipeline call a TurnRecorder at a few points (speech end,
transcription, each clip played, reply finished, turn end); a logging handler
adds the LLM timings and any warnings/errors logged while a turn is open. When
the turn ends the recorder hands one LiveTurn to its callback.

Every hook is defensive: a monitor bug must never break the voice path.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import numpy as np

from ..audio import SAMPLE_RATE_TTS
from ..llm.chat_orchestrator import ERROR_REPLY, GRACEFUL_FAILURE_REPLY
from .text_checks import strip_tags

logger = logging.getLogger(__name__)

# Turn outcomes
ANSWERED = "answered"
ERROR_REPLY_OUTCOME = "error_reply"
GRACEFUL_FAILURE = "graceful_failure"
NO_REPLY = "no_reply"
WAKE_WORD_ONLY = "wake_word_only"
EMPTY = "empty"
CANCELLED = "cancelled"
NOT_ADDRESSED = "not_addressed"  # follow-up without the wake word, not for the bot
LATE_REPLY = "late_reply"  # not a turn: a reply that came after its window closed
CRASHED = "exception"
HUNG = "hung"

# The sink's last packet counts as the end of speech only if it is this recent
_SPEECH_END_MAX_LAG_S = 5.0
_GRACEFUL_TEXT = strip_tags(GRACEFUL_FAILURE_REPLY)


@dataclass
class Clip:
    """One piece of audio the bot played during a turn. Times are from speech end."""

    start: float
    seconds: float  # audio length
    end: float | None = None  # None if playback never reported finishing
    answer: bool = False  # part of the reply (not ack/affirmation/thinking phrase)

    @property
    def finish(self) -> float:
        return self.end if self.end is not None else self.start + self.seconds


@dataclass
class ToolUse:
    name: str
    args: str  # short summary (search query, domains, ...); never page content
    ok: bool = True
    error: str = ""  # first line of the error the model was given
    no_results: bool = False


@dataclass
class LogEvent:
    at: float  # seconds from speech end
    level: str
    logger: str
    message: str


@dataclass
class LiveTurn:
    """Everything observed about one voice turn."""

    started: datetime
    speaker_id: int | None = None
    speaker_name: str | None = None
    heard: str = ""  # question after wake-word stripping
    raw_heard: str = ""
    reply: str = ""  # all text sent to TTS (error replies included)
    reply_ok: bool = False  # the LLM turn completed normally
    outcome: str = ANSWERED
    error: str = ""  # exception text for crashed turns
    tools: list[ToolUse] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    stt_s: float | None = None
    llm_calls: list[dict] = field(default_factory=list)  # one per model turn (cerebras_client)
    tts_s: list[float] = field(default_factory=list)  # Inworld synthesis times
    events: list[LogEvent] = field(default_factory=list)  # warnings and errors
    duration_s: float = 0.0
    idle_s: float = 0.0  # hung turns: seconds with no progress
    last_step: str = ""  # hung turns: what it was waiting on
    hang_reported: bool = False
    spoken: str | None = None  # Whisper transcript of the answer audio (MONITOR_RETRANSCRIBE)
    note: str = ""  # incidents (LATE_REPLY): what happened
    answer_audio: list[np.ndarray] = field(default_factory=list, repr=False)
    # The question as Whisper got it (16 kHz), kept when MONITOR_SAVE_AUDIO is on
    question_audio: np.ndarray | None = field(default=None, repr=False)
    audio_file: str = ""  # where it was saved (shown in the post)

    @property
    def first_audio_s(self) -> float | None:
        return round(self.clips[0].start, 2) if self.clips else None

    @property
    def answer_audio_s(self) -> float | None:
        answer = [c for c in self.clips if c.answer]
        return round(answer[0].start, 2) if answer else None

    @property
    def answer_gaps_s(self) -> list[float]:
        """Silences between consecutive reply clips (sentence to sentence)."""
        answer = [c for c in self.clips if c.answer]
        return [round(b.start - a.finish, 2) for a, b in zip(answer, answer[1:])]

    @property
    def used_web(self) -> bool:
        return any(t.name.startswith("web_") for t in self.tools)


def _safe(method):
    """Hooks run inside the voice path: log and swallow any monitor error."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.debug(f"[MONITOR] {method.__name__} failed: {e}", exc_info=True)
            return None
    return wrapper


class _TurnLogHandler(logging.Handler):
    def __init__(self, recorder: TurnRecorder):
        super().__init__(logging.INFO)
        self._recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(__package__ or "src.monitor"):
            return  # our own logs (and no recursion)
        try:
            self._recorder._on_log(record)
        except Exception:
            pass


class TurnRecorder:
    """Builds a LiveTurn per voice turn and passes it to on_turn when the turn ends."""

    def __init__(
        self,
        on_turn: Callable[[LiveTurn], Any],
        *,
        retain_audio: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """
        Args:
            on_turn: Called (on the event loop) with each finished turn, and with a
                snapshot of a turn that has stopped making progress (check_hung)
            retain_audio: Keep the reply audio on the turn (for re-transcription)
            clock: perf_counter-compatible time source
        """
        self._on_turn = on_turn
        self._retain_audio = retain_audio
        self._clock = clock
        self._lock = threading.RLock()
        self._turn: LiveTurn | None = None
        self._token = 0
        self._t0 = 0.0  # speech end (perf time)
        self._last_activity = 0.0
        self._pipeline_result: str | None = None
        self._answer_next = False
        self._seen_response: Any = None
        self._handler: _TurnLogHandler | None = None
        # MONITOR_SAVE_AUDIO: keep question audio; wake word clips go to on_wake_clip
        self.keep_question_audio = False
        self.on_wake_clip: Callable[[int, np.ndarray, float, bool], None] | None = None

    # ------------------------------------------------------------ setup

    def install(self, target: logging.Logger | None = None) -> None:
        """Start collecting log records (root logger by default)."""
        if self._handler is None:
            self._handler = _TurnLogHandler(self)
            (target or logging.getLogger()).addHandler(self._handler)

    def uninstall(self, target: logging.Logger | None = None) -> None:
        if self._handler is not None:
            (target or logging.getLogger()).removeHandler(self._handler)
            self._handler = None

    @property
    def turn_open(self) -> bool:
        return self._turn is not None

    def _rel(self, t: float | None = None) -> float:
        return round((self._clock() if t is None else t) - self._t0, 3)

    # ------------------------------------------------------------ hooks (VoiceSession)

    @_safe
    def turn_started(self, speaker_id: int | None, speaker_name: str | None,
                     speech_end: float | None = None) -> int:
        """A captured question is about to be processed. Returns a token for turn_ended.

        speech_end: perf time of the speaker's last audio packet, if known
        """
        now = self._clock()
        with self._lock:
            # A turn started inside another (wake word only -> follow-up question)
            # replaces it; the outer one had nothing to report.
            self._token += 1
            self._t0 = now
            if speech_end is not None and 0 <= now - speech_end <= _SPEECH_END_MAX_LAG_S:
                self._t0 = speech_end
            self._last_activity = now
            self._pipeline_result = None
            self._answer_next = False
            self._turn = LiveTurn(started=datetime.now(), speaker_id=speaker_id,
                                  speaker_name=speaker_name)
            return self._token

    @_safe
    def question_audio(self, audio: np.ndarray) -> None:
        """The audio of the question being processed (16 kHz mono float32)."""
        with self._lock:
            if self._turn is not None and self.keep_question_audio:
                self._turn.question_audio = audio

    @_safe
    def wake_clip(self, user_id: int, audio: np.ndarray, score: float, hit: bool) -> None:
        """Audio around a wake word hit or near miss (called from the audio thread)."""
        if self.on_wake_clip is not None:
            self.on_wake_clip(user_id, audio, score, hit)

    @_safe
    def clip_started(self, audio: np.ndarray) -> Clip | None:
        """The bot started playing audio (24 kHz mono). Returns a handle for clip_ended."""
        with self._lock:
            if self._turn is None:
                return None
            now = self._clock()
            clip = Clip(start=self._rel(now), seconds=round(len(audio) / SAMPLE_RATE_TTS, 3),
                        answer=self._answer_next)
            self._answer_next = False
            self._turn.clips.append(clip)
            self._last_activity = now + clip.seconds
            if clip.answer and self._retain_audio:
                self._turn.answer_audio.append(audio)
            return clip

    @_safe
    def clip_ended(self, clip: Clip | None) -> None:
        if clip is None:
            return
        with self._lock:
            now = self._clock()
            clip.end = self._rel(now)
            self._last_activity = max(self._last_activity, now)

    @_safe
    def turn_ended(self, token: int | None, error: BaseException | None = None) -> None:
        with self._lock:
            if self._turn is None or token != self._token:
                return  # replaced by a nested turn, or already closed
            turn = self._turn
            self._turn = None
            turn.duration_s = self._rel()
            turn.outcome = self._outcome(turn, error)
        self._emit(turn)

    @_safe
    def incident(self, speaker_id: int | None, speaker_name: str | None, outcome: str,
                 note: str, reply: str = "") -> None:
        """Something worth reporting that happened outside a turn (LATE_REPLY)."""
        self._emit(LiveTurn(started=datetime.now(), speaker_id=speaker_id,
                            speaker_name=speaker_name, reply=reply, outcome=outcome,
                            note=note))

    # ------------------------------------------------------------ hooks (VoicePipeline)

    @_safe
    def transcribed(self, raw_text: str, text: str) -> None:
        with self._lock:
            if self._turn is None:
                return
            self._turn.raw_heard = raw_text or ""
            self._turn.heard = text or ""
            self._turn.stt_s = self._rel()
            self._last_activity = self._clock()

    @_safe
    def answer_clip_next(self) -> None:
        """The next clip played is part of the reply."""
        self._answer_next = True

    @_safe
    def replied(self, response_turn: Any) -> None:
        """The reply finished streaming (or was cut short).

        response_turn: ResponseStage.last_turn (text, ok, messages)
        """
        with self._lock:
            if self._turn is None or response_turn is None:
                return
            if response_turn is self._seen_response:
                return  # left over from an earlier turn: the LLM never ran this time
            self._seen_response = response_turn
            self._turn.reply = getattr(response_turn, "text", "") or ""
            self._turn.reply_ok = bool(getattr(response_turn, "ok", False))
            self._turn.tools = tools_from_messages(getattr(response_turn, "messages", []))
            self._last_activity = self._clock()

    @_safe
    def pipeline_result(self, result: str | None) -> None:
        """VoicePipeline.process returned (None, "wake_word_only", "empty", "cancelled",
        "not_addressed")."""
        with self._lock:
            if self._turn is not None:
                self._pipeline_result = result

    # ------------------------------------------------------------ logging

    def _on_log(self, record: logging.LogRecord) -> None:
        with self._lock:
            turn = self._turn
            if turn is None:
                return
            if record.name.startswith("src"):
                self._last_activity = max(self._last_activity, self._clock())
            llm = getattr(record, "llm_turn", None)
            if isinstance(llm, dict):
                turn.llm_calls.append(dict(llm))
            tts = getattr(record, "tts_seconds", None)
            if isinstance(tts, (int, float)):
                turn.tts_s.append(round(float(tts), 2))
            if record.levelno >= logging.WARNING:
                turn.events.append(LogEvent(
                    at=self._rel(), level=record.levelname, logger=record.name,
                    message=record.getMessage()[:500],
                ))

    # ------------------------------------------------------------ hang watchdog

    @_safe
    def check_hung(self, hang_seconds: float) -> LiveTurn | None:
        """Report (once) a turn that has made no progress for hang_seconds.

        The snapshot goes to on_turn with outcome "hung"; the turn stays open and
        is still reported normally if it ever ends.
        """
        with self._lock:
            turn = self._turn
            if turn is None or turn.hang_reported:
                return None
            if self._pipeline_result == WAKE_WORD_ONLY:
                return None  # listening for the follow-up question, not stuck
            idle = self._clock() - self._last_activity
            if idle < hang_seconds:
                return None
            turn.hang_reported = True
            snapshot = replace(
                turn, clips=list(turn.clips), events=list(turn.events),
                llm_calls=list(turn.llm_calls), tools=list(turn.tools), answer_audio=[],
                outcome=HUNG, duration_s=self._rel(), idle_s=round(idle, 1),
                last_step=self._last_step(turn),
            )
        self._emit(snapshot)
        return snapshot

    def _last_step(self, turn: LiveTurn) -> str:
        if turn.stt_s is None:
            return "waiting for Whisper to transcribe the question"
        if not turn.llm_calls:
            return "waiting for the LLM's first response"
        last = turn.llm_calls[-1]
        if last.get("tool_calls"):
            return f"running tools after LLM call {len(turn.llm_calls)}"
        if not any(c.answer for c in turn.clips):
            return "LLM answered but no answer audio played (TTS or playback)"
        return f"after LLM call {len(turn.llm_calls)}"

    # ------------------------------------------------------------ helpers

    def _outcome(self, turn: LiveTurn, error: BaseException | None) -> str:
        if isinstance(error, asyncio.CancelledError):
            return CANCELLED
        if error is not None:
            turn.error = f"{type(error).__name__}: {error}"[:300]
            return CRASHED
        result = self._pipeline_result
        if result in (WAKE_WORD_ONLY, EMPTY, CANCELLED, NOT_ADDRESSED):
            return result
        if ERROR_REPLY in turn.reply:
            return ERROR_REPLY_OUTCOME
        if _GRACEFUL_TEXT and _GRACEFUL_TEXT in strip_tags(turn.reply):
            return GRACEFUL_FAILURE
        if not turn.reply.strip():
            return NO_REPLY
        return ANSWERED

    def _emit(self, turn: LiveTurn) -> None:
        try:
            self._on_turn(turn)
        except Exception as e:
            logger.warning(f"[MONITOR] Turn handler failed: {e}")


def _summarize_args(name: str, raw: str) -> str:
    try:
        args = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return "(unparseable arguments)"
    if not isinstance(args, dict):
        return ""
    if name == "web_search":
        return f'"{str(args.get("query", ""))[:100]}"'
    if name == "web_extract":
        urls = args.get("urls") or args.get("url") or []
        if isinstance(urls, str):
            urls = [urls]
        hosts = [urlparse(str(u)).netloc or str(u)[:40] for u in urls][:5]
        return ", ".join(hosts)
    return ", ".join(f"{k}={str(v)[:60]!r}" for k, v in list(args.items())[:3])


def tools_from_messages(messages: list[dict]) -> list[ToolUse]:
    """Tool calls and their outcomes from the assistant/tool messages of one turn."""
    calls: dict[str, ToolUse] = {}
    order: list[ToolUse] = []
    for msg in messages or []:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                use = ToolUse(name=fn.get("name", "?"),
                              args=_summarize_args(fn.get("name", ""), fn.get("arguments", "")))
                calls[tc.get("id", "")] = use
                order.append(use)
        elif msg.get("role") == "tool":
            use = calls.get(msg.get("tool_call_id", ""))
            content = (msg.get("content") or "").strip()
            if use is None:
                continue
            if content.startswith("Error:"):
                error = content[len("Error:"):].strip()
                use.ok = False
                use.error = (error.splitlines()[0] if error else "error")[:200]
            elif content.startswith("No results for"):
                use.no_results = True
    return order
