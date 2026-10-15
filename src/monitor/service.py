"""Live monitor: detect problems in voice turns, post them to Discord, record approvals.

Flow: TurnRecorder -> handle_turn (optional re-transcription, detectors, dedupe)
-> one post per turn in the monitor channel with check/cross reactions. The owner's
check means "something went wrong here" (the post's diagnosis is only a guess) and
appends the post to the approved-fixes file; a cross means "no problem, or it didn't
disrupt the flow" and appends it to the dismissed-findings file. With
MONITOR_SAVE_AUDIO each question (and wake word clips) is also kept on disk
(recordings.py) and the post names the file.

Off unless MONITOR_ENABLED=true.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .detectors import Finding, Thresholds, detect
from .recordings import AudioArchive
from .reporting import (
    CONFIRMED,
    DISMISSED,
    POST_PREFIX,
    Redactor,
    approval_entry,
    format_post,
    recording_path,
    with_seen_count,
)
from .turn import HUNG, LiveTurn, TurnRecorder

logger = logging.getLogger(__name__)

APPROVE = "✅"
REJECT = "❌"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Rate limits: a recurring problem is one post with a count (MONITOR_DEDUPE_MINUTES);
# the count stops growing on an old post after REPOST_AFTER_S so it is seen again.
MAX_POSTS_PER_HOUR = 12
REPOST_AFTER_S = 6 * 3600
WATCHDOG_INTERVAL_S = 2.0

APPROVED_FILE_HEADER = (
    "# Approved fixes\n"
    "\n"
    "Turns the live monitor (src/monitor) posted to Discord that the owner marked with a\n"
    "check reaction: something really went wrong there. The \"What went wrong\" text is\n"
    "the monitor's guess and is sometimes wrong, so confirm the cause from the logs (and\n"
    "the recording, if there is one) before fixing. Newest at the bottom; the HTML\n"
    "comment on each entry keeps the same Discord message from being added twice.\n"
)

DISMISSED_FILE_HEADER = (
    "# Dismissed findings\n"
    "\n"
    "Monitor posts the owner marked with a cross: no problem happened, or it didn't\n"
    "disrupt the natural flow of the conversation (often just a slow reply). Not\n"
    "\"ignore\": worth fixing only when the fix has no tradeoffs, side effects or edge\n"
    "cases. Newest at the bottom.\n"
)


@dataclass
class PostedIssue:
    message: Any  # discord.Message (or a fake)
    content: str
    keys: set[str]
    posted_at: float
    posted_dt: datetime
    last_seen: float
    count: int = 1


class ApprovalLog:
    """Append-only markdown file of approved issues, one entry per Discord message."""

    def __init__(self, path: Path, header: str = APPROVED_FILE_HEADER):
        self.path = path
        self.header = header

    @staticmethod
    def marker(message_id: int) -> str:
        return f"<!-- discord-message:{message_id} -->"

    def contains(self, message_id: int) -> bool:
        if not self.path.exists():
            return False
        return self.marker(message_id) in self.path.read_text(encoding="utf-8")

    def add(self, message_id: int, entry: str) -> bool:
        """Append entry unless this message is already recorded. True if written."""
        if self.marker(message_id) not in entry:
            raise ValueError("entry must contain the message marker")
        existing = ""
        if self.path.exists():
            with open(self.path, encoding="utf-8", newline="") as f:  # keep \r\n visible
                existing = f.read()
        if self.marker(message_id) in existing:
            return False
        newline = "\r\n" if "\r\n" in existing else "\n"
        text = existing or self.header
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + entry.rstrip("\n") + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace("\r\n", "\n").replace("\n", newline))
        return True


class MonitorService:
    """Owns the turn recorder, detectors, Discord posting and approvals."""

    def __init__(
        self,
        settings,
        *,
        get_channel: Callable[[], Any],
        bot_user_id: Callable[[], int | None] = lambda: None,
        transcriber: Any = None,
        approvals_path: Path | None = None,
        dismissals_path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ):
        """
        Args:
            settings: Settings (monitor_* fields)
            get_channel: Returns the monitor text channel (or None if missing)
            bot_user_id: Returns the bot's own user id (its reactions are ignored)
            transcriber: STT client with transcribe_audio(16 kHz float32) for
                MONITOR_RETRANSCRIBE; created from GROQ_API_KEY if not given
            approvals_path: Approved-fixes file (default MONITOR_APPROVED_FILE)
            dismissals_path: Dismissed-findings file (default MONITOR_DISMISSED_FILE)
            clock: Wall-clock seconds (tests)
        """
        self.settings = settings
        self.enabled = bool(settings.monitor_enabled)
        self.thresholds = Thresholds.from_settings(settings)
        self.owner_id: int | None = settings.monitor_owner_id
        self.redact = Redactor(settings)
        self._get_channel = get_channel
        self._bot_user_id = bot_user_id
        self._transcriber = transcriber
        self._retranscribe = bool(settings.monitor_retranscribe)
        self._clock = clock
        self.approvals = ApprovalLog(_in_repo(approvals_path or settings.monitor_approved_file))
        self.dismissals = ApprovalLog(
            _in_repo(dismissals_path or settings.monitor_dismissed_file), DISMISSED_FILE_HEADER)
        self._dedupe_s = settings.monitor_dedupe_minutes * 60
        self.recorder = TurnRecorder(self._on_turn, retain_audio=self._retranscribe)
        self.archive: AudioArchive | None = None
        if self.enabled and settings.monitor_save_audio:
            self.archive = AudioArchive(
                _in_repo(settings.monitor_audio_dir), max_mb=settings.monitor_audio_max_mb,
                wake_min_score=settings.monitor_wake_clip_min_score,
                names=getattr(settings, "user_names", None) or {})
            self.recorder.keep_question_audio = True
            self.recorder.on_wake_clip = self.archive.save_wake
        self._recent: dict[str, PostedIssue] = {}  # finding key -> post
        self._posted: dict[int, PostedIssue] = {}  # message id -> post
        self._post_times: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._watchdog: asyncio.Task | None = None
        self._warned_no_channel = False

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Begin collecting logs and watching for hung turns (safe to call twice)."""
        if not self.enabled:
            return
        self.recorder.install()
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.create_task(self._watch())
        if self.owner_id is None:
            logger.warning("[MONITOR] MONITOR_OWNER_ID not set: posts can't be approved")
        logger.info(f"[MONITOR] On: posting to #{self.settings.monitor_channel_name}")
        if self.archive is not None:
            logger.info(f"[MONITOR] Saving question and wake word audio to {self.archive.root}")

    async def stop(self) -> None:
        self.recorder.uninstall()
        self.recorder.on_wake_clip = None
        tasks = [t for t in [self._watchdog, *self._tasks] if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._watchdog = None
        self._tasks.clear()
        if self.archive is not None:
            await asyncio.to_thread(self.archive.close)  # finish pending writes

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            self.recorder.check_hung(self.thresholds.hang_s)

    def _on_turn(self, turn: LiveTurn) -> None:
        if not self.enabled:
            return
        try:
            task = asyncio.get_running_loop().create_task(self.handle_turn(turn))
        except RuntimeError:
            return  # no loop (shutting down)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------ turns

    async def handle_turn(self, turn: LiveTurn) -> Any | None:
        """Detect problems in a turn and post them. Returns the new message, if any."""
        if not self.enabled:
            return None
        try:
            if self._retranscribe and turn.answer_audio and turn.spoken is None:
                turn.spoken = await self._transcribe(turn.answer_audio)
            turn.answer_audio = []
            findings = detect(turn, self.thresholds)
            self._save_audio(turn, findings)
            if not findings:
                return None
            for f in findings:
                logger.info(f"[MONITOR] {f.detector}: {f.detail[:200]}")
            async with self._lock:
                return await self._post_findings(turn, findings)
        except Exception as e:
            logger.warning(f"[MONITOR] Could not handle turn: {e}", exc_info=True)
            return None

    def _save_audio(self, turn: LiveTurn, findings: list[Finding]) -> None:
        """Keep the question audio (MONITOR_SAVE_AUDIO). A hang report is a snapshot of
        a turn still running; the finished turn is saved when it ends."""
        audio, turn.question_audio = turn.question_audio, None
        if self.archive is None or audio is None or turn.outcome == HUNG:
            return
        turn.question_audio = audio
        try:
            path = self.archive.save_turn(turn, findings)
        finally:
            turn.question_audio = None
        if path is not None:
            try:
                turn.audio_file = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                turn.audio_file = path.as_posix()

    async def _post_findings(self, turn: LiveTurn, findings: list[Finding]) -> Any | None:
        now = self._clock()
        fresh: list[Finding] = []
        repeated: dict[int, PostedIssue] = {}
        for f in findings:
            issue = self._recent.get(f.key)
            if (issue and now - issue.last_seen <= self._dedupe_s
                    and now - issue.posted_at <= REPOST_AFTER_S):
                repeated[id(issue)] = issue
            else:
                fresh.append(f)

        for issue in repeated.values():
            issue.count += 1
            issue.last_seen = now
            await self._edit(issue.message, with_seen_count(
                issue.content, issue.count, datetime.fromtimestamp(now)))
            for key in issue.keys:
                self._recent[key] = issue

        if not fresh:
            return None
        while self._post_times and now - self._post_times[0] > 3600:
            self._post_times.popleft()
        if len(self._post_times) >= MAX_POSTS_PER_HOUR:
            logger.info(f"[MONITOR] Hourly post limit reached, not posting: {fresh[0].summary}")
            return None

        channel = self._get_channel()
        if channel is None:
            if not self._warned_no_channel:
                logger.warning(f"[MONITOR] Text channel #{self.settings.monitor_channel_name} "
                               "not found; findings are only logged")
                self._warned_no_channel = True
            return None

        content = format_post(turn, fresh, self.redact)
        try:
            message = await channel.send(content, allowed_mentions=_no_mentions())
        except Exception as e:
            logger.warning(f"[MONITOR] Posting failed: {e}")
            return None
        self._post_times.append(now)
        issue = PostedIssue(message=message, content=content, keys={f.key for f in fresh},
                            posted_at=now, posted_dt=datetime.fromtimestamp(now), last_seen=now)
        self._posted[message.id] = issue
        for key in issue.keys:
            self._recent[key] = issue
        for emoji in (APPROVE, REJECT):
            try:
                await message.add_reaction(emoji)
            except Exception as e:
                logger.warning(f"[MONITOR] Could not add reaction (needs Add Reactions): {e}")
                break
        return message

    async def _transcribe(self, clips: list[np.ndarray]) -> str | None:
        from scipy.signal import resample_poly

        audio = np.concatenate(clips).astype(np.float32)
        if len(audio) < 12000:  # under 0.5s
            return None
        if self._transcriber is None:
            if not self.settings.groq_api_key:
                return None
            from ..stt import GroqWhisperClient

            self._transcriber = GroqWhisperClient(api_key=self.settings.groq_api_key,
                                                  model="whisper-large-v3-turbo", language="en")
        audio16 = resample_poly(audio, 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz
        text = await self._transcriber.transcribe_audio(audio16)
        return (text or "").strip()

    async def _edit(self, message: Any, content: str) -> None:
        try:
            await message.edit(content=content)
        except Exception as e:
            logger.debug(f"[MONITOR] Could not edit post: {e}")

    # ------------------------------------------------------------ approvals

    async def handle_reaction(
        self,
        message_id: int,
        user_id: int,
        emoji: str,
        fetch_message: Callable[[], Awaitable[Any]] | None = None,
    ) -> str | None:
        """Owner reaction on a monitor post.

        Check: "something went wrong here" -> approved-fixes file. Cross: "no problem, or
        it didn't disrupt the flow" -> dismissed-findings file. Returns "approved",
        "already_approved", "dismissed", "already_dismissed", or None if ignored.
        """
        if not self.enabled or emoji not in (APPROVE, REJECT):
            return None
        if self.owner_id is None or user_id != self.owner_id or user_id == self._bot_user_id():
            return None

        issue = self._posted.get(message_id)
        message, content, posted = (issue.message, issue.content, issue.posted_dt) if issue \
            else (None, None, None)
        if issue is None:
            # Posted before a restart: read it back from Discord
            if fetch_message is None:
                return None
            try:
                message = await fetch_message()
            except Exception as e:
                logger.warning(f"[MONITOR] Could not fetch reacted message {message_id}: {e}")
                return None
            if not self._is_monitor_post(message):
                return None
            content = message.content
            posted = getattr(message, "created_at", None)
            if posted is not None:
                posted = posted.astimezone().replace(tzinfo=None)

        approve = emoji == APPROVE
        log, done = (self.approvals, "approved") if approve else (self.dismissals, "dismissed")
        if log.contains(message_id):
            return f"already_{done}"
        jump_url = getattr(message, "jump_url", "") or ""
        entry = approval_entry(message_id, content, jump_url, posted,
                               datetime.fromtimestamp(self._clock()), self.redact,
                               CONFIRMED if approve else DISMISSED)
        if not log.add(message_id, entry):
            return f"already_{done}"
        logger.info(f"[MONITOR] Owner {done} post {message_id}; added to {log.path}")
        self._note_reaction(content, CONFIRMED if approve else DISMISSED, jump_url)
        if approve and message is not None and content is not None:
            content += "\nConfirmed: added to the fix list (cause still to be checked)."
            if issue is not None:
                issue.content = content
            await self._edit(message, content)
        return done

    def _note_reaction(self, content: str | None, verdict: str, jump_url: str) -> None:
        """Copy the owner's verdict into the recording's sidecar, if the post has one."""
        rel = recording_path(content or "")
        if self.archive is None or rel is None:
            return
        path = Path(rel)
        path = path if path.is_absolute() else REPO_ROOT / path
        if path.resolve().is_relative_to(self.archive.root.resolve()):
            self.archive.note_reaction(path, verdict, jump_url)

    def _is_monitor_post(self, message: Any) -> bool:
        author = getattr(getattr(message, "author", None), "id", None)
        bot_id = self._bot_user_id()
        channel = getattr(getattr(message, "channel", None), "name", None)
        return (bot_id is not None and author == bot_id
                and (channel is None or channel == self.settings.monitor_channel_name)
                and (getattr(message, "content", "") or "").startswith(POST_PREFIX))


def _in_repo(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _no_mentions():
    import discord

    return discord.AllowedMentions.none()


def create_monitor(settings, **kwargs) -> MonitorService | None:
    """MonitorService if MONITOR_ENABLED, else None (nothing is recorded or posted)."""
    if not settings.monitor_enabled:
        return None
    return MonitorService(settings, **kwargs)
