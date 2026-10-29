"""End-to-end voice harness: synthetic speech in, bot speech out, no Discord connection.

Drives the real VoiceSession (sink -> wake word -> VAD -> Groq Whisper -> Cerebras
-> Inworld TTS -> playback) with a fake voice client:

- User speech is synthesized with Inworld TTS and written into the sink as 20 ms
  48 kHz stereo frames, in real time, from a separate thread (like py-cord's
  decoder thread).
- Everything the bot plays is captured frame by frame with timestamps, then
  transcribed with Groq Whisper so each step can be checked from the outside.
- A speech part can also be a Recording: a real clip the live bot saved
  (recordings/turns/*.wav), played instead of synthesized speech.

Live APIs are used (Groq, Cerebras, Inworld). Nothing is sent to Discord, unless
run_voice_e2e --listen sets Rig.mirror to play the run into a voice channel.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from src.audio import resample_24k_mono_to_48k_stereo, resample_48k_stereo_to_16k_mono

# Shared with the live monitor (src/monitor); re-exported for run_voice_e2e
from src.monitor.text_checks import normalize_words, overlap, word_error_rate  # noqa: F401

logger = logging.getLogger("e2e")

FRAME_BYTES = 3840  # 20 ms of 48 kHz stereo s16le
FRAME_SEC = 0.02
TTS_RATE = 24000
BOT_USER_ID = 900000000000000001


# ---------------------------------------------------------------- fake Discord


class FakeMember:
    def __init__(self, user_id: int, display_name: str, activity: Optional[str] = None):
        import discord

        self.id = user_id
        self.display_name = display_name
        self.name = display_name.lower().replace(" ", "")
        self.activities = [discord.Game(activity)] if activity else []
        self.mention = f"<@{user_id}>"
        self.bot = False


class FakeChannel:
    def __init__(self, name: str):
        self.name = name
        self.sent: list[str] = []
        self.forward = None  # --listen: async callable that posts to the real channel

    async def send(self, content=None, **kwargs):
        self.sent.append(content if content is not None else str(kwargs))
        if self.forward is not None and content:
            try:
                await self.forward(content)
            except Exception as e:
                logger.warning(f"Forwarding the post to Discord failed: {e}")


class FakeGuild:
    def __init__(self, members: list[FakeMember], channel_name: str):
        self.members = members
        self.text_channels = [FakeChannel(channel_name)]
        self.name = "e2e-guild"

    def get_member(self, user_id: int) -> Optional[FakeMember]:
        return next((m for m in self.members if m.id == user_id), None)

    @property
    def channel(self) -> FakeChannel:
        return self.text_channels[0]


class _FakeWS:
    async def speak(self, *args, **kwargs):
        return None


@dataclass
class PlaybackSegment:
    start: float
    end: float = 0.0
    frames: list[bytes] = field(default_factory=list)
    interrupted: bool = False
    transcript: str = ""

    @property
    def duration(self) -> float:
        return len(self.frames) * FRAME_SEC


class FakeVoiceClient:
    """Implements the parts of discord.VoiceClient that VoiceSession uses.

    play() runs the source in a thread at real-time pace, like discord's AudioPlayer,
    and records every frame.
    """

    def __init__(self, guild: FakeGuild, clock: Callable[[], float],
                 tap: Optional[Callable[[bytes], None]] = None):
        self.guild = guild
        self._tap = tap  # gets every played frame (--listen)
        self.user = FakeMember(BOT_USER_ID, "Jarvis")
        self.ws = _FakeWS()
        self.recording = False
        self.sink = None
        self.segments: list[PlaybackSegment] = []
        self._clock = clock
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def is_connected(self) -> bool:
        return True

    def start_recording(self, sink, callback, *args):
        sink.init(self)
        self.sink = sink
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        if self.is_playing():
            self._stop.set()
            self._thread.join(timeout=2)

    def play(self, source, *, after=None, **kwargs):
        import discord

        if self.is_playing():
            raise discord.ClientException("Already playing audio.")
        segment = PlaybackSegment(start=self._clock())
        self.segments.append(segment)
        self._stop = threading.Event()
        stop = self._stop

        def run():
            error = None
            try:
                next_at = time.perf_counter()
                while not stop.is_set():
                    data = source.read()
                    if not data:
                        break
                    segment.frames.append(data)
                    if self._tap:
                        self._tap(data)
                    next_at += FRAME_SEC
                    delay = next_at - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
            except Exception as e:  # pragma: no cover - surfaced via after()
                error = e
            segment.end = self._clock()
            segment.interrupted = stop.is_set()
            if after:
                after(error)

        self._thread = threading.Thread(target=run, name="fake-audio-player", daemon=True)
        self._thread.start()


# ---------------------------------------------------------------- user speech

def read_sidecar(wav_path: Path | str) -> dict:
    """The .json saved next to a recording ({} if missing or unreadable)."""
    path = Path(wav_path).with_suffix(".json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class Recording:
    """A real clip played instead of synthesized speech: recordings/turns/<...>.wav from
    the live bot (16 kHz mono int16, exactly what went to Whisper; other rates and widths
    work too). The same-name .json sidecar gives the text shown in reports."""

    path: Path | str
    trim: bool = False  # cut leading/trailing silence (off: keep the real timing)

    @property
    def meta(self) -> dict:
        return read_sidecar(self.path)

    @property
    def text(self) -> str:
        meta = self.meta
        return (meta.get("raw_heard") or meta.get("heard")
                or f"<recording {Path(self.path).name}>")


def load_recording(rec: Recording) -> np.ndarray:
    """A WAV as 24 kHz mono float32, the format synthesized speech is rendered in."""
    from math import gcd

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(str(rec.path))
    if data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        audio = data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != TTS_RATE:
        g = gcd(rate, TTS_RATE)
        audio = resample_poly(audio, TTS_RATE // g, rate // g)
    audio = np.clip(audio, -1, 1).astype(np.float32)
    return _trim_silence(audio) if rec.trim else audio


# Text to speak, seconds of silence, or a recorded clip
Part = Union[str, float, Recording]


@dataclass
class Speech:
    """One person talking. parts mixes text, pauses and recordings:
    ["Hey Jarvis.", 1.0, "what time is it?"] or [Recording("recordings/turns/x.wav")]."""

    user: FakeMember
    parts: list[Part]
    at: float = 0.5  # seconds after scenario start
    voice: str = "ashley"
    noise_snr_db: Optional[float] = None  # add background noise at this SNR
    mode: str = "vad"  # "vad": frames stop in silence (Discord voice activity); "open": open mic
    # Start this many seconds after the bot finishes its previous reply, while it may
    # still be listening for a follow-up (instead of waiting for it to go idle)
    after_reply: Optional[float] = None
    # Start this many seconds after the bot starts speaking its answer (fillers don't
    # count), i.e. talk over it. Like after_reply, needs the scenario's `between`.
    during_answer: float | None = None
    # Start this many seconds after the previous speech ends (its last packet), in the
    # same turn: e.g. a friend talks, then the asker replies
    after_prev: float | None = None
    started_at: float = 0.0  # set when scheduled (scenario clock)
    ended_at: float = 0.0  # time of the last packet sent

    @property
    def text(self) -> str:
        return " ".join(p if isinstance(p, str) else p.text
                        for p in self.parts if isinstance(p, (str, Recording)))


class SpeechFactory:
    """Synthesizes user speech with Inworld (disk-cached) and turns it into Discord frames."""

    def __init__(self, api_key: str, cache_dir: Path):
        self._api_key = api_key
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)
        self._synths: dict = {}

    def _synth(self, voice: str):
        from src.tts.synthesizer import TTSSynthesizer

        if voice not in self._synths:
            self._synths[voice] = TTSSynthesizer(
                model="inworld-tts-2-flash", voice=voice, api_key=self._api_key,
                temperature=0.8, speed=1.0,
            )
        return self._synths[voice]

    async def say(self, text: str, voice: str) -> np.ndarray:
        key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
        path = self._cache / f"{key}.npy"
        if path.exists():
            return np.load(path)
        audio = await self._synth(voice).synthesize(text)
        if len(audio) == 0:
            raise RuntimeError(f"TTS returned no audio for {text!r} ({voice})")
        audio = _trim_silence(audio.astype(np.float32))
        np.save(path, audio)
        return audio

    async def render(self, speech: Speech, seed: int = 0) -> np.ndarray:
        """24 kHz mono float32 for the whole speech."""
        pieces = []
        for part in speech.parts:
            if isinstance(part, str):
                pieces.append(await self.say(part, speech.voice))
            elif isinstance(part, Recording):
                pieces.append(load_recording(part))
            else:
                pieces.append(np.zeros(int(part * TTS_RATE), dtype=np.float32))
        audio = np.concatenate(pieces)
        if speech.noise_snr_db is not None:
            audio = add_noise(audio, speech.noise_snr_db, seed)
        return audio


def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > threshold)
    if len(idx) == 0:
        return audio
    pad = int(0.05 * TTS_RATE)
    return audio[max(0, idx[0] - pad): idx[-1] + pad]


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add brown-ish background noise (fan/room hum) at the given SNR."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(audio)).astype(np.float32)
    brown = np.cumsum(white)
    brown -= np.convolve(brown, np.ones(2400) / 2400, mode="same")  # remove drift
    noise = 0.5 * brown / (np.std(brown) + 1e-9) + 0.5 * white
    voiced = audio[np.abs(audio) > 0.01]
    signal_power = np.mean(voiced ** 2) if len(voiced) else np.mean(audio ** 2)
    noise *= np.sqrt(signal_power / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    return np.clip(audio + noise, -1, 1).astype(np.float32)


def to_frames(audio_24k: np.ndarray, mode: str, hangover: float = 0.3,
              open_tail: float = 3.0) -> list[Optional[bytes]]:
    """Split into 20 ms Discord frames. None = no packet sent for that slot.

    vad:  packets only while voiced, plus `hangover` seconds (Discord voice activity)
    open: every slot gets a packet, with a quiet noise floor and an extra tail
    """
    if mode == "open":
        tail = np.random.default_rng(1).standard_normal(int(open_tail * TTS_RATE)) * 0.0005
        audio_24k = np.concatenate([audio_24k, tail.astype(np.float32)])
    pcm = resample_24k_mono_to_48k_stereo(audio_24k)
    pcm += b"\x00" * (-len(pcm) % FRAME_BYTES)
    frames = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if mode == "open":
        return frames

    samples_per_frame = int(TTS_RATE * FRAME_SEC)
    voiced = []
    for i in range(len(frames)):
        chunk = audio_24k[i * samples_per_frame:(i + 1) * samples_per_frame]
        voiced.append(len(chunk) > 0 and float(np.sqrt(np.mean(chunk ** 2))) > 0.008)
    hang = int(hangover / FRAME_SEC)
    out: list[Optional[bytes]] = []
    last_voiced = -10**9
    for i, frame in enumerate(frames):
        if voiced[i]:
            last_voiced = i
        out.append(frame if i - last_voiced <= hang else None)
    return out


class Injector:
    """Writes every speaker's frames into the sink at real-time pace from one thread."""

    def __init__(self, sink, clock: Callable[[], float],
                 tap: Optional[Callable[[bytes], None]] = None):
        self._sink = sink
        self._clock = clock
        self._tap = tap  # gets every injected frame (--listen)
        self.user_track: list[tuple[float, int, bytes]] = []  # (time, user_id, frame)
        self.last_frame_time: dict[int, float] = {}
        self.error: Optional[BaseException] = None

    def run(self, schedule: list[tuple[float, int, bytes]]) -> None:
        schedule = sorted(schedule, key=lambda x: x[0])
        try:
            for t, user_id, frame in schedule:
                delay = t - self._clock()
                if delay > 0:
                    time.sleep(delay)
                self._sink.write(frame, user_id)
                if self._tap:
                    self._tap(frame)
                now = self._clock()
                self.user_track.append((now, user_id, frame))
                self.last_frame_time[user_id] = now
        except BaseException as e:  # surfaced in the report
            self.error = e
            logger.exception("Injector failed")


def build_schedule(speech: Speech, frames: list[Optional[bytes]],
                   offset: float = 0.0) -> list[tuple[float, int, bytes]]:
    start = offset + speech.at
    speech.started_at = start
    sent = [i for i, f in enumerate(frames) if f]
    speech.ended_at = start + (sent[-1] + 1) * FRAME_SEC if sent else start
    return [(start + i * FRAME_SEC, speech.user.id, frames[i]) for i in sent]


# ---------------------------------------------------------------- analysis helpers


def frames_to_16k(frames: list[bytes]) -> np.ndarray:
    if not frames:
        return np.array([], dtype=np.float32)
    return resample_48k_stereo_to_16k_mono(b"".join(frames))


def write_wav(path: Path, user_track, segments, duration: float) -> None:
    """Stereo 48 kHz WAV: left = everyone talking to the bot, right = the bot."""
    from scipy.io import wavfile

    n = int((duration + 1) * 48000)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    def place(track, t, frame):
        x = np.frombuffer(frame, dtype=np.int16).reshape(-1, 2).mean(axis=1) / 32768.0
        i = int(t * 48000)
        x = x[: max(0, n - i)]
        track[i:i + len(x)] += x

    for t, _, frame in user_track:
        place(left, t - FRAME_SEC, frame)
    for seg in segments:
        for k, frame in enumerate(seg.frames):
            place(right, seg.start + k * FRAME_SEC, frame)
    stereo = np.stack([left, right], axis=1)
    wavfile.write(str(path), 48000, (np.clip(stereo, -1, 1) * 32767).astype(np.int16))


# ---------------------------------------------------------------- scenario run

WEB_TOOLS = ("web_search", "web_extract")


def _answer_started(session) -> bool:
    """The bot has started playing this turn's answer (VoicePipeline resets it per turn)."""
    return bool(getattr(session._voice_pipeline, "answer_started", False))


def _limit_web(limit: int | None) -> Callable[[], None]:
    """Allow at most `limit` web lookups (None: no limit) until the returned function is
    called. Past the limit the tool answers with an error instead of going online, so a
    scenario that shouldn't search can't spend the live search budget."""
    if limit is None:
        return lambda: None
    from src.tools.base import ToolResult
    from src.tools.executor import ToolExecutor

    original = ToolExecutor.execute
    used = [0]

    async def execute(self, tool_name: str, args: dict) -> ToolResult:
        if tool_name in WEB_TOOLS:
            used[0] += 1
            if used[0] > limit:
                logger.warning(f"[E2E] Blocked {tool_name} (web limit {limit} for this scenario)")
                return ToolResult(success=False, output="",
                                  error="Web lookup is unavailable right now.",
                                  metadata={"tool_name": tool_name})
        return await original(self, tool_name, args)

    ToolExecutor.execute = execute

    def restore() -> None:
        ToolExecutor.execute = original

    return restore


@dataclass
class TurnRecord:
    user_text: str  # what the bot heard (from history)
    answer: str  # what the LLM said (from history)
    tools: list[str]


@dataclass
class RunResult:
    name: str
    speeches: list[Speech]
    segments: list[PlaybackSegment]
    turns: list[TurnRecord]
    posts: list[str]
    speech_end: dict[int, float]  # user_id -> last frame time
    duration: float
    error: Optional[str] = None
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    monitor_posts: list[str] = field(default_factory=list)  # --monitor: what would be posted

    @property
    def bot_transcript(self) -> str:
        return " ".join(s.transcript for s in self.segments if s.transcript).strip()


def turns_from_history(history: list[dict]) -> list[TurnRecord]:
    turns: list[TurnRecord] = []
    for msg in history:
        if msg["role"] == "user":
            text = re.sub(r"^\[[^\]]+\]:\s*", "", msg["content"])
            turns.append(TurnRecord(user_text=text, answer="", tools=[]))
        elif turns and msg["role"] == "assistant":
            for tc in msg.get("tool_calls") or []:
                turns[-1].tools.append(tc["function"]["name"])
            if msg.get("content") and not msg.get("tool_calls"):
                turns[-1].answer = msg["content"]
    return turns


class Rig:
    """Shared components (loaded once) + a factory for fresh sessions."""

    def __init__(self, settings, report_dir: Path):
        from src.stt.groq_whisper import GroqWhisperClient

        self.settings = settings
        self.report_dir = report_dir
        self.speech = SpeechFactory(settings.inworld_api_key, report_dir.parent / "speech-cache")
        self.whisper = GroqWhisperClient(api_key=settings.groq_api_key,
                                         model="whisper-large-v3-turbo", language="en")
        self.vad = None
        self.tts = None
        self.mirror = None  # discord_mirror.DiscordMirror (--listen)

    async def run(self, name: str, speeches: list[Speech], members: list[FakeMember],
                  settle: float = 3.0, timeout: float = 90.0,
                  between: Optional[float] = None, observer=None,
                  web_limit: int | None = None) -> RunResult:
        """Run speeches through a fresh VoiceSession and collect everything it did.

        If `between` is set, speeches are played as separate turns: each one starts
        `between` seconds after the bot has gone quiet from the previous turn (a speech
        with after_prev joins the previous one's turn).
        observer: optional live-monitor TurnRecorder (run_voice_e2e --monitor)
        web_limit: web lookups (search or page read) allowed; past it the tool returns
        an error without going online. The call is still in history, so checks see it.
        """
        from src.bot.voice_session import VoiceSession

        restore_web = _limit_web(web_limit)
        t0 = time.perf_counter()
        clock = lambda: time.perf_counter() - t0  # noqa: E731
        guild = FakeGuild(members, self.settings.image_channel_name)
        vc = FakeVoiceClient(guild, clock, tap=self.mirror.bot if self.mirror else None)
        if self.mirror:
            guild.channel.forward = self.mirror.post
        session = VoiceSession(vad=self.vad, groq_client=self.whisper, tts=self.tts,
                               turn_observer=observer)
        session_task = asyncio.create_task(session.start(vc))
        while session.sink is None:
            if session_task.done():
                restore_web()
                session_task.result()
            await asyncio.sleep(0.05)
        # share loaded models with later sessions
        self.vad, self.tts = session.vad, session.tts

        error = None
        injector = Injector(session.sink, clock, tap=self.mirror.user if self.mirror else None)
        try:
            groups: list[list[Speech]] = []
            for s in speeches:
                if groups and (between is None or s.after_prev is not None):
                    groups[-1].append(s)
                else:
                    groups.append([s])
            # Synthesize everything first so it can't eat into a follow-up window
            all_rendered = [[(sp, await self.speech.render(sp, seed=k))
                             for k, sp in enumerate(group)] for group in groups]
            for gi, rendered in enumerate(all_rendered):
                group = groups[gi]
                # Offsets count from now, after models loaded and speech is synthesized
                if gi and group[0].after_reply is not None:
                    base = clock() + group[0].after_reply - group[0].at
                elif gi and group[0].during_answer is not None:
                    base = clock() + group[0].during_answer - group[0].at
                else:
                    base = clock() + (between or 0.0) - (group[0].at if gi else 0.0)
                schedule = []
                prev: Speech | None = None
                for sp, audio in rendered:
                    offset = base
                    if prev is not None and sp.after_prev is not None:
                        offset = prev.ended_at + sp.after_prev - sp.at
                    schedule += build_schedule(sp, to_frames(audio, sp.mode), offset)
                    prev = sp
                # An answer flag still set from an earlier turn: not this turn's answer
                stale = _answer_started(session)
                await asyncio.to_thread(injector.run, schedule)
                if injector.error:
                    raise injector.error
                following = groups[gi + 1][0] if gi + 1 < len(groups) else None
                if following is not None and following.during_answer is not None:
                    await self._wait_answer_started(session, timeout, stale)
                elif following is not None and following.after_reply is not None:
                    await self._wait_reply_done(session, vc, timeout)
                else:
                    await self._wait_quiet(session, vc, settle, timeout)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.exception(f"[{name}] failed")
        finally:
            await session.stop()
            session_task.cancel()
            try:
                await session_task
            except (asyncio.CancelledError, Exception):
                pass
            restore_web()

        for seg in vc.segments:
            audio = frames_to_16k(seg.frames)
            if len(audio) > 1600:
                try:
                    seg.transcript = (await self.whisper.transcribe_audio(audio)).strip()
                except Exception as e:
                    seg.transcript = f"<transcribe failed: {e}>"

        history = session._voice_pipeline._response.get_messages() if session._voice_pipeline else []
        result = RunResult(
            name=name, speeches=speeches, segments=vc.segments,
            turns=turns_from_history(history), posts=list(guild.channel.sent),
            speech_end=dict(injector.last_frame_time), duration=clock(), error=error,
        )
        write_wav(self.report_dir / f"{name}.wav", injector.user_track, vc.segments, result.duration)
        return result

    async def _wait_reply_done(self, session, vc: FakeVoiceClient, timeout: float) -> None:
        """Wait until the bot has handled the speech and stopped talking. It may still be
        listening for a follow-up, which is the point."""
        deadline = time.perf_counter() + timeout
        handled = False
        quiet_since = None
        while time.perf_counter() < deadline:
            if session.state == "processing" or vc.is_playing():
                handled, quiet_since = True, None
            elif handled:
                if quiet_since is None:
                    quiet_since = time.perf_counter()
                elif time.perf_counter() - quiet_since >= 0.3:
                    return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"bot never finished replying within {timeout:.0f}s")

    async def _wait_answer_started(self, session, timeout: float, stale: bool) -> None:
        """Wait until the bot starts playing its answer (after "Sure, ..." and fillers).
        stale: the flag was set before this turn's speech, so wait for it to reset first."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if stale:
                stale = _answer_started(session)
            elif _answer_started(session):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"bot never started answering within {timeout:.0f}s")

    async def _wait_quiet(self, session, vc: FakeVoiceClient, settle: float, timeout: float) -> None:
        """Wait until the session is idle and nothing has played for `settle` seconds."""
        deadline = time.perf_counter() + timeout
        quiet_since = None
        while time.perf_counter() < deadline:
            busy = session.state != "idle" or vc.is_playing()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.perf_counter()
            elif time.perf_counter() - quiet_since >= settle:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"session still busy after {timeout:.0f}s (state={session.state})")
