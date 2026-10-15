"""Replays a real timeline through the real sink and session, faster than real time.

Audio goes in as Discord packets (20 ms, 48 kHz stereo) at the moments the incident
says, with the clock scaled by SCALE; the bot's timings (silence that ends a turn,
follow-up windows) come from the real Settings defaults, scaled the same way. Each
utterance gets its own loudness so the tests can tell whose audio ended up where.
"""

import asyncio
from types import SimpleNamespace

import numpy as np

from src.bot.voice_session import VoiceSession
from src.bot.voice_sink import StreamingSink
from src.config import Settings

from .cases import Speaker, Speech

SCALE = 0.1  # 1 s of the incident takes 0.1 s here
SAMPLE_RATE = 16000
PACKET_SECONDS = 0.02
CHUNK_PACKETS = 25  # packets written together (0.5 s of audio), then the clock catches up


def scaled_settings(speaking_detection: bool = True) -> SimpleNamespace:
    """The real defaults the session and sink use, time-scaled."""
    real = Settings(_env_file=None, DISCORD_TOKEN="x")
    return SimpleNamespace(
        speaking_detection_enabled=speaking_detection,
        silence_seconds=real.min_silence_ms / 1000 * SCALE,
        max_question_duration=real.max_question_duration * SCALE,
        followup_after_answer_seconds=real.followup_after_answer_seconds * SCALE,
        followup_window_seconds=real.followup_window_seconds * SCALE,
        followup_grace_seconds=real.followup_grace_seconds * SCALE,
        followup_max_wait_seconds=real.followup_max_wait_seconds * SCALE,
    )


def pcm(amplitude: int) -> bytes:
    """One Discord packet: a tone at this loudness (s16le, 48 kHz stereo)."""
    samples = int(48000 * PACKET_SECONDS)
    tone = (np.sin(np.arange(samples) / 5) * amplitude).astype(np.int16)
    return np.repeat(tone, 2).tobytes()


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0


def seconds(audio: np.ndarray | None) -> float:
    return 0.0 if audio is None else len(audio) / SAMPLE_RATE


class EnergyVAD:
    """Stands in for Silero: speech starts on the first loud packet. (Discord stops
    sending when someone stops talking, so turns end on the sink's silence timeout.)"""

    def __init__(self):
        self.is_speech = False

    def reset(self) -> None:
        self.is_speech = False

    def detect_endpoint(self, audio: np.ndarray) -> str | None:
        if rms(audio) >= 0.01 and not self.is_speech:
            self.is_speech = True
            return "start"
        return None


class FakeDetector:
    """Speaking detector: who is talking right now."""

    def __init__(self):
        self.speaking: set[int] = set()

    def get_speaking_users(self) -> list[int]:
        return list(self.speaking)


def make_sink(settings: SimpleNamespace) -> StreamingSink:
    sink = StreamingSink(vad=EnergyVAD())
    sink._silence_timeout = settings.silence_seconds
    # The wake word isn't under test here (and the real model is slow to load)
    sink._wake_processor.process_audio = lambda user_id, audio: False
    return sink


class Room:
    """The incident's clock, scaled. `start` is the incident time at creation."""

    def __init__(self, start: float):
        self._loop = asyncio.get_running_loop()
        self._t0 = self._loop.time() - start * SCALE

    def now(self) -> float:
        return (self._loop.time() - self._t0) / SCALE

    async def at(self, when: float) -> None:
        await asyncio.sleep(max(0.0, self._t0 + when * SCALE - self._loop.time()))

    async def say(self, sink: StreamingSink, who: Speaker, speech: Speech, amplitude: int,
                  detector: FakeDetector | None = None) -> None:
        """Send `speech` to the sink as Discord would, in real-time order."""
        await self.at(speech.start)
        if detector is not None:
            detector.speaking.add(who.id)
        packets = round(speech.seconds / PACKET_SECONDS)
        sent = 0
        while sent < packets:
            chunk = min(CHUNK_PACKETS, packets - sent)
            await self.at(speech.start + (sent + chunk) * PACKET_SECONDS)  # once said
            for _ in range(chunk):
                sink.write(pcm(amplitude), who.id)
            sent += chunk
        if detector is not None:
            detector.speaking.discard(who.id)


class ScriptedPipeline:
    """Stands in for VoicePipeline: thinks for `answer_after` (maybe playing a filler
    meanwhile), then plays the answer for `answer_seconds`. Incident seconds."""

    def __init__(self, answer_after: float, answer_seconds: float,
                 filler: Speech | None = None):
        self.answer_after = answer_after
        self.answer_seconds = answer_seconds
        self.filler = filler  # relative to the start of process()
        self.answer_started = False
        self.bot_playing = False
        self.heard: list[np.ndarray] = []  # audio given to each process() call
        self.cancelled: list[int] = []  # calls cancelled (1-based)
        self.answered: list[int] = []  # calls whose answer played to the end
        self.cancel_calls = 0

    async def process(self, audio: np.ndarray, opening=None, **kwargs) -> str | None:
        self.heard.append(audio.copy())
        call = len(self.heard)
        self.answer_started = False
        waited = 0.0
        try:
            if self.filler is not None and call == 1:
                await asyncio.sleep(self.filler.start * SCALE)
                self.bot_playing = True
                await asyncio.sleep(self.filler.seconds * SCALE)
                self.bot_playing = False
                waited = self.filler.end
            await asyncio.sleep(max(0.0, self.answer_after - waited) * SCALE)
            self.answer_started = self.bot_playing = True
            await asyncio.sleep(self.answer_seconds * SCALE)
            self.answered.append(call)
            return None
        except asyncio.CancelledError:
            self.cancelled.append(call)
            raise
        finally:
            self.bot_playing = False
            if opening is not None:
                opening.cancel()

    async def cancel(self) -> None:
        self.cancel_calls += 1


async def _no_audio(*args) -> np.ndarray:
    return np.array([], dtype=np.float32)


def make_session(sink: StreamingSink, speaker: Speaker, settings: SimpleNamespace,
                 pipeline: ScriptedPipeline | None = None,
                 detector: FakeDetector | None = None) -> VoiceSession:
    """A VoiceSession with only what the turn and follow-up code paths use."""
    session = VoiceSession.__new__(VoiceSession)
    session.settings = settings
    session.sink = sink
    session.speaking_detector = detector
    session.turn_observer = None
    session.voice_client = None
    session.tts = SimpleNamespace(synthesize_cached=_no_audio)
    session._voice_pipeline = pipeline
    session._thinking_generator = None
    session._state = "idle"
    session._running = True
    session._current_speaker_id = speaker.id
    session._current_speaker_name = speaker.name
    session._current_speaker_activity = None
    session._speech_end_event = asyncio.Event()
    session._pending_audio = None
    session._carry_on_audio = None
    session._wake_from_followup = None
    session._followup_open = False
    session._followup_others_talked = 0.0
    session._late_reply_task = None

    async def speech_confirmed(while_bot_quiet):  # the replay's tones stand in for speech
        return True

    session._speech_confirmed = speech_confirmed
    return session


async def listen_for_turn(sink: StreamingSink, speaker: Speaker) -> np.ndarray:
    """What the main loop does after the wake word: listen until the speaker goes quiet."""
    sink.start_listening(user_id=speaker.id)
    while not sink.check_audio_timeout():
        await asyncio.sleep(0.01)
    audio = sink.get_audio_buffer()
    sink.stop_listening()
    return audio
