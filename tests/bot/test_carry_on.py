"""Nothing the speaker says is lost: a pause that ended their turn too early starts the
turn over, and follow-up windows wait while other people talk."""

import asyncio
import time
from types import SimpleNamespace

import numpy as np

from src.bot.voice_session import CARRY_ON, VoiceSession
from src.bot.voice_sink import StreamingSink
from src.config import Settings

SPEAKER, OTHER = 111, 222


def pcm(ms: int = 20, loud: bool = True) -> bytes:
    """48kHz stereo s16le (Discord's decoded format); quiet is near-silence."""
    samples = 48 * ms
    tone = (np.sin(np.arange(samples) / 5) * (8000 if loud else 10)).astype(np.int16)
    return np.repeat(tone, 2).tobytes()


def talk(sink: StreamingSink, user: int, seconds: float, loud: bool = True) -> None:
    for _ in range(int(seconds / 0.02)):
        sink.write(pcm(loud=loud), user)


def make_sink() -> StreamingSink:
    return StreamingSink(vad=None)


class TestSettings:
    def test_two_seconds_of_quiet_ends_a_question(self):
        assert Settings(_env_file=None, DISCORD_TOKEN="x").min_silence_ms == 2000


class TestCapture:
    def test_keeps_only_the_speakers_speech(self):
        sink = make_sink()
        sink.start_capture(SPEAKER)
        talk(sink, OTHER, 1.0)
        talk(sink, SPEAKER, 0.5, loud=False)
        assert sink.captured_speech() == 0
        talk(sink, SPEAKER, 0.6)
        assert abs(sink.captured_speech() - 0.6) < 0.05

    def test_speech_while_the_bot_plays_is_kept_but_not_counted_as_carrying_on(self, monkeypatch):
        sink = make_sink()
        sink.start_capture(SPEAKER)
        monkeypatch.setattr(sink, "_bot_is_playing", lambda: True)
        talk(sink, SPEAKER, 0.6)
        assert sink.captured_speech(while_bot_quiet=True) == 0
        assert sink.captured_speech() > 0.5

    def test_listening_to_the_speaker_starts_from_what_they_said(self):
        sink = make_sink()
        sink.start_capture(SPEAKER)
        talk(sink, SPEAKER, 1.0)
        sink.start_listening(user_id=SPEAKER)
        assert abs(len(sink.get_audio_buffer()) / 16000 - 1.0) < 0.05
        assert sink.is_speech_active()
        assert sink.captured_speech() == 0  # taken

    def test_restart_puts_the_first_part_first(self):
        sink = make_sink()
        first = np.full(16000, 0.5, dtype=np.float32)
        sink.start_capture(SPEAKER)
        talk(sink, SPEAKER, 0.5)
        sink.start_listening(user_id=SPEAKER, prefix=first)
        audio = sink.get_audio_buffer()
        assert abs(len(audio) / 16000 - 1.5) < 0.05
        assert np.all(audio[:16000] == 0.5)

    def test_measuring_capture_is_not_folded_into_the_next_turn(self):
        sink = make_sink()
        sink.start_capture(SPEAKER, fold=False)
        talk(sink, SPEAKER, 1.0)
        assert sink.captured_speech() > 0.9
        sink.start_listening(user_id=SPEAKER)
        assert not sink.is_speech_active()
        assert sink.captured_speech() == 0  # a new turn ends the measurement

    def test_another_speakers_turn_ends_the_capture(self):
        sink = make_sink()
        sink.start_capture(SPEAKER)
        talk(sink, SPEAKER, 1.0)
        sink.start_listening(user_id=OTHER)
        assert len(sink.get_audio_buffer()) == 0 or not sink.is_speech_active()
        assert sink.captured_speech() == 0

    def test_continued_listening_ends_after_the_speaker_goes_quiet(self):
        # Discord can keep sending packets (noise) after they stop talking
        sink = make_sink()
        sink.start_capture(SPEAKER)
        talk(sink, SPEAKER, 0.5)
        sink.start_listening(user_id=SPEAKER)
        talk(sink, SPEAKER, 0.2, loud=False)
        assert not sink.check_audio_timeout()
        sink._last_voiced_time -= sink._silence_timeout
        assert sink.check_audio_timeout()

    def test_stop_capture_only_fold_leaves_a_measurement_running(self):
        sink = make_sink()
        sink.start_capture(SPEAKER, fold=False)
        sink.stop_capture(only_fold=True)
        talk(sink, SPEAKER, 0.5)
        assert sink.captured_speech() > 0.4


class FakePipeline:
    def __init__(self):
        self.answer_started = False
        self.cancelled = False

    async def cancel(self):
        self.cancelled = True


def session_with(sink: StreamingSink) -> VoiceSession:
    session = VoiceSession.__new__(VoiceSession)
    session.sink = sink
    session._current_speaker_id = SPEAKER
    session._current_speaker_name = "Sam"
    session._voice_pipeline = FakePipeline()
    session.voice_client = None
    session._carry_on_audio = None
    session.speech_checks = []

    async def speech_confirmed(while_bot_quiet):  # the tones here stand in for speech
        session.speech_checks.append(while_bot_quiet)
        return True

    session._speech_confirmed = speech_confirmed
    return session


class TestStartOver:
    async def test_speaker_carries_on_before_the_answer(self):
        sink = make_sink()
        session = session_with(sink)
        heard = []

        async def hear_the_rest(speaker, said):
            heard.append((speaker, len(said)))
            return np.zeros(len(said) + 16000, dtype=np.float32)

        session._hear_the_rest = hear_the_rest
        process = asyncio.create_task(asyncio.sleep(5))
        loop = asyncio.get_running_loop()
        loop.call_later(0.15, talk, sink, SPEAKER, 0.6)

        said = np.zeros(8000, dtype=np.float32)
        assert await session._carried_on(process, said)
        assert process.cancelled() and session._voice_pipeline.cancelled
        assert heard == [(SPEAKER, 8000)]
        assert len(session._carry_on_audio) == 24000

    async def test_talking_during_the_answer_does_not_start_over(self):
        sink = make_sink()
        session = session_with(sink)
        session._voice_pipeline.answer_started = True

        async def pipeline():
            await asyncio.sleep(0.15)
            talk(sink, SPEAKER, 0.6)
            await asyncio.sleep(0.15)

        process = asyncio.create_task(pipeline())
        assert not await session._carried_on(process, np.zeros(10, dtype=np.float32))
        assert sink.captured_speech() > 0.5  # kept for the follow-up window

    async def test_clicks_do_not_start_over(self):
        """2026-10-15: clicks through Pat's mic counted as 0.4s of speech (the speaking
        detector holds each burst for 300ms); Silero found none, so no restart."""
        sink = make_sink()
        session = session_with(sink)

        async def not_speech(while_bot_quiet):
            session.speech_checks.append(while_bot_quiet)
            return False

        session._speech_confirmed = not_speech
        loop = asyncio.get_running_loop()
        loop.call_later(0.15, talk, sink, SPEAKER, 0.6)
        process = asyncio.create_task(asyncio.sleep(1.2))
        assert not await session._carried_on(process, np.zeros(10, dtype=np.float32))
        assert not process.cancelled() and not session._voice_pipeline.cancelled
        # Checked once it reached 0.4s, then only as it grew (0.6s: one more at most)
        assert 1 <= len(session.speech_checks) <= 2 and all(session.speech_checks)

    async def test_nothing_more_said(self):
        sink = make_sink()
        session = session_with(sink)
        process = asyncio.create_task(asyncio.sleep(0.2))
        assert not await session._carried_on(process, np.zeros(10, dtype=np.float32))

    def test_sentinel_is_not_a_pipeline_result(self):
        assert CARRY_ON not in (None, "wake_word_only", "empty", "cancelled", "not_addressed")


class FakeDetector:
    def __init__(self):
        self.speaking: list[int] = []

    def get_speaking_users(self):
        return list(self.speaking)


def followup_session(grace: float) -> tuple[VoiceSession, FakeDetector]:
    session = session_with(make_sink())
    detector = FakeDetector()
    session.speaking_detector = detector
    session.settings = SimpleNamespace(
        speaking_detection_enabled=True, followup_grace_seconds=grace,
        followup_max_wait_seconds=10.0, max_question_duration=30.0,
    )
    session._state = "idle"
    session._running = True
    session._speech_end_event = asyncio.Event()
    session._pending_audio = None
    session._wake_from_followup = None
    session._followup_open = False
    return session, detector


class TestFollowUpWindow:
    async def test_window_closes_on_time_in_a_quiet_room(self):
        session, _ = followup_session(grace=0.6)
        started = time.perf_counter()
        assert await session._listen_for_followup(SPEAKER, 0.4, bot_asked=True) is None
        assert time.perf_counter() - started < 0.9

    async def test_others_talking_pauses_the_window_and_adds_grace(self):
        session, detector = followup_session(grace=0.6)
        detector.speaking = [OTHER]
        asyncio.get_running_loop().call_later(0.8, detector.speaking.clear)
        started = time.perf_counter()
        assert await session._listen_for_followup(SPEAKER, 0.4, bot_asked=True) is None
        # 0.8 s of someone else talking, then the 0.6 s grace (more than the 0.4 s left)
        assert time.perf_counter() - started >= 1.3
        assert 0.6 <= session._followup_others_talked <= 1.0

    async def test_the_speakers_own_talk_is_not_someone_else(self):
        session, detector = followup_session(grace=0.6)
        detector.speaking = [SPEAKER]
        started = time.perf_counter()
        await session._listen_for_followup(SPEAKER, 0.4, bot_asked=True)
        assert time.perf_counter() - started < 0.9

    async def test_who_is_talking_is_read_under_the_sinks_lock(self):
        """The audio thread adds people to the detector while this reads it."""
        session, detector = followup_session(grace=0.6)
        held = []

        def get_speaking_users():
            held.append(session.sink.lock._is_owned())
            return [OTHER]

        detector.get_speaking_users = get_speaking_users
        assert session._others_talking(SPEAKER)
        assert held == [True]


class FakeObserver:
    def __init__(self):
        self.incidents = []

    def incident(self, speaker_id, name, outcome, note, reply=""):
        self.incidents.append((speaker_id, outcome, reply))


class TestLateReply:
    def watcher(self, monkeypatch):
        import src.bot.voice_session as vs

        monkeypatch.setattr(vs, "LATE_REPLY_WATCH_SECONDS", 0.2)
        sink = make_sink()
        session = session_with(sink)
        session.turn_observer = FakeObserver()
        session._followup_others_talked = 4.0
        session._late_reply_task = None
        return session, sink

    async def test_talking_just_after_the_window_closed_is_reported(self, monkeypatch):
        session, sink = self.watcher(monkeypatch)
        session._watch_for_late_reply(SPEAKER, "Want the recipe?")
        talk(sink, SPEAKER, 0.6)
        await session._late_reply_task
        assert session.turn_observer.incidents == [(SPEAKER, "late_reply", "Want the recipe?")]

    async def test_a_new_turn_is_not_a_late_reply(self, monkeypatch):
        session, sink = self.watcher(monkeypatch)
        session._watch_for_late_reply(SPEAKER, "Want the recipe?")
        talk(sink, SPEAKER, 0.6)
        sink.start_listening(user_id=SPEAKER)  # they said the wake word
        sink.start_capture(SPEAKER)  # and the new turn is running
        await session._late_reply_task
        assert session.turn_observer.incidents == []
        assert sink.is_capturing(SPEAKER, fold=True)  # left alone

    async def test_silence_after_the_window_is_fine(self, monkeypatch):
        session, sink = self.watcher(monkeypatch)
        session._watch_for_late_reply(SPEAKER, "Want the recipe?")
        await session._late_reply_task
        assert session.turn_observer.incidents == []


class TestSaidDuringTheAnswer:
    """2026-10-15: Pat's "mm-hmm" during the answer started the reply window, which
    ended at once (he'd finished 5s before); it wasn't for the bot, and his real
    follow-up had no window left."""

    def test_sink_flags_speech_that_was_already_over(self):
        sink = make_sink()
        sink.start_capture(SPEAKER)
        talk(sink, SPEAKER, 1.0)
        sink.start_listening(user_id=SPEAKER)
        assert not sink.seed_finished  # still talking: listen on

        sink = make_sink()
        sink.start_capture(SPEAKER)
        talk(sink, SPEAKER, 1.0)
        sink._capture_last_voiced -= sink._silence_timeout + 3
        sink.start_listening(user_id=SPEAKER)
        assert sink.seed_finished and sink.check_audio_timeout()

    def test_nothing_said_is_not_flagged(self):
        sink = make_sink()
        sink.start_capture(SPEAKER)
        sink.start_listening(user_id=SPEAKER)
        assert not sink.seed_finished

    @staticmethod
    def session(during_answer: list[bool], results: list) -> VoiceSession:
        session = VoiceSession.__new__(VoiceSession)
        session.settings = Settings(_env_file=None, DISCORD_TOKEN="x")
        session._addressee_judge = object()
        session._running = True
        session._current_speaker_id = SPEAKER
        session._wake_from_followup = None
        session._last_reply = lambda: "It's Canberra."
        session.listens = 0

        async def listen(speaker_id, window, bot_asked):
            session.listens += 1
            if not during_answer:
                return None
            session._followup_said_during_answer = during_answer.pop(0)
            return np.zeros(16000, dtype=np.float32)

        async def process(audio, followup=None):
            return results.pop(0)

        session._listen_for_followup = listen
        session._process_speech = process
        return session

    async def test_window_reopens_once_after_a_reaction(self):
        session = self.session([True, True], ["not_addressed", "not_addressed"])
        await session._followup_after(None)
        assert session.listens == 2  # once more, not forever

    async def test_reopened_window_hears_the_follow_up(self):
        session = self.session([True, False], ["not_addressed", None])
        await session._followup_after(None)
        assert session.listens == 3  # reaction, the follow-up, then the next window

    async def test_ordinary_not_for_the_bot_ends_it(self):
        session = self.session([False], ["not_addressed"])
        await session._followup_after(None)
        assert session.listens == 1
