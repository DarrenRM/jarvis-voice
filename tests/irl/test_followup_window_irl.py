"""The follow-up window when other people are talking, replayed through the real sink and
VoiceSession._listen_for_followup (time-scaled real settings)."""

import asyncio

import numpy as np

from src.llm.addressee_judge import FollowUp

from .cases import ANSWER_PLAYS, DARREN_FOLLOWUPS, POLITE_WAIT, TALKS_OVER_ANSWER
from .replay import (
    SCALE,
    FakeDetector,
    Room,
    ScriptedPipeline,
    make_session,
    make_sink,
    rms,
    scaled_settings,
    seconds,
)

ASKER_LOUDNESS, OTHER_LOUDNESS = 9000, 4000


def window_setup(case, pipeline=None):
    settings = scaled_settings()
    sink = make_sink(settings)
    detector = FakeDetector()
    session = make_session(sink, case.asker, settings, pipeline, detector)
    return settings, sink, detector, session


async def test_asker_who_waits_for_someone_else_is_still_heard():
    """The owner's concern, on Darren's 2026-10-15 12:12 correction: right after the
    bot answers, Sam talks for 6 s; Darren waits for him to finish and replies 1.5 s
    later. A plain 4 s window would have closed while Sam was talking. Now the
    window is paused while others talk and he gets at least the grace period after."""
    case = POLITE_WAIT
    settings, sink, detector, session = window_setup(case)
    room = Room(start=0.0)

    async def room_talks():
        await room.say(sink, case.other, case.other_talks, OTHER_LOUDNESS, detector)
        await room.say(sink, case.asker, case.reply, ASKER_LOUDNESS, detector)

    talking = asyncio.create_task(room_talks())
    audio = await session._listen_for_followup(
        case.asker.id, settings.followup_after_answer_seconds, bot_asked=False)
    await talking

    assert case.reply.start > settings.followup_after_answer_seconds / SCALE
    # Only Darren's reply, all of it (Sam's audio isn't his)
    assert abs(seconds(audio) - case.reply.seconds) < 0.1
    assert abs(rms(audio) - rms_of(ASKER_LOUDNESS)) < 0.2 * rms_of(ASKER_LOUDNESS)
    # Window ended on his silence, not before
    assert room.now() >= case.reply.end + settings.silence_seconds / SCALE - 0.5
    others = session._followup_others_talked / SCALE
    assert abs(others - case.other_talks.seconds) < 2.5


async def test_same_reply_in_a_quiet_room_is_too_late():
    """Control for the case above: with nobody else talking, a reply 7.5 s after the
    answer is past the window (the pause is what kept it open, not a long window)."""
    case = POLITE_WAIT
    settings, sink, detector, session = window_setup(case)
    room = Room(start=0.0)
    talking = asyncio.create_task(
        room.say(sink, case.asker, case.reply, ASKER_LOUDNESS, detector))
    audio = await session._listen_for_followup(
        case.asker.id, settings.followup_after_answer_seconds, bot_asked=False)
    talking.cancel()
    assert audio is None


async def test_reply_started_during_the_answer_is_folded_in(monkeypatch):
    """Darren starts his correction while the answer is still playing (people talk over
    the end of it) and carries on after. What he said during the answer isn't lost: the
    follow-up window starts from it and ends once he has been quiet for the turn
    silence."""
    case = TALKS_OVER_ANSWER
    pipeline = ScriptedPipeline(answer_after=ANSWER_PLAYS.start,
                                answer_seconds=ANSWER_PLAYS.seconds)
    settings, sink, detector, session = window_setup(case, pipeline)
    monkeypatch.setattr(sink, "_bot_is_playing", lambda: pipeline.bot_playing)
    room = Room(start=0.0)
    talking = asyncio.create_task(
        room.say(sink, case.asker, case.reply, ASKER_LOUDNESS, detector))

    # The follow-up turn whose answer he talks over, then the next window
    followup = FollowUp(DARREN_FOLLOWUPS[2].bot_said, bot_asked=False)
    result = await session._process_speech(np.zeros(16000, dtype=np.float32), followup)
    assert result is None and pipeline.answered == [1] and not pipeline.cancelled
    audio = await session._listen_for_followup(
        case.asker.id, settings.followup_after_answer_seconds, bot_asked=False)
    await talking

    assert abs(seconds(audio) - case.reply.seconds) < 0.1
    # Closed on his silence (2s), before a window that started over when he stopped
    # could have (4s of quiet). Under load the session's polling adds up to ~1s here.
    assert room.now() - case.reply.end < settings.followup_after_answer_seconds / SCALE


def rms_of(amplitude: int) -> float:
    """RMS of pcm(amplitude) once it's 16 kHz float."""
    return amplitude / 32768 / np.sqrt(2)
