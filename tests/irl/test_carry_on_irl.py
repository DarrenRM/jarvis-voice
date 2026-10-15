"""Sam was cut off mid-question (2026-10-15 10:26), replayed through the real sink
and VoiceSession turn code with a scripted pipeline."""

import asyncio
import logging

import numpy as np

from .cases import SAM_CUT_OFF, Speech
from .replay import (
    SAMPLE_RATE,
    SCALE,
    Room,
    ScriptedPipeline,
    listen_for_turn,
    make_session,
    make_sink,
    rms,
    scaled_settings,
    seconds,
)

CASE = SAM_CUT_OFF
WAKE, SUP, MORE = 3000, 6000, 15000  # loudness of each part, to tell them apart


async def first_turn(room: Room, sink, speech_after: Speech):
    """Wake word into the pre-buffer, then the main loop listens until he goes quiet;
    he keeps talking (speech_after) from a background task."""
    await room.say(sink, CASE.speaker, CASE.wake, WAKE)

    async def keeps_talking():
        await room.say(sink, CASE.speaker, CASE.question, SUP)
        await room.say(sink, CASE.speaker, speech_after, MORE)

    talking = asyncio.create_task(keeps_talking())
    said = await listen_for_turn(sink, CASE.speaker)
    return said, talking


async def test_sam_is_heard_out_when_he_carries_on(caplog):
    """2026-10-15 10:26, Sam (100000000000000002): "Hey Jarvis" at :26, the bot's
    "Sup?" at :27-:28, he said "Sup" at :30-:32 and paused; the turn ended on the old
    0.5 s silence. He went on about Duluth, Minnesota at :35-:44 while the bot worked on it;
    that was thrown away and the bot answered "Sup yourself..." to just "Sup".
    Now: his later speech is kept, the first answer is cancelled before it plays, and
    the turn starts over with all of it, first part first."""
    settings = scaled_settings(speaking_detection=False)
    sink = make_sink(settings)
    pipeline = ScriptedPipeline(answer_after=CASE.answer_after, answer_seconds=2.0)
    session = make_session(sink, CASE.speaker, settings, pipeline)
    room = Room(start=CASE.wake.start)

    with caplog.at_level(logging.INFO, logger="src.bot.voice_session"):
        said, talking = await first_turn(room, sink, CASE.carried_on)
        result = await session._process_speech(said)
        await talking

    assert result is None
    # The first turn: the wake word's pre-buffer and "Sup" (what Whisper heard as
    # "Hey Jarvis. Sup."), ended by the silence after it
    assert abs(seconds(said) - (CASE.wake.seconds + CASE.question.seconds)) < 0.05
    # Its answer never played; the second run got everything
    assert len(pipeline.heard) == 2
    assert pipeline.cancelled == [1] and pipeline.answered == [2]
    assert pipeline.cancel_calls >= 1
    merged = pipeline.heard[1]
    assert np.array_equal(merged[:len(said)], said)
    assert abs(seconds(merged) - seconds(said) - CASE.carried_on.seconds) < 0.1
    rest = merged[len(said):]
    sup = said[int(CASE.wake.seconds * SAMPLE_RATE):]
    assert rms(rest) > 2 * rms(sup)  # it's the Duluth question, not "Sup" again
    assert any(
        r.levelno == logging.WARNING
        and f"[CARRY-ON] {CASE.speaker.name} kept talking after the question seemed "
            "finished" in r.getMessage()
        for r in caplog.records
    )


async def test_talking_over_the_filler_does_not_start_over(caplog, monkeypatch):
    """Same turn, but he only talks while the bot plays its filler ("Sure, Sam...",
    which can come back through his mic). That isn't carrying on: no restart, the
    answer plays, and his speech is kept for the follow-up window."""
    settings = scaled_settings(speaking_detection=False)
    sink = make_sink(settings)
    # The turn ends at :34 (2 s of silence); the filler plays :34.3-:38, then the answer
    turn_end = CASE.question.end + settings.silence_seconds / SCALE
    filler = Speech(0.3, CASE.answer_after, "Sure, Sam...")  # from the turn's end
    over_filler = Speech(turn_end + 1.0, turn_end + 3.0, "(talking over the filler)")
    pipeline = ScriptedPipeline(answer_after=CASE.answer_after, answer_seconds=1.0,
                                filler=filler)
    monkeypatch.setattr(sink, "_bot_is_playing", lambda: pipeline.bot_playing)
    session = make_session(sink, CASE.speaker, settings, pipeline)
    room = Room(start=CASE.wake.start)

    with caplog.at_level(logging.INFO, logger="src.bot.voice_session"):
        said, talking = await first_turn(room, sink, over_filler)
        result = await session._process_speech(said)
        await talking

    assert result is None
    assert len(pipeline.heard) == 1 and pipeline.answered == [1] and not pipeline.cancelled
    assert sink.captured_speech(while_bot_quiet=True) == 0
    assert abs(sink.captured_speech() - over_filler.seconds) < 0.05
    assert sink.is_capturing(CASE.speaker.id, fold=True)  # the follow-up starts from it
    messages = [r.getMessage() for r in caplog.records]
    assert not any("kept talking after the question" in m for m in messages)
    assert any("talked during the answer" in m for m in messages)
