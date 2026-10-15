"""What Whisper heard and what the bot said, through the real transcription stage (with
the real game-notes corrections) and the real TTS stage and synthesizer text path."""

import numpy as np
import pytest

from src.llm.game_notes import GameNotes
from src.pipeline.base import PipelineContext
from src.pipeline.transcription_stage import TranscriptionStage
from src.pipeline.tts_stage import TTSStage
from src.tts.synthesizer import TTSSynthesizer

from .cases import TRANSCRIPTS, VERSION_REPLY


class HeardSTT:
    def __init__(self, text: str):
        self._text = text

    async def transcribe_audio(self, audio: np.ndarray) -> str:
        return self._text


@pytest.mark.parametrize("when, heard, expected", TRANSCRIPTS, ids=[t[1] for t in TRANSCRIPTS])
async def test_transcripts_reach_the_model_right(when, heard, expected):
    """2026-10-14 21:15: Whisper heard "Enshrouded" as "in shrouded", so the model (and
    its web search) got the wrong name; the game notes' misheard list fixes it. Also the
    wake word stripped from real first turns ("Hey Jarvis. Sup." from 2026-10-15 10:26
    must leave "Sup.", not be taken for the wake word alone)."""
    stage = TranscriptionStage(HeardSTT(heard), corrections=GameNotes.load().corrections())
    result = await stage.process(np.zeros(1600, dtype=np.float32), PipelineContext())
    assert not result.wake_word_only and not result.is_empty
    assert result.text == expected


class Inworld:
    """What would be sent to Inworld."""

    def __init__(self):
        self.tts = TTSSynthesizer(api_key="test-key")
        self.texts: list[str] = []

        async def synthesize(text):
            self.texts.append(text)
            return np.zeros(2400, dtype=np.float32)

        self.tts._synthesize_inworld = synthesize


async def speak(chunks: list[str]) -> list[str]:
    inworld = Inworld()

    async def stream():
        for chunk in chunks:
            yield chunk

    async def play(audio):
        pass

    await TTSStage(inworld.tts).process_stream(stream(), PipelineContext(), play)
    return inworld.texts


async def test_version_is_read_as_one_point_oh():
    """2026-10-14: the bot read "Enshrouded 1.0" as "Enshrouded one"."""
    reply, spoken = VERSION_REPLY
    assert await speak([reply]) == [spoken]


async def test_version_split_across_stream_chunks():
    """Same reply, streamed the way the model sends tokens ("1", ".", "0"). The splitter
    used to send "Enshrouded 1." on its own: "Enshrouded one." then "zero is the..."."""
    reply, spoken = VERSION_REPLY
    head, tail = reply.split("1.0")
    assert await speak([head + "1", ".", "0" + tail]) == [spoken]


async def test_sentence_ending_in_a_number_still_splits():
    """Waiting to see if "1." continues as "1.0" doesn't merge real sentences."""
    said = await speak(["You need 1", ".", " Then craft it", "."])
    assert said == ["You need 1.", "Then craft it."]
    assert await speak(["It costs 5", "."]) == ["It costs 5."]
