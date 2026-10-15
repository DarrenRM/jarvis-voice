"""Official game notes added to the prompt while the game is in play."""

import datetime as dt
import json
from unittest.mock import MagicMock

from src.llm.game_notes import STICKY_SECONDS, GameNotes
from src.pipeline.base import PipelineContext
from src.pipeline.response_stage import ResponseStage


def make_notes(tmp_path, now=None):
    game = tmp_path / "enshrouded"
    game.mkdir()
    (game / "1.0.txt").write_text("Max level is now 50.", encoding="utf-8")
    (game / "info.json").write_text(json.dumps({
        "game": "Enshrouded", "version": "1.0", "released": "2026-10-15", "notes": "1.0.txt",
        "aliases": ["enshrouded", "enshrowded"], "source": "Keen Games"}), encoding="utf-8")
    clock = now or [0.0]
    return GameNotes.load(tmp_path, clock=lambda: clock[0],
                          today=lambda: dt.date(2026, 10, 16)), clock


def test_playing_the_game(tmp_path):
    notes, _ = make_notes(tmp_path)
    block = notes.prompt_block("what's the max level now?", "Playing Enshrouded")
    assert "Max level is now 50." in block
    assert "came out on October 15, 2026" in block
    assert "notes are right" in block and "amounts from older" in block


def test_someone_else_in_the_channel_playing(tmp_path):
    """2026-10-15: Pat shows no game on Discord; Darren was playing Enshrouded."""
    notes, _ = make_notes(tmp_path)
    assert notes.prompt_block("where are the necromancy skills?", None,
                              ["Listening Spotify", "Playing Enshrouded"])
    (tmp_path / "fresh").mkdir()
    notes, _ = make_notes(tmp_path / "fresh")  # no "came up recently"
    assert notes.prompt_block("where are the necromancy skills?", None,
                              ["Playing Valheim"]) == ""


def test_named_even_misheard(tmp_path):
    notes, _ = make_notes(tmp_path)
    assert notes.prompt_block("max level in Enshrowded?", None)


def test_other_questions_get_nothing(tmp_path):
    notes, _ = make_notes(tmp_path)
    assert notes.prompt_block("what's twelve times eight?", "Playing Valheim") == ""
    assert notes.prompt_block("unshrouded truth", None) == ""  # whole words only


def test_follow_ups_keep_the_notes_for_a_while(tmp_path):
    notes, clock = make_notes(tmp_path)
    assert notes.prompt_block("enshrouded max level?", None)
    clock[0] = 60.0
    assert notes.prompt_block("and the new biome?", None)  # follow-up, game not named
    clock[0] = 60.0 + STICKY_SECONDS + 1
    assert notes.prompt_block("and the new biome?", None) == ""


def test_missing_or_broken_dir_loads_nothing(tmp_path):
    assert GameNotes.load(tmp_path / "nope").sets == []
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "info.json").write_text("{", encoding="utf-8")
    assert GameNotes.load(tmp_path).sets == []


def test_real_notes_load():
    notes = GameNotes.load()
    assert [s.game for s in notes.sets] == ["Enshrouded"]
    assert "Cinder" in notes.sets[0].text


def test_response_stage_adds_notes_to_the_first_system_prompt(tmp_path):
    notes, _ = make_notes(tmp_path)
    stage = ResponseStage(MagicMock(), lambda: "BASE", game_notes=notes)
    prompt = stage._build_system_prompt(
        PipelineContext(speaker_name="Riley", speaker_activity="Playing Enshrouded"),
        "what's the max level?")
    assert prompt.startswith("BASE") and "Max level is now 50." in prompt
    plain = ResponseStage(MagicMock(), lambda: "BASE")  # no notes configured
    assert plain._build_system_prompt(PipelineContext(), "enshrouded max level?") == "BASE"


async def test_misheard_game_name_is_corrected_before_the_model_sees_it():
    from unittest.mock import AsyncMock

    from src.pipeline.transcription_stage import TranscriptionStage

    corrections = GameNotes.load().corrections()
    assert corrections["in shrouded"] == "Enshrouded"
    for raw, expected in [
        ("Hey Jarvis, where do I get in shrouded oil?", "where do I get Enshrouded oil?"),
        ("Hey Jarvis, max level in In-Shrouded?", "max level in Enshrouded?"),
        ("Hey Jarvis, is it hidden in shroud?", "is it hidden in shroud?"),
    ]:
        stt = AsyncMock()
        stt.transcribe_audio = AsyncMock(return_value=raw)
        result = await TranscriptionStage(stt, corrections=corrections).process(
            None, PipelineContext())
        assert result.text == expected
        assert result.raw_text == raw  # the follow-up check still sees what was heard
