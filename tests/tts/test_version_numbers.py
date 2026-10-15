"""Game versions are read the way people say them; other decimals are left alone."""

import pytest

from src.tts.text_filter import normalize_numbers_for_tts


@pytest.mark.parametrize("text, spoken", [
    ("Enshrouded 1.0 is out.", "Enshrouded one point oh is out."),
    ("the 1.0 launch", "the one point oh launch"),
    ("patch 1.0.2 notes", "patch one point oh point two notes"),
    ("update 1.05 fixed it", "update one point oh five fixed it"),
    ("v1.0 today", "v one point oh today"),
])
def test_versions(text, spoken):
    assert normalize_numbers_for_tts(text) == spoken


@pytest.mark.parametrize("text, spoken", [
    ("version 2.5 ships", "version two point five ships"),
    ("about 0.5 seconds", "about zero point five seconds"),
    ("up 10.0% today", "up 10.0% today"),
    ("It takes 42 wood.", "It takes 42 wood."),
])
def test_other_numbers_unchanged(text, spoken):
    assert normalize_numbers_for_tts(text) == spoken
