"""Short sentences that are the actual answer must reach TTS; fragments must not."""

import pytest

from src.tts.text_filter import is_reasoning_text


@pytest.mark.parametrize("text", [
    "[nice] 96.", "[happy] 24.", "42", "3.5", "Yes.", "No.", "[happy] Yes!", "Ten.", "Four.",
])
def test_short_answers_are_spoken(text):
    assert not is_reasoning_text(text)


@pytest.mark.parametrize("text", ["...", "].", "[......]", "So", "hm.", "We"])
def test_fragments_are_dropped(text):
    assert is_reasoning_text(text)
