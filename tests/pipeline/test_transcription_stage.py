"""Tests for TranscriptionStage.

Tests the STT and wake word stripping logic from voice_session.py:956-984.
"""

import re
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.pipeline.base import PipelineContext
from src.pipeline.transcription_stage import (
    ConfigurableTranscriptionStage,
    TranscriptionResult,
    TranscriptionStage,
)


@pytest.fixture
def mock_stt():
    """Create a mock STT client."""
    stt = MagicMock()
    stt.transcribe_audio = AsyncMock(return_value="hey jarvis what's the weather")
    return stt


@pytest.fixture
def stage(mock_stt):
    """Create a TranscriptionStage with mock STT."""
    return TranscriptionStage(mock_stt)


@pytest.fixture
def context():
    """Create a pipeline context."""
    return PipelineContext(speaker_name="Alice")


def make_audio(samples: int = 1000) -> np.ndarray:
    """Create test audio data."""
    return np.zeros(samples, dtype=np.float32)


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_normal_result(self):
        """Normal result has text and raw_text."""
        result = TranscriptionResult(text="hello", raw_text="hey jarvis hello")

        assert result.text == "hello"
        assert result.raw_text == "hey jarvis hello"
        assert result.wake_word_only is False
        assert result.is_empty is False

    def test_empty_result(self):
        """Empty result factory method."""
        result = TranscriptionResult.empty()

        assert result.text == ""
        assert result.raw_text == ""
        assert result.is_empty is True
        assert result.wake_word_only is False

    def test_wake_only_result(self):
        """Wake-word-only result factory method."""
        result = TranscriptionResult.wake_only("hey jarvis")

        assert result.text == ""
        assert result.raw_text == "hey jarvis"
        assert result.wake_word_only is True
        assert result.is_empty is False


class TestTranscriptionStage:
    """Test TranscriptionStage functionality."""

    def test_name(self, stage):
        """Stage has correct name."""
        assert stage.name == "transcription"

    @pytest.mark.asyncio
    async def test_normal_transcription(self, stage, mock_stt, context):
        """Normal transcription strips wake word and returns text."""
        mock_stt.transcribe_audio.return_value = "hey jarvis what's the weather"
        audio = make_audio()

        result = await stage.process(audio, context)

        assert result.text == "what's the weather"
        assert result.raw_text == "hey jarvis what's the weather"
        assert result.wake_word_only is False
        assert result.is_empty is False
        mock_stt.transcribe_audio.assert_called_once_with(audio)

    @pytest.mark.asyncio
    async def test_no_speech(self, stage, mock_stt, context):
        """Empty transcription returns empty result."""
        mock_stt.transcribe_audio.return_value = ""

        result = await stage.process(make_audio(), context)

        assert result.is_empty is True

    @pytest.mark.asyncio
    async def test_whitespace_only(self, stage, mock_stt, context):
        """Whitespace-only transcription returns empty result."""
        mock_stt.transcribe_audio.return_value = "   "

        result = await stage.process(make_audio(), context)

        assert result.is_empty is True

    @pytest.mark.asyncio
    async def test_wake_word_only(self, stage, mock_stt, context):
        """Wake word only returns wake_word_only result."""
        mock_stt.transcribe_audio.return_value = "hey jarvis"

        result = await stage.process(make_audio(), context)

        assert result.wake_word_only is True
        assert result.text == ""
        assert result.raw_text == "hey jarvis"

    @pytest.mark.asyncio
    async def test_wake_word_with_punctuation(self, stage, mock_stt, context):
        """Wake word with trailing punctuation still detected as wake-only."""
        test_cases = [
            "hey jarvis.",
            "hey jarvis!",
            "hey jarvis?",
            "hey jarvis,",
            "hey, jarvis",
        ]

        for transcription in test_cases:
            mock_stt.transcribe_audio.return_value = transcription

            result = await stage.process(make_audio(), context)

            assert result.wake_word_only is True, f"Failed for: {transcription}"

    @pytest.mark.asyncio
    async def test_short_text_after_wake_word(self, stage, mock_stt, context):
        """Text shorter than MIN_TEXT_LENGTH treated as wake-only."""
        mock_stt.transcribe_audio.return_value = "hey jarvis hi"  # "hi" is only 2 chars

        result = await stage.process(make_audio(), context)

        assert result.wake_word_only is True

    @pytest.mark.asyncio
    async def test_minimum_length_text(self, stage, mock_stt, context):
        """Text at MIN_TEXT_LENGTH is accepted."""
        mock_stt.transcribe_audio.return_value = "hey jarvis yes"  # "yes" is 3 chars

        result = await stage.process(make_audio(), context)

        assert result.wake_word_only is False
        assert result.text == "yes"

    # ========== Wake Word Variations ==========

    @pytest.mark.asyncio
    async def test_hey_jarvis_lowercase(self, stage, mock_stt, context):
        """Lowercase wake word is stripped."""
        mock_stt.transcribe_audio.return_value = "hey jarvis hello"

        result = await stage.process(make_audio(), context)

        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_hey_jarvis_uppercase(self, stage, mock_stt, context):
        """Uppercase wake word is stripped."""
        mock_stt.transcribe_audio.return_value = "HEY JARVIS hello"

        result = await stage.process(make_audio(), context)

        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_hey_jarvis_mixed_case(self, stage, mock_stt, context):
        """Mixed case wake word is stripped."""
        mock_stt.transcribe_audio.return_value = "Hey Jarvis hello"

        result = await stage.process(make_audio(), context)

        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_hey_comma_jarvis(self, stage, mock_stt, context):
        """'hey, jarvis' variation is stripped."""
        mock_stt.transcribe_audio.return_value = "hey, jarvis hello"

        result = await stage.process(make_audio(), context)

        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_a_jarvis(self, stage, mock_stt, context):
        """'a jarvis' variation is stripped."""
        mock_stt.transcribe_audio.return_value = "a jarvis hello"

        result = await stage.process(make_audio(), context)

        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_no_wake_word(self, stage, mock_stt, context):
        """Text without wake word is returned as-is."""
        mock_stt.transcribe_audio.return_value = "what's the weather today"

        result = await stage.process(make_audio(), context)

        assert result.text == "what's the weather today"
        assert result.raw_text == "what's the weather today"

    @pytest.mark.asyncio
    async def test_wake_word_not_at_start(self, stage, mock_stt, context):
        """Wake word not at start is not stripped."""
        mock_stt.transcribe_audio.return_value = "please hey jarvis help"

        result = await stage.process(make_audio(), context)

        assert result.text == "please hey jarvis help"

    # ========== Edge Cases ==========

    @pytest.mark.asyncio
    async def test_preserves_trailing_punctuation(self, stage, mock_stt, context):
        """Trailing punctuation in text is preserved."""
        mock_stt.transcribe_audio.return_value = "hey jarvis what's the weather?"

        result = await stage.process(make_audio(), context)

        assert result.text == "what's the weather?"

    @pytest.mark.asyncio
    async def test_preserves_internal_punctuation(self, stage, mock_stt, context):
        """Internal punctuation in text is preserved."""
        mock_stt.transcribe_audio.return_value = "hey jarvis tell me about cats, dogs, and birds"

        result = await stage.process(make_audio(), context)

        assert result.text == "tell me about cats, dogs, and birds"

    @pytest.mark.asyncio
    async def test_whitespace_normalization(self, stage, mock_stt, context):
        """Extra whitespace is normalized."""
        mock_stt.transcribe_audio.return_value = "  hey jarvis   hello world  "

        result = await stage.process(make_audio(), context)

        assert result.text == "hello world"


class TestCustomWakePattern:
    """Test TranscriptionStage with custom wake patterns."""

    @pytest.mark.asyncio
    async def test_custom_wake_pattern(self, mock_stt):
        """Custom wake pattern is used."""
        custom_pattern = re.compile(r"^(alexa|ok google)[,\s]*", re.IGNORECASE)
        stage = TranscriptionStage(mock_stt, wake_pattern=custom_pattern)
        context = PipelineContext()

        mock_stt.transcribe_audio.return_value = "alexa what time is it"

        result = await stage.process(make_audio(), context)

        assert result.text == "what time is it"

    @pytest.mark.asyncio
    async def test_custom_pattern_doesnt_match_default(self, mock_stt):
        """Custom pattern doesn't match default wake word."""
        custom_pattern = re.compile(r"^(alexa)[,\s]*", re.IGNORECASE)
        stage = TranscriptionStage(mock_stt, wake_pattern=custom_pattern)
        context = PipelineContext()

        mock_stt.transcribe_audio.return_value = "hey jarvis hello"

        result = await stage.process(make_audio(), context)

        # "hey jarvis hello" not stripped - kept as-is
        assert result.text == "hey jarvis hello"


class TestConfigurableTranscriptionStage:
    """Test ConfigurableTranscriptionStage."""

    @pytest.mark.asyncio
    async def test_custom_wake_words(self, mock_stt):
        """Custom wake words list works."""
        stage = ConfigurableTranscriptionStage(
            mock_stt, wake_words=["computer", "jarvis"]
        )
        context = PipelineContext()

        mock_stt.transcribe_audio.return_value = "computer what is the date"

        result = await stage.process(make_audio(), context)

        assert result.text == "what is the date"

    @pytest.mark.asyncio
    async def test_update_wake_words(self, mock_stt):
        """Wake words can be updated at runtime."""
        stage = ConfigurableTranscriptionStage(mock_stt)
        context = PipelineContext()

        # Default pattern
        mock_stt.transcribe_audio.return_value = "hey jarvis hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hello"

        # Update to new wake words
        stage.set_wake_words(["computer"])

        mock_stt.transcribe_audio.return_value = "computer hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hello"

        # Old wake word no longer works
        mock_stt.transcribe_audio.return_value = "hey jarvis hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hey jarvis hello"  # Not stripped

    @pytest.mark.asyncio
    async def test_empty_wake_words_uses_default(self, mock_stt):
        """Empty wake words list reverts to default."""
        stage = ConfigurableTranscriptionStage(mock_stt, wake_words=["computer"])
        context = PipelineContext()

        # Custom wake word works
        mock_stt.transcribe_audio.return_value = "computer hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hello"

        # Revert to default
        stage.set_wake_words([])

        # Default works again
        mock_stt.transcribe_audio.return_value = "hey jarvis hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_special_characters_in_wake_words(self, mock_stt):
        """Special characters in wake words are escaped."""
        stage = ConfigurableTranscriptionStage(
            mock_stt, wake_words=["hey. jarvis"]  # Dot should be literal
        )
        context = PipelineContext()

        # Exact match works
        mock_stt.transcribe_audio.return_value = "hey. jarvis hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hello"

        # Dot is not regex wildcard
        mock_stt.transcribe_audio.return_value = "heyx jarvis hello"
        result = await stage.process(make_audio(), context)
        # Not stripped because "heyx" doesn't match "hey."
        assert result.text == "heyx jarvis hello"


class TestMinTextLength:
    """Test minimum text length configuration."""

    @pytest.mark.asyncio
    async def test_custom_min_length(self, mock_stt):
        """Custom minimum text length is respected."""
        stage = TranscriptionStage(mock_stt, min_text_length=5)
        context = PipelineContext()

        # 4 chars - too short
        mock_stt.transcribe_audio.return_value = "hey jarvis test"
        result = await stage.process(make_audio(), context)
        assert result.wake_word_only is True

        # 5 chars - exactly enough
        mock_stt.transcribe_audio.return_value = "hey jarvis hello"
        result = await stage.process(make_audio(), context)
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_zero_min_length(self, mock_stt):
        """Zero minimum length accepts any text."""
        stage = TranscriptionStage(mock_stt, min_text_length=0)
        context = PipelineContext()

        mock_stt.transcribe_audio.return_value = "hey jarvis a"
        result = await stage.process(make_audio(), context)
        assert result.text == "a"
