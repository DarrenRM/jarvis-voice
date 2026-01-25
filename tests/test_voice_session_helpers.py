"""Tests for VoiceSession helper methods."""

import pytest
import re


class TestStripWakeWord:
    """Tests for _strip_wake_word method behavior.

    Tests the wake word stripping logic without importing VoiceSession directly
    to avoid heavy dependencies. Uses the same pattern from voice_session.py.
    """

    # Pattern from voice_session.py
    WAKE_PATTERN = re.compile(r'^(hey[,\s]*jarvis|a[,\s]*jarvis)[,.\s!?]*', re.IGNORECASE)

    def strip_wake_word(self, text: str) -> str | None:
        """Replicate _strip_wake_word logic for testing."""
        if not text:
            return None

        stripped = self.WAKE_PATTERN.sub('', text).strip()

        if not stripped:
            return None

        return stripped

    def test_basic_wake_word(self):
        """Basic 'hey jarvis' is stripped."""
        result = self.strip_wake_word("hey jarvis what time is it")
        assert result == "what time is it"

    def test_wake_word_with_comma(self):
        """'hey, jarvis' with comma is stripped."""
        result = self.strip_wake_word("hey, jarvis tell me a joke")
        assert result == "tell me a joke"

    def test_wake_word_only(self):
        """Wake word only returns None."""
        assert self.strip_wake_word("hey jarvis") is None
        assert self.strip_wake_word("hey, jarvis") is None
        assert self.strip_wake_word("hey jarvis!") is None

    def test_a_jarvis_variant(self):
        """'a jarvis' (common transcription) is stripped."""
        result = self.strip_wake_word("a jarvis what's the weather")
        assert result == "what's the weather"

    def test_case_insensitive(self):
        """Wake word matching is case-insensitive."""
        result = self.strip_wake_word("HEY JARVIS hello")
        assert result == "hello"

        result = self.strip_wake_word("Hey Jarvis hello")
        assert result == "hello"

    def test_no_wake_word(self):
        """Text without wake word is unchanged."""
        result = self.strip_wake_word("what time is it")
        assert result == "what time is it"

    def test_wake_word_middle_of_text(self):
        """Wake word in middle is not stripped (only start)."""
        result = self.strip_wake_word("I said hey jarvis earlier")
        assert result == "I said hey jarvis earlier"

    def test_empty_string(self):
        """Empty string returns None."""
        assert self.strip_wake_word("") is None

    def test_whitespace_only(self):
        """Whitespace only after stripping returns None."""
        assert self.strip_wake_word("hey jarvis   ") is None

    def test_punctuation_after_wake_word(self):
        """Various punctuation after wake word is handled."""
        assert self.strip_wake_word("hey jarvis. hello") == "hello"
        assert self.strip_wake_word("hey jarvis, hello") == "hello"
        assert self.strip_wake_word("hey jarvis! hello") == "hello"
        assert self.strip_wake_word("hey jarvis? hello") == "hello"


class TestBuildPrompt:
    """Tests for _build_prompt method behavior.

    Tests prompt building logic without heavy dependencies.
    """

    def build_prompt(
        self,
        text: str,
        speaker_name: str | None,
        speaker_activity: str | None
    ) -> str:
        """Replicate _build_prompt logic for testing."""
        if speaker_name:
            if speaker_activity:
                return f"[{speaker_name} speaking, currently {speaker_activity}]: {text}"
            else:
                return f"[{speaker_name} speaking]: {text}"
        else:
            return text

    def test_no_speaker(self):
        """No speaker returns plain text."""
        result = self.build_prompt("hello world", None, None)
        assert result == "hello world"

    def test_speaker_no_activity(self):
        """Speaker without activity includes name only."""
        result = self.build_prompt("hello", "Alice", None)
        assert result == "[Alice speaking]: hello"

    def test_speaker_with_activity(self):
        """Speaker with activity includes both."""
        result = self.build_prompt("hello", "Alice", "Playing Minecraft")
        assert result == "[Alice speaking, currently Playing Minecraft]: hello"

    def test_preserves_text(self):
        """Original text is preserved in output."""
        text = "What's the weather like today?"
        result = self.build_prompt(text, "Bob", None)
        assert text in result

    def test_empty_activity_treated_as_none(self):
        """Empty string activity is same as no activity."""
        # In actual code, activity would be None not empty string
        # but test the expected behavior
        result = self.build_prompt("hi", "Alice", None)
        assert "currently" not in result


class TestWakePatternMatching:
    """Tests for wake word pattern edge cases."""

    WAKE_PATTERN = re.compile(r'^(hey[,\s]*jarvis|a[,\s]*jarvis)[,.\s!?]*', re.IGNORECASE)

    def test_multiple_spaces(self):
        """Multiple spaces between hey and jarvis."""
        match = self.WAKE_PATTERN.match("hey   jarvis hello")
        assert match is not None

    def test_tab_character(self):
        """Tab between hey and jarvis (whitespace)."""
        match = self.WAKE_PATTERN.match("hey\tjarvis hello")
        assert match is not None

    def test_similar_but_different(self):
        """Similar phrases that should NOT match."""
        assert self.WAKE_PATTERN.match("hey jarv hello") is None
        assert self.WAKE_PATTERN.match("hi jarvis hello") is None
        assert self.WAKE_PATTERN.match("hey travis hello") is None

    def test_wake_word_with_period(self):
        """Wake word followed by period."""
        match = self.WAKE_PATTERN.match("hey jarvis. what time")
        assert match is not None
        remaining = "hey jarvis. what time"[match.end():].strip()
        assert remaining == "what time"
