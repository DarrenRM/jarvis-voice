"""Tests for session phrase constants and helpers."""

import pytest

from src.session.phrases import (
    ACK_PHRASES,
    THINKING_PHRASES,
    THINKING_PHRASES_PERSONAL,
    get_random_ack,
    get_thinking_phrase,
)


class TestAckPhrases:
    """Tests for acknowledgment phrases."""

    def test_ack_phrases_not_empty(self):
        """ACK_PHRASES should contain phrases."""
        assert len(ACK_PHRASES) > 0

    def test_all_ack_phrases_are_strings(self):
        """All phrases should be non-empty strings."""
        for phrase in ACK_PHRASES:
            assert isinstance(phrase, str)
            assert len(phrase) > 0

    def test_get_random_ack_returns_valid_phrase(self):
        """get_random_ack should return a phrase from the list."""
        phrase = get_random_ack()
        assert phrase in ACK_PHRASES

    def test_get_random_ack_returns_different_phrases(self):
        """get_random_ack should return different phrases over many calls."""
        # Get many phrases and check we got at least 2 different ones
        phrases = {get_random_ack() for _ in range(100)}
        assert len(phrases) > 1


class TestThinkingPhrases:
    """Tests for thinking phrases."""

    def test_thinking_phrases_not_empty(self):
        """THINKING_PHRASES should contain phrases."""
        assert len(THINKING_PHRASES) > 0

    def test_all_thinking_phrases_are_strings(self):
        """All generic phrases should be non-empty strings."""
        for phrase in THINKING_PHRASES:
            assert isinstance(phrase, str)
            assert len(phrase) > 0

    def test_thinking_phrases_personal_not_empty(self):
        """THINKING_PHRASES_PERSONAL should contain phrases."""
        assert len(THINKING_PHRASES_PERSONAL) > 0

    def test_all_personal_phrases_have_placeholder(self):
        """All personal phrases should have {name} placeholder."""
        for phrase in THINKING_PHRASES_PERSONAL:
            assert "{name}" in phrase

    def test_get_thinking_phrase_without_name(self):
        """get_thinking_phrase without name returns generic phrase."""
        phrase = get_thinking_phrase()
        assert phrase in THINKING_PHRASES

    def test_get_thinking_phrase_with_none_name(self):
        """get_thinking_phrase with None returns generic phrase."""
        phrase = get_thinking_phrase(None)
        assert phrase in THINKING_PHRASES

    def test_get_thinking_phrase_with_name(self):
        """get_thinking_phrase with name returns personalized phrase."""
        phrase = get_thinking_phrase("Alice")
        assert "Alice" in phrase
        # Should not still have the placeholder
        assert "{name}" not in phrase

    def test_get_thinking_phrase_with_name_formats_correctly(self):
        """Personalized phrase has name inserted at placeholder position."""
        # Test multiple times to catch different phrases
        names_found = 0
        for _ in range(50):
            phrase = get_thinking_phrase("Bob")
            if "Bob" in phrase:
                names_found += 1
        # All should have the name
        assert names_found == 50

    def test_get_thinking_phrase_with_empty_string_uses_generic(self):
        """Empty string is falsy so should use generic phrase."""
        phrase = get_thinking_phrase("")
        assert phrase in THINKING_PHRASES

    def test_get_thinking_phrase_returns_different_phrases(self):
        """get_thinking_phrase should return variety over many calls."""
        # Generic phrases
        generic_phrases = {get_thinking_phrase() for _ in range(100)}
        assert len(generic_phrases) > 1

        # Personal phrases
        personal_phrases = {get_thinking_phrase("Test") for _ in range(100)}
        assert len(personal_phrases) > 1


class TestPhraseConsistency:
    """Tests for phrase list consistency."""

    def test_ack_phrases_count(self):
        """Should have expected number of ack phrases."""
        # Based on extracted constants
        assert len(ACK_PHRASES) == 10

    def test_thinking_phrases_count(self):
        """Should have expected number of generic thinking phrases."""
        assert len(THINKING_PHRASES) == 6

    def test_thinking_phrases_personal_count(self):
        """Should have expected number of personal thinking phrases."""
        assert len(THINKING_PHRASES_PERSONAL) == 9
