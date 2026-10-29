"""Tests for speaker tracking."""

import pytest
from unittest.mock import MagicMock

from src.session.speaker_tracker import SpeakerTracker, SpeakerContext
from src.bot.speaker_manager import SpeakerManager, SpeakerInfo


@pytest.fixture
def mock_speaker_manager():
    """Create a mock speaker manager."""
    manager = MagicMock(spec=SpeakerManager)
    return manager


@pytest.fixture
def mock_voice_client():
    """Create a mock voice client."""
    client = MagicMock()
    client.guild = MagicMock()
    return client


@pytest.fixture
def tracker(mock_speaker_manager):
    """Create a speaker tracker with mock manager."""
    return SpeakerTracker(mock_speaker_manager)


class TestSpeakerContext:
    """Tests for SpeakerContext dataclass."""

    def test_default_values(self):
        """Context should have None values by default."""
        context = SpeakerContext()
        assert context.user_id is None
        assert context.name is None
        assert context.activity is None

    def test_with_values(self):
        """Context should store provided values."""
        context = SpeakerContext(
            user_id=12345,
            name="Alice",
            activity="Playing Minecraft",
        )
        assert context.user_id == 12345
        assert context.name == "Alice"
        assert context.activity == "Playing Minecraft"

    def test_clear(self):
        """Clear should reset all values to None."""
        context = SpeakerContext(
            user_id=12345,
            name="Alice",
            activity="Playing Minecraft",
        )
        context.clear()
        assert context.user_id is None
        assert context.name is None
        assert context.activity is None


class TestSpeakerTracker:
    """Tests for SpeakerTracker class."""

    def test_initial_state(self, tracker):
        """Tracker should start with no speaker."""
        assert tracker.current_speaker_id is None
        assert tracker.current_speaker_name is None
        assert tracker.current_speaker_activity is None

    def test_context_property(self, tracker):
        """Context property should return SpeakerContext."""
        context = tracker.context
        assert isinstance(context, SpeakerContext)

    def test_set_speaker_updates_context(self, tracker, mock_speaker_manager, mock_voice_client):
        """set_speaker should update all context fields."""
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(
            user_id=12345,
            name="Bob",
            activity="Watching YouTube",
        )

        tracker.set_speaker(12345, mock_voice_client)

        assert tracker.current_speaker_id == 12345
        assert tracker.current_speaker_name == "Bob"
        assert tracker.current_speaker_activity == "Watching YouTube"

    def test_set_speaker_calls_manager(self, tracker, mock_speaker_manager, mock_voice_client):
        """set_speaker should delegate to SpeakerManager."""
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(None, None, None)

        tracker.set_speaker(99999, mock_voice_client)

        mock_speaker_manager.resolve_speaker.assert_called_once_with(
            99999, mock_voice_client.guild
        )

    def test_set_speaker_with_none_client(self, tracker, mock_speaker_manager):
        """set_speaker with None voice_client should pass None guild."""
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(None, None, None)

        tracker.set_speaker(12345, None)

        mock_speaker_manager.resolve_speaker.assert_called_once_with(12345, None)

    def test_set_speaker_with_none_user_id(self, tracker, mock_speaker_manager, mock_voice_client):
        """set_speaker with None user_id should still update context."""
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(None, None, None)

        tracker.set_speaker(None, mock_voice_client)

        assert tracker.current_speaker_id is None
        assert tracker.current_speaker_name is None
        assert tracker.current_speaker_activity is None

    def test_clear_resets_context(self, tracker, mock_speaker_manager, mock_voice_client):
        """clear should reset speaker context."""
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(
            user_id=12345,
            name="Alice",
            activity="Playing",
        )
        tracker.set_speaker(12345, mock_voice_client)

        tracker.clear()

        assert tracker.current_speaker_id is None
        assert tracker.current_speaker_name is None
        assert tracker.current_speaker_activity is None

    def test_set_speaker_overwrites_previous(self, tracker, mock_speaker_manager, mock_voice_client):
        """set_speaker should overwrite previous speaker."""
        # First speaker
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(
            user_id=11111,
            name="Alice",
            activity="Playing Game A",
        )
        tracker.set_speaker(11111, mock_voice_client)

        # Second speaker
        mock_speaker_manager.resolve_speaker.return_value = SpeakerInfo(
            user_id=22222,
            name="Bob",
            activity="Playing Game B",
        )
        tracker.set_speaker(22222, mock_voice_client)

        assert tracker.current_speaker_id == 22222
        assert tracker.current_speaker_name == "Bob"
        assert tracker.current_speaker_activity == "Playing Game B"


class TestSpeakerTrackerFindUser:
    """Tests for find_user_by_name method."""

    def test_find_user_by_name_delegates_to_manager(self, tracker, mock_speaker_manager, mock_voice_client):
        """find_user_by_name should delegate to SpeakerManager."""
        mock_speaker_manager.find_user_by_name.return_value = (12345, "Alice")

        result = tracker.find_user_by_name("alice", mock_voice_client)

        mock_speaker_manager.find_user_by_name.assert_called_once_with(
            "alice", mock_voice_client.guild
        )
        assert result == (12345, "Alice")

    def test_find_user_by_name_not_found(self, tracker, mock_speaker_manager, mock_voice_client):
        """find_user_by_name should return None when not found."""
        mock_speaker_manager.find_user_by_name.return_value = None

        result = tracker.find_user_by_name("unknown", mock_voice_client)

        assert result is None

    def test_find_user_by_name_with_none_client(self, tracker, mock_speaker_manager):
        """find_user_by_name with None voice_client should pass None guild."""
        mock_speaker_manager.find_user_by_name.return_value = None

        tracker.find_user_by_name("alice", None)

        mock_speaker_manager.find_user_by_name.assert_called_once_with("alice", None)

    def test_find_user_by_name_exact_match(self, tracker, mock_speaker_manager, mock_voice_client):
        """find_user_by_name should work with exact match."""
        mock_speaker_manager.find_user_by_name.return_value = (99999, "ExactName")

        result = tracker.find_user_by_name("ExactName", mock_voice_client)

        assert result == (99999, "ExactName")

    def test_find_user_by_name_partial_match(self, tracker, mock_speaker_manager, mock_voice_client):
        """find_user_by_name should work with partial match."""
        mock_speaker_manager.find_user_by_name.return_value = (88888, "PartialNameHere")

        result = tracker.find_user_by_name("Partial", mock_voice_client)

        assert result == (88888, "PartialNameHere")


class TestSpeakerTrackerIntegration:
    """Integration-style tests with real SpeakerManager."""

    def test_with_real_speaker_manager(self):
        """Test with actual SpeakerManager instance."""
        user_names = {12345: "Alice", 67890: "Bob"}
        manager = SpeakerManager(user_names)
        tracker = SpeakerTracker(manager)

        # Set speaker with known user
        tracker.set_speaker(12345, None)

        assert tracker.current_speaker_id == 12345
        assert tracker.current_speaker_name == "Alice"
        assert tracker.current_speaker_activity is None  # No guild to get activity

    def test_with_real_manager_unknown_user(self):
        """Test with unknown user ID."""
        user_names = {12345: "Alice"}
        manager = SpeakerManager(user_names)
        tracker = SpeakerTracker(manager)

        # Set speaker with unknown user
        tracker.set_speaker(99999, None)

        assert tracker.current_speaker_id == 99999
        assert tracker.current_speaker_name is None
        assert tracker.current_speaker_activity is None

    def test_find_user_with_real_manager(self):
        """Test find_user_by_name with real manager."""
        user_names = {12345: "Alice", 67890: "Bob"}
        manager = SpeakerManager(user_names)
        tracker = SpeakerTracker(manager)

        # Exact match
        result = tracker.find_user_by_name("Alice", None)
        assert result == (12345, "Alice")

        # Partial match (case insensitive)
        result = tracker.find_user_by_name("bob", None)
        assert result == (67890, "Bob")

        # Not found
        result = tracker.find_user_by_name("Casey", None)
        assert result is None
