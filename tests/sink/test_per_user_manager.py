"""Tests for PerUserAudioManager.

Tests the lifecycle management of per-user audio states including:
- State creation and retrieval
- Lazy detector initialization
- Max detector limits
- User cleanup
- Batch operations
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sink.per_user_manager import PerUserAudioManager
from src.sink.user_audio_state import UserAudioState


def make_audio(samples: int = 1000) -> np.ndarray:
    """Create test audio data."""
    return np.full(samples, 0.5, dtype=np.float32)


class TestPerUserManagerInit:
    """Test PerUserAudioManager initialization."""

    def test_init_defaults(self):
        """Manager initializes with correct defaults."""
        manager = PerUserAudioManager()

        assert len(manager) == 0
        assert manager._wake_model_name == "hey_jarvis"
        assert manager._wake_threshold == 0.5
        assert manager._wake_sample_rate == 16000
        assert manager._max_detectors == 10

    def test_init_custom_config(self):
        """Manager accepts custom configuration."""
        manager = PerUserAudioManager(
            wake_model_name="hey_mycroft",
            wake_threshold=0.7,
            wake_sample_rate=8000,
            max_detectors=5,
        )

        assert manager._wake_model_name == "hey_mycroft"
        assert manager._wake_threshold == 0.7
        assert manager._wake_sample_rate == 8000
        assert manager._max_detectors == 5


class TestGetOrCreate:
    """Test get_or_create functionality."""

    def test_create_new_user(self):
        """Creates new state for unknown user."""
        manager = PerUserAudioManager()

        state = manager.get_or_create(user_id=12345)

        assert isinstance(state, UserAudioState)
        assert state.user_id == 12345
        assert len(manager) == 1

    def test_return_existing_user(self):
        """Returns existing state for known user."""
        manager = PerUserAudioManager()
        state1 = manager.get_or_create(user_id=12345)
        state1.silence_attempts = 3  # Modify it

        state2 = manager.get_or_create(user_id=12345)

        assert state1 is state2
        assert state2.silence_attempts == 3  # Same instance

    def test_multiple_users(self):
        """Handles multiple users independently."""
        manager = PerUserAudioManager()

        state1 = manager.get_or_create(user_id=111)
        state2 = manager.get_or_create(user_id=222)
        state3 = manager.get_or_create(user_id=333)

        assert len(manager) == 3
        assert state1.user_id == 111
        assert state2.user_id == 222
        assert state3.user_id == 333


class TestGet:
    """Test get functionality."""

    def test_get_existing_user(self):
        """Returns state for existing user."""
        manager = PerUserAudioManager()
        manager.get_or_create(user_id=12345)

        state = manager.get(12345)

        assert state is not None
        assert state.user_id == 12345

    def test_get_unknown_user(self):
        """Returns None for unknown user."""
        manager = PerUserAudioManager()

        state = manager.get(99999)

        assert state is None


class TestEnsureDetector:
    """Test lazy detector initialization."""

    def test_creates_detector(self):
        """Creates detector when called."""
        manager = PerUserAudioManager()
        state = manager.get_or_create(user_id=12345)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock_instance = MagicMock()
            MockDetector.return_value = mock_instance

            detector = manager.ensure_detector(state)

            MockDetector.assert_called_once_with(
                model_name="hey_jarvis",
                threshold=0.5,
                sample_rate=16000,
            )
            assert detector is mock_instance
            assert state.detector is mock_instance

    def test_returns_existing_detector(self):
        """Returns existing detector without creating new one."""
        manager = PerUserAudioManager()
        state = manager.get_or_create(user_id=12345)
        mock_detector = MagicMock()
        state.detector = mock_detector

        with patch("src.detection.WakeWordDetector") as MockDetector:
            detector = manager.ensure_detector(state)

            MockDetector.assert_not_called()
            assert detector is mock_detector

    def test_max_detectors_limit(self):
        """Respects max detectors limit."""
        manager = PerUserAudioManager(max_detectors=2)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock_instance = MagicMock()
            MockDetector.return_value = mock_instance

            # Create 2 detectors (at limit)
            state1 = manager.get_or_create(user_id=111)
            manager.ensure_detector(state1)
            state2 = manager.get_or_create(user_id=222)
            manager.ensure_detector(state2)

            # Third should fail
            state3 = manager.get_or_create(user_id=333)
            detector3 = manager.ensure_detector(state3)

            assert detector3 is None
            assert state3.detector is None
            assert MockDetector.call_count == 2

    def test_detector_count_property(self):
        """Detector count tracks active detectors."""
        manager = PerUserAudioManager()

        with patch("src.detection.WakeWordDetector") as MockDetector:
            MockDetector.return_value = MagicMock()

            assert manager.detector_count == 0

            state1 = manager.get_or_create(user_id=111)
            manager.ensure_detector(state1)
            assert manager.detector_count == 1

            state2 = manager.get_or_create(user_id=222)
            manager.ensure_detector(state2)
            assert manager.detector_count == 2


class TestCleanupUser:
    """Test user cleanup."""

    def test_cleanup_removes_user(self):
        """Cleanup removes user from manager."""
        manager = PerUserAudioManager()
        manager.get_or_create(user_id=12345)
        assert len(manager) == 1

        manager.cleanup_user(12345)

        assert len(manager) == 0
        assert manager.get(12345) is None

    def test_cleanup_calls_state_cleanup(self):
        """Cleanup calls state cleanup method."""
        manager = PerUserAudioManager()
        state = manager.get_or_create(user_id=12345)
        mock_detector = MagicMock()
        state.detector = mock_detector
        state.add_to_pre_buffer(make_audio(1000))

        manager.cleanup_user(12345)

        mock_detector.reset.assert_called_once()

    def test_cleanup_unknown_user_safe(self):
        """Cleanup is safe for unknown user."""
        manager = PerUserAudioManager()

        # Should not raise
        manager.cleanup_user(99999)


class TestCleanupAll:
    """Test cleanup_all functionality."""

    def test_cleanup_all_removes_all_users(self):
        """Cleanup all removes all users."""
        manager = PerUserAudioManager()
        manager.get_or_create(user_id=111)
        manager.get_or_create(user_id=222)
        manager.get_or_create(user_id=333)
        assert len(manager) == 3

        manager.cleanup_all()

        assert len(manager) == 0

    def test_cleanup_all_cleans_each_state(self):
        """Cleanup all calls cleanup on each state."""
        manager = PerUserAudioManager()
        state1 = manager.get_or_create(user_id=111)
        state2 = manager.get_or_create(user_id=222)
        mock1, mock2 = MagicMock(), MagicMock()
        state1.detector = mock1
        state2.detector = mock2

        manager.cleanup_all()

        mock1.reset.assert_called_once()
        mock2.reset.assert_called_once()


class TestBatchOperations:
    """Test batch operations on all users."""

    def test_reset_all_detectors(self):
        """Reset all detectors resets each detector."""
        manager = PerUserAudioManager()
        state1 = manager.get_or_create(user_id=111)
        state2 = manager.get_or_create(user_id=222)
        mock1, mock2 = MagicMock(), MagicMock()
        state1.detector = mock1
        state2.detector = mock2

        manager.reset_all_detectors()

        mock1.reset.assert_called_once()
        mock2.reset.assert_called_once()

    def test_clear_all_silence_attempts(self):
        """Clear all silence attempts resets counters."""
        manager = PerUserAudioManager()
        state1 = manager.get_or_create(user_id=111)
        state2 = manager.get_or_create(user_id=222)
        state1.silence_attempts = 3
        state2.silence_attempts = 5

        manager.clear_all_silence_attempts()

        assert state1.silence_attempts == 0
        assert state2.silence_attempts == 0

    def test_clear_all_pre_buffers(self):
        """Clear all pre-buffers clears each buffer."""
        manager = PerUserAudioManager()
        state1 = manager.get_or_create(user_id=111)
        state2 = manager.get_or_create(user_id=222)
        state1.add_to_pre_buffer(make_audio(1000))
        state2.add_to_pre_buffer(make_audio(2000))

        manager.clear_all_pre_buffers()

        assert len(state1.pre_buffer) == 0
        assert len(state2.pre_buffer) == 0


class TestIteration:
    """Test iteration over users."""

    def test_iterate_over_users(self):
        """Can iterate over all user states."""
        manager = PerUserAudioManager()
        manager.get_or_create(user_id=111)
        manager.get_or_create(user_id=222)
        manager.get_or_create(user_id=333)

        states = list(manager)

        assert len(states) == 3
        user_ids = {s.user_id for s in states}
        assert user_ids == {111, 222, 333}

    def test_len(self):
        """Length returns user count."""
        manager = PerUserAudioManager()

        assert len(manager) == 0

        manager.get_or_create(user_id=111)
        assert len(manager) == 1

        manager.get_or_create(user_id=222)
        assert len(manager) == 2

    def test_user_ids_property(self):
        """User IDs property returns set of IDs."""
        manager = PerUserAudioManager()
        manager.get_or_create(user_id=111)
        manager.get_or_create(user_id=222)

        ids = manager.user_ids

        assert ids == {111, 222}
