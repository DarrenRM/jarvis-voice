"""Tests for UserAudioState.

Tests the per-user audio state dataclass including:
- Pre-buffer management with FIFO eviction
- Timing methods
- Silence attempt tracking
- Resource cleanup
"""

import time
from unittest.mock import MagicMock

import numpy as np

from src.sink.user_audio_state import UserAudioState, SAMPLE_RATE_PROCESSING


def make_audio(samples: int = 1000, value: float = 0.5) -> np.ndarray:
    """Create test audio data."""
    return np.full(samples, value, dtype=np.float32)


def make_audio_seconds(seconds: float, value: float = 0.5) -> np.ndarray:
    """Create test audio of specified duration."""
    samples = int(seconds * SAMPLE_RATE_PROCESSING)
    return np.full(samples, value, dtype=np.float32)


class TestUserAudioStateInit:
    """Test UserAudioState initialization."""

    def test_init_with_user_id(self):
        """State initializes with user ID."""
        state = UserAudioState(user_id=12345)
        assert state.user_id == 12345

    def test_init_defaults(self):
        """State has correct defaults."""
        state = UserAudioState(user_id=123)
        assert state.detector is None
        assert state.pre_buffer == []
        assert state.silence_attempts == 0
        assert state.last_seen > 0  # Has timestamp

    def test_init_max_pre_buffer_samples(self):
        """Max pre-buffer samples calculated correctly."""
        state = UserAudioState(user_id=123)
        # 1.5 seconds at 16kHz = 24000 samples
        assert state.max_pre_buffer_samples == 24000


class TestPreBufferManagement:
    """Test pre-buffer add/get/clear operations."""

    def test_add_to_pre_buffer(self):
        """Audio can be added to pre-buffer."""
        state = UserAudioState(user_id=123)
        audio = make_audio(1000)

        state.add_to_pre_buffer(audio)

        assert len(state.pre_buffer) == 1
        assert len(state.pre_buffer[0]) == 1000

    def test_add_multiple_chunks(self):
        """Multiple chunks can be added."""
        state = UserAudioState(user_id=123)

        state.add_to_pre_buffer(make_audio(500))
        state.add_to_pre_buffer(make_audio(500))
        state.add_to_pre_buffer(make_audio(500))

        assert len(state.pre_buffer) == 3

    def test_get_pre_buffer_empty(self):
        """Get empty pre-buffer returns empty array."""
        state = UserAudioState(user_id=123)

        result = state.get_pre_buffer()

        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_get_pre_buffer_concatenates(self):
        """Get pre-buffer concatenates all chunks."""
        state = UserAudioState(user_id=123)
        state.add_to_pre_buffer(make_audio(100, 0.1))
        state.add_to_pre_buffer(make_audio(200, 0.2))

        result = state.get_pre_buffer()

        assert len(result) == 300
        assert result[0] == 0.1
        assert result[100] == 0.2

    def test_get_pre_buffer_duration(self):
        """Pre-buffer duration calculated correctly."""
        state = UserAudioState(user_id=123)
        state.add_to_pre_buffer(make_audio_seconds(0.5))

        duration = state.get_pre_buffer_duration()

        assert abs(duration - 0.5) < 0.01

    def test_clear_pre_buffer(self):
        """Clear empties the pre-buffer."""
        state = UserAudioState(user_id=123)
        state.add_to_pre_buffer(make_audio(1000))

        state.clear_pre_buffer()

        assert len(state.pre_buffer) == 0
        assert len(state.get_pre_buffer()) == 0

    def test_add_copies_audio(self):
        """Add makes a copy of audio (doesn't store reference)."""
        state = UserAudioState(user_id=123)
        audio = make_audio(100, 0.5)

        state.add_to_pre_buffer(audio)
        audio[0] = 0.9  # Modify original

        # Stored copy should be unchanged
        assert state.pre_buffer[0][0] == 0.5


class TestPreBufferFIFOEviction:
    """Test FIFO eviction when pre-buffer exceeds 1.5s limit.

    Critical edge case from refactor-edge-cases.md Section 7.2.
    """

    def test_fifo_eviction_triggers_at_limit(self):
        """FIFO eviction triggers when exceeding 1.5s."""
        state = UserAudioState(user_id=123)

        # Add 2 seconds of audio (exceeds 1.5s limit)
        state.add_to_pre_buffer(make_audio_seconds(2.0))

        # Should have trimmed to ~1.5s
        total_samples = sum(len(chunk) for chunk in state.pre_buffer)
        assert total_samples <= state.max_pre_buffer_samples

    def test_fifo_eviction_removes_oldest_first(self):
        """FIFO removes oldest chunks first."""
        state = UserAudioState(user_id=123)

        # Add chunks with distinct values
        state.add_to_pre_buffer(make_audio_seconds(1.0, 0.1))  # Old
        state.add_to_pre_buffer(make_audio_seconds(1.0, 0.2))  # New

        # Should have evicted the older chunk (0.1 values)
        result = state.get_pre_buffer()

        # Recent audio should be preserved
        assert result[-1] == 0.2  # Most recent chunk's value

    def test_fifo_eviction_multiple_small_chunks(self):
        """FIFO works with many small chunks."""
        state = UserAudioState(user_id=123)

        # Add many 0.1s chunks (total 2s, exceeds limit)
        for i in range(20):
            state.add_to_pre_buffer(make_audio_seconds(0.1, float(i) / 20))

        # Should have evicted oldest
        duration = state.get_pre_buffer_duration()
        assert duration <= 1.5

    def test_exact_limit_no_eviction(self):
        """No eviction when at exactly 1.5s."""
        state = UserAudioState(user_id=123)

        state.add_to_pre_buffer(make_audio_seconds(1.5))

        # Should not have evicted anything
        duration = state.get_pre_buffer_duration()
        assert abs(duration - 1.5) < 0.01


class TestTimingMethods:
    """Test last_seen tracking and time calculations."""

    def test_update_last_seen(self):
        """Update last seen sets current time."""
        state = UserAudioState(user_id=123)
        old_time = state.last_seen

        time.sleep(0.01)  # Small delay
        state.update_last_seen()

        assert state.last_seen > old_time

    def test_time_since_seen(self):
        """Time since seen calculates correctly."""
        state = UserAudioState(user_id=123)
        state.last_seen = time.time() - 1.0  # 1 second ago

        elapsed = state.time_since_seen()

        assert abs(elapsed - 1.0) < 0.1  # Allow some tolerance

    def test_time_since_seen_just_updated(self):
        """Time since seen is near zero after update."""
        state = UserAudioState(user_id=123)
        state.update_last_seen()

        elapsed = state.time_since_seen()

        assert elapsed < 0.1


class TestSilenceAttemptTracking:
    """Test silence padding attempt tracking.

    Critical edge case from refactor-edge-cases.md Section 2.3:
    Max 5 attempts per user per gap to prevent infinite loops.
    """

    def test_increment_silence_attempts(self):
        """Increment increases counter and returns new value."""
        state = UserAudioState(user_id=123)

        result1 = state.increment_silence_attempts()
        result2 = state.increment_silence_attempts()

        assert result1 == 1
        assert result2 == 2
        assert state.silence_attempts == 2

    def test_reset_silence_attempts(self):
        """Reset clears counter to zero."""
        state = UserAudioState(user_id=123)
        state.silence_attempts = 3

        state.reset_silence_attempts()

        assert state.silence_attempts == 0

    def test_silence_attempts_start_at_zero(self):
        """Silence attempts start at zero."""
        state = UserAudioState(user_id=123)
        assert state.silence_attempts == 0


class TestDetectorManagement:
    """Test wake word detector handling."""

    def test_detector_initially_none(self):
        """Detector is None initially."""
        state = UserAudioState(user_id=123)
        assert state.detector is None

    def test_detector_can_be_set(self):
        """Detector can be assigned."""
        state = UserAudioState(user_id=123)
        mock_detector = MagicMock()

        state.detector = mock_detector

        assert state.detector is mock_detector

    def test_reset_detector_calls_reset(self):
        """Reset detector calls detector.reset()."""
        state = UserAudioState(user_id=123)
        mock_detector = MagicMock()
        state.detector = mock_detector

        state.reset_detector()

        mock_detector.reset.assert_called_once()

    def test_reset_detector_with_none(self):
        """Reset detector is safe when detector is None."""
        state = UserAudioState(user_id=123)
        state.detector = None

        # Should not raise
        state.reset_detector()


class TestCleanup:
    """Test resource cleanup."""

    def test_cleanup_resets_detector(self):
        """Cleanup resets and clears detector."""
        state = UserAudioState(user_id=123)
        mock_detector = MagicMock()
        state.detector = mock_detector

        state.cleanup()

        mock_detector.reset.assert_called_once()
        assert state.detector is None

    def test_cleanup_clears_pre_buffer(self):
        """Cleanup clears pre-buffer."""
        state = UserAudioState(user_id=123)
        state.add_to_pre_buffer(make_audio(1000))

        state.cleanup()

        assert len(state.pre_buffer) == 0

    def test_cleanup_resets_silence_attempts(self):
        """Cleanup resets silence attempts."""
        state = UserAudioState(user_id=123)
        state.silence_attempts = 5

        state.cleanup()

        assert state.silence_attempts == 0

    def test_cleanup_preserves_user_id(self):
        """Cleanup preserves user ID."""
        state = UserAudioState(user_id=12345)

        state.cleanup()

        assert state.user_id == 12345

    def test_cleanup_safe_when_empty(self):
        """Cleanup is safe when state is already empty."""
        state = UserAudioState(user_id=123)

        # Should not raise
        state.cleanup()


class TestRepr:
    """Test string representation."""

    def test_repr_includes_user_id(self):
        """Repr includes user ID."""
        state = UserAudioState(user_id=12345)

        repr_str = repr(state)

        assert "12345" in repr_str
        assert "UserAudioState" in repr_str
