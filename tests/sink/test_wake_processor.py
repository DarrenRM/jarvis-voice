"""Tests for WakeWordProcessor.

Tests the timing-sensitive wake word detection logic including:
- Real-time detection from audio stream
- Silence padding for PTT detection
- Wake word user ID capture at detection time
- Gap window constraints (250ms-2.5s)
- Max silence attempts limit

Critical edge cases from refactor-edge-cases.md:
- Section 2.3: Wake Word Silence Padding (gap window, attempt limits)
- Section 3.5: Wake Word User ID Capture Timing
- Section 4.1: Wake Word User vs Last User Race
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sink.per_user_manager import PerUserAudioManager
from src.sink.wake_processor import (
    WakeWordProcessor,
    MIN_GAP_SECONDS,
    MAX_GAP_SECONDS,
    MAX_SILENCE_ATTEMPTS,
    SILENCE_SAMPLES,
)


def make_audio(samples: int = 1000, value: float = 0.5) -> np.ndarray:
    """Create test audio data."""
    return np.full(samples, value, dtype=np.float32)


@pytest.fixture
def mock_detector():
    """Create a mock wake word detector."""
    detector = MagicMock()
    detector.process = MagicMock(return_value=False)
    detector.flush = MagicMock(return_value=False)
    return detector


@pytest.fixture
def manager_with_mock_detector(mock_detector):
    """Create a manager that returns mock detectors."""
    manager = PerUserAudioManager()

    with patch("src.detection.WakeWordDetector", return_value=mock_detector):
        yield manager, mock_detector


class TestWakeWordProcessorInit:
    """Test WakeWordProcessor initialization."""

    def test_init(self):
        """Processor initializes correctly."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        assert processor._user_manager is manager
        assert processor.wake_word_user_id is None


class TestProcessAudio:
    """Test real-time wake word detection."""

    def test_no_detection(self):
        """Returns False when no wake word detected."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = False
            MockDetector.return_value = mock

            result = processor.process_audio(user_id=123, audio_16k=make_audio())

            assert result is False
            assert processor.wake_word_user_id is None

    def test_detection_captures_user_id(self):
        """Wake word detection captures user ID at detection time.

        Critical edge case: refactor-edge-cases.md Section 3.5
        """
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = True  # Wake word detected!
            MockDetector.return_value = mock

            result = processor.process_audio(user_id=12345, audio_16k=make_audio())

            assert result is True
            assert processor.wake_word_user_id == 12345

    def test_detection_updates_user_id(self):
        """New detection updates the captured user ID."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = True
            MockDetector.return_value = mock

            # First detection from user 111
            processor.process_audio(user_id=111, audio_16k=make_audio())
            assert processor.wake_word_user_id == 111

            # Reset for second detection
            processor.reset_wake_user()

            # Second detection from user 222
            processor.process_audio(user_id=222, audio_16k=make_audio())
            assert processor.wake_word_user_id == 222

    def test_resets_silence_attempts_on_audio(self):
        """New audio resets silence attempts counter."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            MockDetector.return_value = MagicMock(process=MagicMock(return_value=False))

            # Process audio to create state
            processor.process_audio(user_id=123, audio_16k=make_audio())

            # Manually set silence attempts
            state = manager.get(123)
            state.silence_attempts = 3

            # Process more audio - should reset attempts
            processor.process_audio(user_id=123, audio_16k=make_audio())

            assert state.silence_attempts == 0


class TestCheckIdleAudioGap:
    """Test silence padding for PTT detection.

    Critical edge cases from refactor-edge-cases.md Section 2.3.
    """

    def test_no_users(self):
        """Returns None when no users tracked."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        result = processor.check_idle_audio_gap()

        assert result is None

    def test_gap_too_short(self):
        """No silence padding for gaps < 250ms."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = True  # Would detect if called
            MockDetector.return_value = mock

            # Create state with audio just received (gap too short)
            state = manager.get_or_create(user_id=123)
            manager.ensure_detector(state)
            state.last_seen = time.time()  # Just now

            result = processor.check_idle_audio_gap()

            # Detector should not be called - gap too short
            assert result is None

    def test_gap_too_long(self):
        """No silence padding for gaps > 2.5s (stale)."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = True  # Would detect if called
            MockDetector.return_value = mock

            # Create state with stale audio
            state = manager.get_or_create(user_id=123)
            manager.ensure_detector(state)
            state.last_seen = time.time() - 3.0  # 3 seconds ago (> 2.5s)

            result = processor.check_idle_audio_gap()

            # Detector should not be called - gap too long
            assert result is None

    def test_gap_in_window_no_detection(self):
        """Silence padding called for gaps in 250ms-2.5s window."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = False
            mock.flush.return_value = False
            MockDetector.return_value = mock

            # Create state with gap in valid range
            state = manager.get_or_create(user_id=123)
            manager.ensure_detector(state)
            state.last_seen = time.time() - 1.0  # 1 second ago (in range)

            result = processor.check_idle_audio_gap()

            # Detector should be called with silence
            assert mock.process.called
            args = mock.process.call_args[0]
            assert len(args[0]) == SILENCE_SAMPLES  # 8000 samples of silence
            assert result is None

    def test_gap_in_window_detection(self):
        """Wake word detected via silence padding captures user ID.

        Critical edge case: refactor-edge-cases.md Section 2.3
        """
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = True  # Detect on silence
            MockDetector.return_value = mock

            # Create state with gap in valid range
            state = manager.get_or_create(user_id=12345)
            manager.ensure_detector(state)
            state.last_seen = time.time() - 0.5  # 500ms ago

            result = processor.check_idle_audio_gap()

            assert result == 12345
            assert processor.wake_word_user_id == 12345

    def test_flush_detection(self):
        """Wake word detected via flush captures user ID."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = False
            mock.flush.return_value = True  # Detect on flush
            MockDetector.return_value = mock

            # Create state with gap in valid range
            state = manager.get_or_create(user_id=12345)
            manager.ensure_detector(state)
            state.last_seen = time.time() - 0.5

            result = processor.check_idle_audio_gap()

            assert result == 12345
            assert processor.wake_word_user_id == 12345

    def test_max_silence_attempts_enforced(self):
        """Silence padding stops after 5 attempts.

        Critical edge case: refactor-edge-cases.md Section 2.3
        Prevents infinite loops.
        """
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = False
            MockDetector.return_value = mock

            # Create state with gap in valid range
            state = manager.get_or_create(user_id=123)
            manager.ensure_detector(state)
            state.last_seen = time.time() - 0.5

            # Call 5 times (max attempts)
            for i in range(5):
                processor.check_idle_audio_gap()

            call_count = mock.process.call_count
            assert call_count == 5

            # 6th call should not process
            processor.check_idle_audio_gap()

            assert mock.process.call_count == 5  # No additional calls

    def test_silence_attempts_increment(self):
        """Silence attempts increment on each call."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = False
            MockDetector.return_value = mock

            state = manager.get_or_create(user_id=123)
            manager.ensure_detector(state)
            state.last_seen = time.time() - 0.5

            assert state.silence_attempts == 0

            processor.check_idle_audio_gap()
            assert state.silence_attempts == 1

            processor.check_idle_audio_gap()
            assert state.silence_attempts == 2


class TestMultipleUsers:
    """Test handling of multiple users."""

    def test_checks_all_users_in_window(self):
        """Checks all users with gaps in valid window."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock = MagicMock()
            mock.process.return_value = False
            mock.flush.return_value = False
            MockDetector.return_value = mock

            # Create two users with gaps in valid range
            state1 = manager.get_or_create(user_id=111)
            manager.ensure_detector(state1)
            state1.last_seen = time.time() - 0.5

            state2 = manager.get_or_create(user_id=222)
            manager.ensure_detector(state2)
            state2.last_seen = time.time() - 1.0

            processor.check_idle_audio_gap()

            # Both should have been processed
            assert state1.silence_attempts == 1
            assert state2.silence_attempts == 1

    def test_first_detection_wins(self):
        """First user to detect wins (returns immediately)."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        detection_order = []

        def make_mock_that_detects_on_second():
            mock = MagicMock()
            call_count = [0]

            def process_side_effect(audio):
                call_count[0] += 1
                return call_count[0] == 2  # Detect on second call

            mock.process.side_effect = process_side_effect
            mock.flush.return_value = False
            return mock

        with patch("src.detection.WakeWordDetector") as MockDetector:
            mock1 = make_mock_that_detects_on_second()
            mock2 = make_mock_that_detects_on_second()

            # Return different mocks for each user
            MockDetector.side_effect = [mock1, mock2]

            state1 = manager.get_or_create(user_id=111)
            manager.ensure_detector(state1)
            state1.last_seen = time.time() - 0.5

            state2 = manager.get_or_create(user_id=222)
            manager.ensure_detector(state2)
            state2.last_seen = time.time() - 0.5

            # First call - no detection
            result1 = processor.check_idle_audio_gap()
            assert result1 is None

            # Second call - first user detects
            result2 = processor.check_idle_audio_gap()
            assert result2 is not None


class TestResetMethods:
    """Test state reset methods."""

    def test_reset_wake_user(self):
        """Reset clears wake word user ID."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)
        processor._wake_word_user_id = 12345

        processor.reset_wake_user()

        assert processor.wake_word_user_id is None

    def test_on_audio_gap(self):
        """Audio gap clears silence attempts for all users."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)

        state1 = manager.get_or_create(user_id=111)
        state2 = manager.get_or_create(user_id=222)
        state1.silence_attempts = 3
        state2.silence_attempts = 5

        processor.on_audio_gap()

        assert state1.silence_attempts == 0
        assert state2.silence_attempts == 0

    def test_on_idle_transition(self):
        """Idle transition resets all state."""
        manager = PerUserAudioManager()
        processor = WakeWordProcessor(manager)
        processor._wake_word_user_id = 12345

        state = manager.get_or_create(user_id=111)
        mock_detector = MagicMock()
        state.detector = mock_detector
        state.silence_attempts = 5

        processor.on_idle_transition()

        assert processor.wake_word_user_id is None
        assert state.silence_attempts == 0
        mock_detector.reset.assert_called_once()


class TestConstants:
    """Test timing constants are correct."""

    def test_gap_window(self):
        """Gap window is 250ms-2.5s."""
        assert MIN_GAP_SECONDS == 0.25
        assert MAX_GAP_SECONDS == 2.5

    def test_max_attempts(self):
        """Max silence attempts is 5."""
        assert MAX_SILENCE_ATTEMPTS == 5

    def test_silence_samples(self):
        """Silence chunk is 500ms at 16kHz = 8000 samples."""
        assert SILENCE_SAMPLES == 8000  # 0.5 * 16000
