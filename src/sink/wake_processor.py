"""Wake word processor with timing-sensitive detection logic.

Handles wake word detection including:
- Real-time detection from audio stream
- Silence padding for PTT (push-to-talk) detection
- Wake word user ID capture at detection time

Critical timing constraints (docs/archive/refactor-edge-cases.md, Section 2.3):
- Gap window: 250ms-2.5s (too short = false positives, too long = stale)
- Max 5 silence attempts per user per gap
- Capture wake_word_user_id at detection time (NOT later)
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .per_user_manager import PerUserAudioManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Silence chunk for wake word flush (500ms at 16kHz = 8000 samples)
SILENCE_SAMPLES = 8000
SAMPLE_RATE = 16000

# Gap window for silence padding
MIN_GAP_SECONDS = 0.25  # 250ms minimum gap
MAX_GAP_SECONDS = 2.5   # 2.5s maximum gap

# Max silence attempts per user per gap
MAX_SILENCE_ATTEMPTS = 5
NEAR_MISS_SCORE = 0.05  # log speech bursts whose best wake score reached this


class WakeWordProcessor:
    """Processes audio for wake word detection.

    Handles:
    - Real-time wake word detection from audio stream
    - Silence padding to flush pending detections on audio gaps
    - Wake word user ID capture at detection time

    Critical edge cases preserved:
    - Wake word user captured at detection time (not start_listening time)
    - Gap window 250ms-2.5s for silence padding
    - Max 5 silence attempts per user per gap

    Example:
        processor = WakeWordProcessor(user_manager)

        # Process incoming audio
        if processor.process_audio(user_id, audio_16k):
            # Wake word detected!
            speaker_id = processor.wake_word_user_id

        # Poll for pending detections on audio gaps
        detected_user = processor.check_idle_audio_gap()
        if detected_user is not None:
            # Wake word completed via silence padding
    """

    def __init__(self, user_manager: PerUserAudioManager):
        """Initialize wake word processor.

        Args:
            user_manager: Per-user audio state manager
        """
        self._user_manager = user_manager

        # Wake word user ID captured at detection time
        # Critical: This prevents race conditions where another user's
        # packet arrives between detection and start_listening()
        self._wake_word_user_id: Optional[int] = None

        # (user_id, recent audio, score, hit) for hits and near misses: the monitor
        # keeps these clips when MONITOR_SAVE_AUDIO is on. Runs on the audio thread.
        self.on_clip: Optional[Callable[[int, np.ndarray, float, bool], None]] = None

    def report_clip(self, state, hit: bool, score: Optional[float] = None) -> None:
        """Hand the user's last ~1.5s (their pre-buffer) to on_clip."""
        if self.on_clip is None:
            return
        if score is None:
            score = getattr(state.detector, "last_score", 0.0)
        try:
            self.on_clip(state.user_id, state.get_pre_buffer(), float(score or 0.0), hit)
        except Exception as e:
            logger.debug(f"[WAKE_PROCESSOR] on_clip failed: {e}")

    @property
    def wake_word_user_id(self) -> Optional[int]:
        """Get the user ID who triggered the wake word.

        This is captured at detection time, not at start_listening time,
        to prevent race conditions with audio packets from other users.
        """
        return self._wake_word_user_id

    def process_audio(self, user_id: int, audio_16k: np.ndarray) -> bool:
        """Process audio for wake word detection.

        Called for each audio packet in idle state.
        If wake word is detected, captures user_id for speaker isolation.

        Args:
            user_id: Discord user ID
            audio_16k: Audio data (16kHz mono float32)

        Returns:
            True if wake word detected, False otherwise
        """
        state = self._user_manager.get_or_create(user_id)
        detector = self._user_manager.ensure_detector(state)

        if detector is None:
            # Max detectors reached
            return False

        # Reset silence attempts on new audio
        state.reset_silence_attempts()

        # Process through detector
        if detector.process(audio_16k):
            # Wake word detected! Capture user ID at detection time.
            # This is critical for speaker isolation (docs/archive/refactor-edge-cases.md, Section 3.5)
            self._wake_word_user_id = user_id
            logger.info(f"[WAKE_PROCESSOR] Wake word detected from user {user_id}")
            self.report_clip(state, hit=True)
            return True

        return False

    def check_idle_audio_gap(self) -> Optional[int]:
        """Check for pending wake word detection on audio gaps.

        Called periodically when no audio is arriving. Feeds silence to
        wake word detectors to complete pending detections.

        OpenWakeWord is a streaming model - it may have the wake word
        "loaded" in its internal state but needs more audio to complete
        the detection.

        Critical timing constraints:
        - Only check users with 250ms-2.5s gap (recent enough to be relevant)
        - Max 5 attempts per user per gap (prevents infinite loops)
        - Feed 500ms of silence (8000 samples at 16kHz)

        Returns:
            User ID if wake word detected, None otherwise
        """
        silence = np.zeros(SILENCE_SAMPLES, dtype=np.float32)

        for state in self._user_manager:
            user_gap = state.time_since_seen()

            # Only check users with recent audio (250ms - 2.5s gap)
            if not (MIN_GAP_SECONDS <= user_gap <= MAX_GAP_SECONDS):
                continue

            detector = state.detector
            if detector is None:
                continue

            # Limit attempts per user (prevents infinite loops)
            attempts = state.silence_attempts
            if attempts >= MAX_SILENCE_ATTEMPTS:
                continue

            # Increment attempt counter
            state.increment_silence_attempts()
            logger.debug(
                f"[WAKE_PROCESSOR] Feeding silence to user {state.user_id} "
                f"attempt {attempts + 1}/{MAX_SILENCE_ATTEMPTS} (gap={user_gap:.2f}s)"
            )

            # Feed silence to complete any pending detection
            detected = detector.process(silence)
            if attempts == 0 and not detected and hasattr(detector, "take_peak"):
                # First look at this gap: report how close a missed wake word came
                peak = detector.take_peak()
                if isinstance(peak, float) and peak >= NEAR_MISS_SCORE:
                    logger.info(
                        f"[WAKE] Near miss from user {state.user_id}: best score {peak:.2f} "
                        f"(needs {getattr(detector, 'threshold', 0):.2f})"
                    )
                    self.report_clip(state, hit=False, score=peak)
            if detected:
                # User attribution is 100% accurate - we know which detector fired
                self._wake_word_user_id = state.user_id
                logger.info(
                    f"[WAKE_PROCESSOR] Wake word detected from silence padding "
                    f"(gap={user_gap:.2f}s, user: {state.user_id})"
                )
                self.report_clip(state, hit=True)
                return state.user_id

            # Also try flush in case there's buffered audio
            if hasattr(detector, 'flush') and detector.flush():
                self._wake_word_user_id = state.user_id
                logger.info(
                    f"[WAKE_PROCESSOR] Wake word detected from flush "
                    f"(gap={user_gap:.2f}s, user: {state.user_id})"
                )
                self.report_clip(state, hit=True)
                return state.user_id

        return None

    def reset_wake_user(self) -> None:
        """Clear the wake word user ID.

        Called on idle transition to clear stale state.
        """
        self._wake_word_user_id = None

    def on_audio_gap(self) -> None:
        """Handle audio gap (silence detected).

        Called when a significant gap in audio is detected.
        Clears silence attempt counters since this is a new gap period.
        """
        self._user_manager.clear_all_silence_attempts()

    def on_idle_transition(self) -> None:
        """Handle transition to idle state.

        Resets all detectors and clears state.
        """
        self._user_manager.reset_all_detectors()
        self._user_manager.clear_all_silence_attempts()
        self._wake_word_user_id = None
