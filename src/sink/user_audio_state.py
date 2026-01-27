"""Per-user audio processing state.

Consolidates 4 scattered dictionaries from voice_sink.py into a single dataclass:
- _per_user_detectors
- _per_user_pre_buffers
- _per_user_last_seen
- _per_user_silence_attempts
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..detection import WakeWordDetector


# Audio processing constants
SAMPLE_RATE_PROCESSING = 16000


@dataclass
class UserAudioState:
    """Per-user audio processing state.

    Encapsulates all state for a single user's audio processing:
    - Wake word detector (lazy initialized)
    - Pre-buffer for capturing question start before wake word
    - Last seen timestamp for gap detection
    - Silence padding attempt counter

    Attributes:
        user_id: Discord user ID
        detector: Wake word detector instance (lazy initialized by manager)
        pre_buffer: Audio chunks captured before wake word
        last_seen: Timestamp of last audio received from this user
        silence_attempts: Counter for silence padding attempts
    """

    user_id: int
    detector: Optional["WakeWordDetector"] = None
    pre_buffer: list[np.ndarray] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    silence_attempts: int = 0

    # Pre-buffer limits (1.5 seconds at 16kHz = 24000 samples)
    MAX_PRE_BUFFER_SECONDS: float = field(default=1.5, repr=False)
    SAMPLE_RATE: int = field(default=SAMPLE_RATE_PROCESSING, repr=False)

    @property
    def max_pre_buffer_samples(self) -> int:
        """Maximum samples to keep in pre-buffer."""
        return int(self.MAX_PRE_BUFFER_SECONDS * self.SAMPLE_RATE)

    def add_to_pre_buffer(self, audio: np.ndarray) -> None:
        """Add audio to pre-buffer with FIFO eviction.

        Maintains a rolling buffer of recent audio to capture question
        starts when user speaks continuously (e.g., "hey jarvis what's the weather").

        Args:
            audio: Audio chunk (16kHz mono float32)
        """
        self.pre_buffer.append(audio.copy())
        self._trim_pre_buffer()

    def _trim_pre_buffer(self) -> None:
        """Trim pre-buffer to max size using FIFO eviction.

        Ported from voice_sink.py:513-518.
        """
        total_samples = sum(len(chunk) for chunk in self.pre_buffer)
        max_samples = self.max_pre_buffer_samples

        while total_samples > max_samples and self.pre_buffer:
            removed = self.pre_buffer.pop(0)
            total_samples -= len(removed)

    def get_pre_buffer(self) -> np.ndarray:
        """Get concatenated pre-buffer audio.

        Returns:
            Concatenated audio as float32 numpy array, or empty array if no buffer.
        """
        if not self.pre_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.pre_buffer)

    def get_pre_buffer_duration(self) -> float:
        """Get pre-buffer duration in seconds.

        Returns:
            Duration in seconds.
        """
        total_samples = sum(len(chunk) for chunk in self.pre_buffer)
        return total_samples / self.SAMPLE_RATE

    def clear_pre_buffer(self) -> None:
        """Clear the pre-buffer."""
        self.pre_buffer.clear()

    def update_last_seen(self) -> None:
        """Update last seen timestamp to current time."""
        self.last_seen = time.time()

    def time_since_seen(self) -> float:
        """Get seconds since last audio received.

        Returns:
            Seconds since last audio packet from this user.
        """
        return time.time() - self.last_seen

    def increment_silence_attempts(self) -> int:
        """Increment and return silence attempt counter.

        Used to track silence padding attempts for wake word flush.
        Limited to 5 attempts per gap (see refactor-edge-cases.md Section 2.3).

        Returns:
            New attempt count after increment.
        """
        self.silence_attempts += 1
        return self.silence_attempts

    def reset_silence_attempts(self) -> None:
        """Reset silence attempt counter.

        Called when new audio arrives or on idle transition.
        """
        self.silence_attempts = 0

    def reset_detector(self) -> None:
        """Reset the wake word detector state.

        Called on idle transition to clear any pending detection state.
        """
        if self.detector:
            self.detector.reset()

    def cleanup(self) -> None:
        """Clean up all resources.

        Resets detector, clears buffer, and resets counters.
        Called when user leaves voice channel.
        """
        if self.detector:
            self.detector.reset()
            self.detector = None
        self.pre_buffer.clear()
        self.silence_attempts = 0
