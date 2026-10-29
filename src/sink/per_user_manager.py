"""Per-user audio state manager.

Manages the lifecycle of UserAudioState instances, including:
- Creating/retrieving user states
- Lazy wake word detector initialization
- User cleanup on disconnect
- Batch operations on all users
"""

import logging
import time
from typing import TYPE_CHECKING, Iterator, Optional

from .user_audio_state import UserAudioState

if TYPE_CHECKING:
    from ..detection import WakeWordDetector

logger = logging.getLogger(__name__)

# After a wake word model fails to load, wait this long before trying again (seconds)
DETECTOR_RETRY_SECONDS = 30.0


class PerUserAudioManager:
    """Manages per-user audio state lifecycle.

    Responsibilities:
    - Create/retrieve user states on demand
    - Lazy wake word detector initialization (max 10 detectors)
    - User cleanup on disconnect
    - Batch reset operations

    Example:
        manager = PerUserAudioManager(
            wake_model_name="hey_jarvis",
            wake_threshold=0.5,
            wake_sample_rate=16000,
        )

        # Get or create state for user
        state = manager.get_or_create(user_id=12345)

        # Ensure detector exists (lazy init)
        detector = manager.ensure_detector(state)

        # Cleanup when user leaves
        manager.cleanup_user(12345)
    """

    def __init__(
        self,
        wake_model_name: str = "hey_jarvis",
        wake_threshold: float = 0.5,
        wake_sample_rate: int = 16000,
        max_detectors: int = 10,
    ):
        """Initialize the per-user manager.

        Args:
            wake_model_name: Wake word model name (e.g., "hey_jarvis")
            wake_threshold: Detection confidence threshold
            wake_sample_rate: Audio sample rate for wake detection
            max_detectors: Maximum number of wake detectors to create
        """
        self._users: dict[int, UserAudioState] = {}
        self._wake_model_name = wake_model_name
        self._wake_threshold = wake_threshold
        self._wake_sample_rate = wake_sample_rate
        self._max_detectors = max_detectors
        self._detector_retry_at = 0.0  # time.monotonic() before which loading is skipped

    def get_or_create(self, user_id: int) -> UserAudioState:
        """Get existing state or create new one for user.

        Args:
            user_id: Discord user ID

        Returns:
            UserAudioState instance for this user
        """
        if user_id not in self._users:
            self._users[user_id] = UserAudioState(user_id=user_id)
            logger.debug(f"[USER_MANAGER] Created state for user {user_id}")
        return self._users[user_id]

    def get(self, user_id: int) -> Optional[UserAudioState]:
        """Get state if exists, None otherwise.

        Args:
            user_id: Discord user ID

        Returns:
            UserAudioState if exists, None otherwise
        """
        return self._users.get(user_id)

    def ensure_detector(self, state: UserAudioState) -> Optional["WakeWordDetector"]:
        """Ensure user has a wake word detector (lazy initialization).

        Each user gets their own wake word detector to prevent cross-talk
        interference and ensure accurate user attribution.

        Args:
            state: User's audio state

        Returns:
            WakeWordDetector instance, or None if max detectors reached or the model
            could not be loaded
        """
        if state.detector is not None:
            return state.detector
        if time.monotonic() < self._detector_retry_at:
            return None

        # Check detector limit (snapshot to avoid race condition)
        detector_count = sum(1 for s in list(self._users.values()) if s.detector is not None)
        if detector_count >= self._max_detectors:
            logger.warning(
                f"[USER_MANAGER] Max detectors ({self._max_detectors}) reached, "
                f"skipping user {state.user_id}"
            )
            return None

        # Lazy import to avoid circular dependency
        from ..detection import WakeWordDetector

        logger.debug(f"[USER_MANAGER] Creating detector for user {state.user_id}")
        try:
            state.detector = WakeWordDetector(
                model_name=self._wake_model_name,
                threshold=self._wake_threshold,
                sample_rate=self._wake_sample_rate,
            )
        except Exception:
            # This runs on py-cord's audio thread at a new speaker's first packet: a failed
            # model download or ONNX load must not stop recording for everyone else
            self._detector_retry_at = time.monotonic() + DETECTOR_RETRY_SECONDS
            logger.exception(
                f"[USER_MANAGER] Wake word model failed to load for user {state.user_id}; "
                f"trying again in {DETECTOR_RETRY_SECONDS:g}s"
            )
            return None
        return state.detector

    def cleanup_user(self, user_id: int) -> None:
        """Clean up user state on disconnect.

        Releases all resources associated with the user.

        Args:
            user_id: Discord user ID
        """
        state = self._users.pop(user_id, None)
        if state:
            state.cleanup()
            logger.debug(f"[USER_MANAGER] Cleaned up user {user_id}")

    def cleanup_all(self) -> None:
        """Clean up all users.

        Called on session stop or when sink goes to idle state
        and we want to release all resources.
        """
        # Snapshot to avoid race condition during iteration
        for state in list(self._users.values()):
            state.cleanup()
        self._users.clear()
        logger.debug("[USER_MANAGER] Cleaned up all users")

    def reset_all_detectors(self) -> None:
        """Reset all wake word detectors.

        Called on idle transition to clear pending detection state.
        Does not destroy detectors, just resets their internal state.
        """
        # Snapshot to avoid race condition during iteration
        for state in list(self._users.values()):
            state.reset_detector()

    def clear_all_silence_attempts(self) -> None:
        """Clear silence attempt counters for all users.

        Called on idle transition.
        """
        # Snapshot to avoid race condition during iteration
        for state in list(self._users.values()):
            state.reset_silence_attempts()

    def clear_all_pre_buffers(self) -> None:
        """Clear pre-buffers for all users.

        Called after wake word transfer to prevent stale audio.
        """
        # Snapshot to avoid race condition during iteration
        for state in list(self._users.values()):
            state.clear_pre_buffer()

    def __iter__(self) -> Iterator[UserAudioState]:
        """Iterate over all user states.

        Yields:
            UserAudioState for each tracked user

        Note: Returns snapshot to avoid race conditions.
        """
        return iter(list(self._users.values()))

    def __len__(self) -> int:
        """Number of tracked users.

        Returns:
            Count of users with state
        """
        return len(self._users)

    @property
    def user_ids(self) -> set[int]:
        """Set of tracked user IDs.

        Returns:
            Set of user IDs currently being tracked
        """
        return set(self._users.keys())

    @property
    def detector_count(self) -> int:
        """Number of active detectors.

        Returns:
            Count of users with active detectors
        """
        # Snapshot to avoid race condition during iteration
        return sum(1 for s in list(self._users.values()) if s.detector is not None)
