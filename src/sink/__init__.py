"""Per-user audio management for voice sink.

This module extracts per-user audio state management from the monolithic
StreamingSink class into focused, testable components.

Modules:
- user_audio_state: Per-user state dataclass
- per_user_manager: Lifecycle management of user states
- wake_processor: Wake word detection with timing-sensitive logic
"""

from .user_audio_state import UserAudioState
from .per_user_manager import PerUserAudioManager
from .wake_processor import WakeWordProcessor

__all__ = [
    "UserAudioState",
    "PerUserAudioManager",
    "WakeWordProcessor",
]
