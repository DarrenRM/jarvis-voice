"""Speaker tracking for the voice session.

Tracks the current speaker during a voice interaction, including their
user ID, display name, and current Discord activity.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import discord
    from ..bot.speaker_manager import SpeakerManager

logger = logging.getLogger(__name__)


@dataclass
class SpeakerContext:
    """Current speaker context.

    Attributes:
        user_id: Discord user ID of the current speaker.
        name: Display name of the current speaker.
        activity: Current Discord activity (e.g., "Playing Minecraft").
    """

    user_id: Optional[int] = None
    name: Optional[str] = None
    activity: Optional[str] = None

    def clear(self) -> None:
        """Reset all fields to None."""
        self.user_id = None
        self.name = None
        self.activity = None


class SpeakerTracker:
    """Tracks the current speaker during voice interactions.

    Wraps SpeakerManager to maintain current speaker state and provide
    convenient access to speaker information.
    """

    def __init__(self, speaker_manager: "SpeakerManager"):
        """Initialize speaker tracker.

        Args:
            speaker_manager: SpeakerManager for resolving user info.
        """
        self._speaker_manager = speaker_manager
        self._context = SpeakerContext()

    @property
    def current_speaker_id(self) -> Optional[int]:
        """Get current speaker's user ID."""
        return self._context.user_id

    @property
    def current_speaker_name(self) -> Optional[str]:
        """Get current speaker's display name."""
        return self._context.name

    @property
    def current_speaker_activity(self) -> Optional[str]:
        """Get current speaker's Discord activity."""
        return self._context.activity

    @property
    def context(self) -> SpeakerContext:
        """Get the full speaker context."""
        return self._context

    def set_speaker(
        self,
        user_id: Optional[int],
        voice_client: Optional["discord.VoiceClient"],
    ) -> None:
        """Set current speaker by resolving their info.

        Args:
            user_id: Discord user ID of the speaker.
            voice_client: Voice client for guild access.
        """
        guild = voice_client.guild if voice_client else None
        info = self._speaker_manager.resolve_speaker(user_id, guild)

        self._context.user_id = info.user_id
        self._context.name = info.name
        self._context.activity = info.activity

        logger.debug(
            f"Speaker set: {self._context.name} (ID: {self._context.user_id}), "
            f"Activity: {self._context.activity}"
        )

    def clear(self) -> None:
        """Clear current speaker context."""
        self._context.clear()
        logger.debug("Speaker context cleared")

    def find_user_by_name(
        self,
        name: str,
        voice_client: Optional["discord.VoiceClient"],
    ) -> Optional[tuple[int, str]]:
        """Find a user by fuzzy name match.

        Args:
            name: Name to search for (case-insensitive, partial match).
            voice_client: Voice client for guild access.

        Returns:
            Tuple of (user_id, display_name) if found, None otherwise.
        """
        guild = voice_client.guild if voice_client else None
        return self._speaker_manager.find_user_by_name(name, guild)
