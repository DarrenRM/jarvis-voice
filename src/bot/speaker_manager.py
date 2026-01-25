"""Speaker identification and personalization."""

from dataclasses import dataclass
from typing import Dict, Optional

import discord

logger = __import__("logging").getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Information about a speaker."""

    user_id: Optional[int]
    name: Optional[str]
    activity: Optional[str]


class SpeakerManager:
    """Manages speaker identification and name resolution."""

    def __init__(self, user_names: Dict[int, str]):
        """Initialize speaker manager.

        Args:
            user_names: Mapping of Discord user ID to friendly name.
        """
        self._user_names = user_names

    def resolve_speaker(
        self,
        user_id: Optional[int],
        guild: Optional[discord.Guild],
    ) -> SpeakerInfo:
        """Resolve speaker info from user ID.

        Args:
            user_id: Discord user ID.
            guild: Discord guild for member lookup.

        Returns:
            SpeakerInfo with user_id, name, and activity.
        """
        if not user_id:
            return SpeakerInfo(None, None, None)

        name = None
        activity = None

        # Check configured names first
        if user_id in self._user_names:
            name = self._user_names[user_id]

        # Try Discord lookup
        if guild:
            try:
                member = guild.get_member(user_id)
                if member:
                    if not name:
                        name = member.display_name
                    activity = self._get_activity(member)
            except Exception:
                pass

        return SpeakerInfo(user_id, name, activity)

    def find_user_by_name(
        self,
        name: str,
        guild: Optional[discord.Guild],
    ) -> Optional[tuple[int, str]]:
        """Fuzzy match user by name.

        Args:
            name: Name to search for (case-insensitive, partial match).
            guild: Discord guild for member lookup.

        Returns:
            Tuple of (user_id, display_name) if found, None otherwise.
        """
        if not name:
            return None

        name_lower = name.lower().strip()

        # Exact match in configured names
        for user_id, friendly_name in self._user_names.items():
            if friendly_name.lower() == name_lower:
                return (user_id, friendly_name)

        # Partial match
        for user_id, friendly_name in self._user_names.items():
            if name_lower in friendly_name.lower():
                return (user_id, friendly_name)

        # Fallback: search guild members
        if guild:
            for member in guild.members:
                if name_lower in member.display_name.lower():
                    return (member.id, member.display_name)

        return None

    def _get_activity(self, member: discord.Member) -> Optional[str]:
        """Get member's current activity as string.

        Args:
            member: Discord member.

        Returns:
            Activity string or None.
        """
        if not member or not member.activities:
            return None
        for activity in member.activities:
            if isinstance(activity, discord.Game):
                return f"Playing {activity.name}"
            elif isinstance(activity, discord.Activity):
                return f"{activity.type.name.title()} {activity.name}"
        return None
