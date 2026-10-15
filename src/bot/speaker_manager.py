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
        name_normalized = name_lower.replace(" ", "")

        # Exact match in configured names (with space normalization)
        for user_id, friendly_name in self._user_names.items():
            friendly_lower = friendly_name.lower()
            friendly_normalized = friendly_lower.replace(" ", "")
            # Exact match (with or without spaces)
            if friendly_lower == name_lower or friendly_normalized == name_normalized:
                return (user_id, friendly_name)

        # Bidirectional partial match (search in name OR name in search)
        for user_id, friendly_name in self._user_names.items():
            friendly_lower = friendly_name.lower()
            friendly_normalized = friendly_lower.replace(" ", "")
            # Check both directions, with and without spaces
            if (name_lower in friendly_lower or friendly_lower in name_lower or
                name_normalized in friendly_normalized or friendly_normalized in name_normalized):
                return (user_id, friendly_name)

        # Fallback: search guild members (with space normalization)
        if guild:
            for member in guild.members:
                display_lower = member.display_name.lower()
                display_normalized = display_lower.replace(" ", "")
                # Check both directions, with and without spaces
                if (name_lower in display_lower or display_lower in name_lower or
                    name_normalized in display_normalized or display_normalized in name_normalized):
                    return (member.id, member.display_name)

        return None

    def channel_activities(self, channel, exclude_id: Optional[int] = None) -> list[str]:
        """Activities of the people in a voice channel (bots and exclude_id left out)."""
        out = []
        for member in getattr(channel, "members", None) or []:
            if getattr(member, "bot", False) or getattr(member, "id", None) == exclude_id:
                continue
            activity = self._get_activity(member)
            if activity:
                out.append(activity)
        return out

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
