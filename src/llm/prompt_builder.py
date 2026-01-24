"""Context-aware prompt building."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import discord

logger = logging.getLogger(__name__)


@dataclass
class DiscordContext:
    """Discord context for prompt enrichment."""

    guild_name: Optional[str] = None
    channel_name: Optional[str] = None
    user_name: Optional[str] = None
    user_activity: Optional[str] = None  # Current game/activity
    voice_channel_members: list[str] = field(default_factory=list)


class PromptBuilder:
    """Builds context-aware prompts for Claude.

    Enriches prompts with:
    - Current Discord channel info
    - User's current activity (game, etc.)
    - Voice channel members
    """

    def __init__(self):
        self._context: Optional[DiscordContext] = None

    def update_context(
        self,
        guild: Optional[discord.Guild] = None,
        channel: Optional[discord.VoiceChannel] = None,
        user: Optional[discord.Member] = None,
    ) -> None:
        """Update context from Discord objects.

        Args:
            guild: Discord guild
            channel: Voice channel
            user: Speaking user
        """
        self._context = DiscordContext()

        if guild:
            self._context.guild_name = guild.name

        if channel:
            self._context.channel_name = channel.name
            self._context.voice_channel_members = [
                m.display_name for m in channel.members if not m.bot
            ]

        if user:
            self._context.user_name = user.display_name

            # Get user's activity (game, etc.)
            if user.activities:
                for activity in user.activities:
                    if isinstance(activity, discord.Game):
                        self._context.user_activity = f"Playing {activity.name}"
                        break
                    elif isinstance(activity, discord.Activity):
                        self._context.user_activity = f"{activity.type.name}: {activity.name}"
                        break

    def build_context_string(self) -> str:
        """Build context string for Claude.

        Returns:
            Context string to include in system prompt
        """
        if not self._context:
            return ""

        parts = []

        if self._context.user_name:
            parts.append(f"Speaking with: {self._context.user_name}")

        if self._context.user_activity:
            parts.append(f"User activity: {self._context.user_activity}")

        if self._context.channel_name:
            parts.append(f"Voice channel: {self._context.channel_name}")

        if self._context.voice_channel_members:
            others = [
                m for m in self._context.voice_channel_members
                if m != self._context.user_name
            ]
            if others:
                parts.append(f"Others in channel: {', '.join(others)}")

        if self._context.guild_name:
            parts.append(f"Server: {self._context.guild_name}")

        return "\n".join(parts)

    def build_enriched_message(self, transcribed_text: str) -> str:
        """Build enriched message with context hints.

        Args:
            transcribed_text: Raw transcribed user speech

        Returns:
            Message with context (if relevant)
        """
        # For now, just return the transcribed text
        # Context is provided via system prompt
        return transcribed_text

    @property
    def has_context(self) -> bool:
        return self._context is not None
