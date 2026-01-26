"""Discord-specific operations for posting images and messages."""

import io
import logging
from typing import Callable, List, Optional

import discord

logger = logging.getLogger(__name__)


class DiscordService:
    """Handles Discord API operations like posting images and tagging users."""

    MIME_TO_EXT = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }

    def __init__(
        self,
        channel_name: str,
        find_user_fn: Callable[[str], Optional[tuple[int, str]]],
        get_known_users_fn: Callable[[], List[str]],
    ):
        """Initialize Discord service.

        Args:
            channel_name: Default text channel name for posting.
            find_user_fn: Function to find user by name, returns (user_id, display_name).
            get_known_users_fn: Function to get list of known user names.
        """
        self._channel_name = channel_name
        self._find_user = find_user_fn
        self._get_known_users = get_known_users_fn

    async def post_image(
        self,
        voice_client: discord.VoiceClient,
        image_bytes: bytes,
        requester: str,
        mime_type: str = "image/png",
    ) -> str:
        """Post image to Discord channel.

        Args:
            voice_client: Discord voice client for guild access.
            image_bytes: The image data.
            requester: Name of the requester (for caption).
            mime_type: MIME type of the image.

        Returns:
            Status message for TTS feedback.
        """
        extension = self.MIME_TO_EXT.get(mime_type, "png")
        filename = f"generated.{extension}"

        logger.info(f"[IMAGE] post_image called with {len(image_bytes)} bytes, mime_type={mime_type}")

        if not voice_client or not voice_client.guild:
            logger.warning("[IMAGE] Cannot post image - no voice client or guild")
            return "Could not post image - not connected to a server"

        logger.info(f"[IMAGE] Looking for channel: #{self._channel_name}")

        # List available channels for debugging
        available_channels = [ch.name for ch in voice_client.guild.text_channels]
        logger.info(f"[IMAGE] Available text channels: {available_channels}")

        channel = discord.utils.get(
            voice_client.guild.text_channels,
            name=self._channel_name,
        )

        if not channel:
            logger.warning(f"[IMAGE] Could not find channel #{self._channel_name}")
            return f"Could not find channel #{self._channel_name}. Available: {available_channels}"

        # Check permissions
        perms = channel.permissions_for(channel.guild.me)
        logger.info(f"[IMAGE] Bot permissions: send_messages={perms.send_messages}, attach_files={perms.attach_files}")
        if not perms.attach_files:
            return f"Bot lacks 'Attach Files' permission in #{self._channel_name}"

        try:
            logger.info(f"[IMAGE] Sending to #{self._channel_name}...")
            await channel.send(
                content=f"*{requester}'s requested image:*",
                file=discord.File(io.BytesIO(image_bytes), filename=filename),
            )
            logger.info(f"[IMAGE] Posted generated image to #{self._channel_name}")
            return f"Done! I've posted the image to #{self._channel_name}."
        except discord.errors.DiscordServerError as e:
            logger.warning(f"[IMAGE] Discord server error: {e}")
            return "Discord is blocking image uploads at the moment."
        except Exception as e:
            logger.error(f"[IMAGE] Failed to post image: {e}", exc_info=True)
            return f"Failed to post image: {str(e)}"

    async def tag_user(
        self,
        voice_client: discord.VoiceClient,
        target_name: str,
        message: str,
        requester: str,
    ) -> str:
        """Tag a user in Discord.

        Args:
            voice_client: Discord voice client for guild access.
            target_name: Name of user to tag (fuzzy matched).
            message: Message content.
            requester: Name of the requester.

        Returns:
            Status message for TTS feedback.
        """
        logger.info(f"[TAG] tag_user called: target={target_name}, message={message[:50]}...")

        if not voice_client or not voice_client.guild:
            logger.warning("[TAG] Cannot tag user - no voice client or guild")
            return "Could not tag user - not connected to a server"

        result = self._find_user(target_name)
        if not result:
            known = self._get_known_users()
            logger.warning(f"[TAG] Could not find user '{target_name}'. Known: {known}")
            return f"Could not find user '{target_name}'. Known users: {', '.join(known)}"

        target_id, target_display_name = result
        logger.info(f"[TAG] Found user: {target_display_name} (ID: {target_id})")

        logger.info(f"[TAG] Looking for channel: #{self._channel_name}")

        channel = discord.utils.get(
            voice_client.guild.text_channels,
            name=self._channel_name,
        )

        if not channel:
            available = [ch.name for ch in voice_client.guild.text_channels]
            logger.warning(f"[TAG] Could not find channel #{self._channel_name}. Available: {available}")
            return f"Could not find channel #{self._channel_name}"

        # Use <@user_id> format for mention - works even if user is offline/uncached
        formatted = f"<@{target_id}> - {requester} says: {message}"

        try:
            await channel.send(formatted)
            logger.info(f"[TAG] Posted tag message to #{self._channel_name}")
            return f"Done! I've tagged {target_display_name} in {self._channel_name} chat."
        except Exception as e:
            logger.error(f"[TAG] Failed to post tag: {e}", exc_info=True)
            return f"Failed to send message: {str(e)}"

    async def post_message(
        self,
        voice_client: discord.VoiceClient,
        message: str,
        requester: str,
    ) -> str:
        """Post a message to Discord channel.

        Args:
            voice_client: Discord voice client for guild access.
            message: Message content to post.
            requester: Name of the requester.

        Returns:
            Status message for TTS feedback.
        """
        logger.info(f"[POST] post_message called: {message[:50]}...")

        if not voice_client or not voice_client.guild:
            logger.warning("[POST] Cannot post message - no voice client or guild")
            return "Could not post message - not connected to a server"

        channel = discord.utils.get(
            voice_client.guild.text_channels,
            name=self._channel_name,
        )

        if not channel:
            available = [ch.name for ch in voice_client.guild.text_channels]
            logger.warning(f"[POST] Could not find channel #{self._channel_name}. Available: {available}")
            return f"Could not find channel #{self._channel_name}"

        try:
            formatted = f"*{requester} asked me to share:*\n{message}"
            await channel.send(formatted)
            logger.info(f"[POST] Posted message to #{self._channel_name}")
            return f"Done! I've posted that to {self._channel_name} chat."
        except Exception as e:
            logger.error(f"[POST] Failed to post message: {e}", exc_info=True)
            return f"Failed to post message: {str(e)}"
