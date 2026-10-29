"""Tests for DiscordService."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import discord

from src.bot.discord_service import DiscordService


class TestDiscordService:
    """Tests for DiscordService class."""

    @pytest.fixture
    def find_user_fn(self):
        """Mock find_user function."""
        def finder(name):
            users = {
                "alice": (123, "Alice"),
                "bob": (456, "Bob"),
            }
            return users.get(name.lower())
        return finder

    @pytest.fixture
    def get_known_users_fn(self):
        """Mock get_known_users function."""
        return lambda: ["Alice", "Bob", "Casey"]

    @pytest.fixture
    def service(self, find_user_fn, get_known_users_fn):
        """DiscordService instance."""
        return DiscordService(
            channel_name="bot-chat",
            find_user_fn=find_user_fn,
            get_known_users_fn=get_known_users_fn,
        )

    @pytest.fixture
    def mock_voice_client(self):
        """Mock Discord voice client with guild."""
        vc = Mock()
        vc.guild = Mock()
        vc.guild.me = Mock()

        # Create mock channel
        channel = AsyncMock()
        channel.name = "bot-chat"
        channel.permissions_for = Mock(return_value=Mock(
            send_messages=True,
            attach_files=True,
        ))

        vc.guild.text_channels = [channel]

        return vc, channel

    # post_image tests

    @pytest.mark.asyncio
    async def test_post_image_no_voice_client(self, service):
        """post_image returns error when no voice client."""
        result = await service.post_image(None, b"data", "User")
        assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_post_image_no_guild(self, service):
        """post_image returns error when voice client has no guild."""
        vc = Mock()
        vc.guild = None

        result = await service.post_image(vc, b"data", "User")
        assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_post_image_channel_not_found(self, service):
        """post_image returns error when channel not found."""
        vc = Mock()
        vc.guild = Mock()
        vc.guild.text_channels = []  # No channels

        with patch.object(discord.utils, 'get', return_value=None):
            result = await service.post_image(vc, b"data", "User")

        assert "could not find channel" in result.lower()

    @pytest.mark.asyncio
    async def test_post_image_no_permission(self, service, mock_voice_client):
        """post_image returns error when lacking permissions."""
        vc, channel = mock_voice_client
        channel.permissions_for = Mock(return_value=Mock(
            send_messages=True,
            attach_files=False,  # No permission
        ))

        with patch.object(discord.utils, 'get', return_value=channel):
            result = await service.post_image(vc, b"data", "User")

        assert "permission" in result.lower()

    @pytest.mark.asyncio
    async def test_post_image_success(self, service, mock_voice_client):
        """post_image posts to channel successfully."""
        vc, channel = mock_voice_client

        with patch.object(discord.utils, 'get', return_value=channel):
            result = await service.post_image(
                vc,
                b"fake image data",
                "TestUser",
                mime_type="image/png"
            )

        assert "done" in result.lower()
        assert "bot-chat" in result.lower()
        channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_image_discord_error(self, service, mock_voice_client):
        """post_image handles Discord server errors."""
        vc, channel = mock_voice_client
        channel.send = AsyncMock(side_effect=discord.errors.DiscordServerError(Mock(), "error"))

        with patch.object(discord.utils, 'get', return_value=channel):
            result = await service.post_image(vc, b"data", "User")

        assert "blocking" in result.lower()

    @pytest.mark.asyncio
    async def test_post_image_generic_error(self, service, mock_voice_client):
        """post_image handles generic exceptions."""
        vc, channel = mock_voice_client
        channel.send = AsyncMock(side_effect=Exception("Network error"))

        with patch.object(discord.utils, 'get', return_value=channel):
            result = await service.post_image(vc, b"data", "User")

        assert "failed" in result.lower()

    # MIME type handling

    def test_mime_to_ext_mapping(self, service):
        """MIME types map to correct extensions."""
        assert service.MIME_TO_EXT["image/png"] == "png"
        assert service.MIME_TO_EXT["image/jpeg"] == "jpg"
        assert service.MIME_TO_EXT["image/webp"] == "webp"
        assert service.MIME_TO_EXT["image/gif"] == "gif"

    # tag_user tests

    @pytest.mark.asyncio
    async def test_tag_user_no_voice_client(self, service):
        """tag_user returns error when no voice client."""
        result = await service.tag_user(None, "Alice", "Hello!", "Requester")
        assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_tag_user_not_found(self, service):
        """tag_user returns error when user not found."""
        vc = Mock()
        vc.guild = Mock()

        result = await service.tag_user(vc, "Unknown", "Hello!", "Requester")
        assert "could not find user" in result.lower()
        assert "Unknown" in result

    @pytest.mark.asyncio
    async def test_tag_user_shows_known_users(self, service):
        """tag_user shows known users when target not found."""
        vc = Mock()
        vc.guild = Mock()

        result = await service.tag_user(vc, "Unknown", "Hello!", "Requester")
        assert "Alice" in result or "known" in result.lower()

    @pytest.mark.asyncio
    async def test_tag_user_channel_not_found(self, service, find_user_fn):
        """tag_user returns error when channel not found."""
        vc = Mock()
        vc.guild = Mock()
        vc.guild.text_channels = []

        with patch.object(discord.utils, 'get', return_value=None):
            result = await service.tag_user(vc, "Alice", "Hello!", "Requester")

        assert "could not find channel" in result.lower()

    @pytest.mark.asyncio
    async def test_tag_user_success(self, service, mock_voice_client):
        """tag_user sends message successfully."""
        vc, channel = mock_voice_client

        with patch.object(discord.utils, 'get', return_value=channel):
            result = await service.tag_user(vc, "Alice", "Hello!", "Requester")

        assert "done" in result.lower()
        assert "Alice" in result
        channel.send.assert_called_once()

        # Check the message format includes mention
        call_args = channel.send.call_args
        message = call_args[0][0] if call_args[0] else call_args[1].get('content', '')
        assert "<@123>" in message  # User ID mention
        assert "Requester" in message

    @pytest.mark.asyncio
    async def test_tag_user_send_error(self, service, mock_voice_client):
        """tag_user handles send errors."""
        vc, channel = mock_voice_client
        channel.send = AsyncMock(side_effect=Exception("Send failed"))

        with patch.object(discord.utils, 'get', return_value=channel):
            result = await service.tag_user(vc, "Alice", "Hello!", "Requester")

        assert "failed" in result.lower()
