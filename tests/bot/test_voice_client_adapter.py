"""Tests for voice client adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.voice_client_adapter import (
    DiscordVoiceClientAdapter,
    MockVoiceClientAdapter,
    VoiceClientAdapter,
)


class TestVoiceClientAdapterProtocol:
    """Tests for VoiceClientAdapter protocol."""

    def test_mock_adapter_is_protocol_compatible(self):
        """MockVoiceClientAdapter should satisfy the protocol."""
        adapter = MockVoiceClientAdapter()
        assert isinstance(adapter, VoiceClientAdapter)


class TestDiscordVoiceClientAdapter:
    """Tests for DiscordVoiceClientAdapter."""

    @pytest.fixture
    def mock_voice_client(self):
        """Create a mock discord.VoiceClient."""
        client = MagicMock()
        client.play = MagicMock()
        client.stop = MagicMock()
        client.is_playing = MagicMock(return_value=False)
        client.start_recording = MagicMock()
        client.stop_recording = MagicMock()
        client.recording = False
        client.is_connected = MagicMock(return_value=True)
        client.ws = MagicMock()
        client.ws.speak = AsyncMock()
        client.guild = MagicMock()
        client.user = MagicMock()
        return client

    @pytest.fixture
    def adapter(self, mock_voice_client):
        """Create adapter with mock client."""
        return DiscordVoiceClientAdapter(mock_voice_client)

    def test_play_delegates_to_client(self, adapter, mock_voice_client):
        """play() should delegate to underlying client."""
        source = MagicMock()
        callback = MagicMock()

        adapter.play(source, after=callback)

        mock_voice_client.play.assert_called_once_with(source, after=callback)

    def test_stop_delegates_to_client(self, adapter, mock_voice_client):
        """stop() should delegate to underlying client."""
        adapter.stop()
        mock_voice_client.stop.assert_called_once()

    def test_is_playing_delegates_to_client(self, adapter, mock_voice_client):
        """is_playing() should delegate to underlying client."""
        mock_voice_client.is_playing.return_value = True
        assert adapter.is_playing() is True

        mock_voice_client.is_playing.return_value = False
        assert adapter.is_playing() is False

    def test_start_recording_delegates_to_client(self, adapter, mock_voice_client):
        """start_recording() should delegate to underlying client."""
        sink = MagicMock()
        callback = MagicMock()

        adapter.start_recording(sink, callback)

        mock_voice_client.start_recording.assert_called_once_with(sink, callback)

    def test_stop_recording_delegates_to_client(self, adapter, mock_voice_client):
        """stop_recording() should delegate to underlying client."""
        adapter.stop_recording()
        mock_voice_client.stop_recording.assert_called_once()

    def test_recording_property(self, adapter, mock_voice_client):
        """recording property should delegate to underlying client."""
        mock_voice_client.recording = True
        assert adapter.recording is True

        mock_voice_client.recording = False
        assert adapter.recording is False

    def test_is_connected_delegates_to_client(self, adapter, mock_voice_client):
        """is_connected() should delegate to underlying client."""
        mock_voice_client.is_connected.return_value = True
        assert adapter.is_connected() is True

        mock_voice_client.is_connected.return_value = False
        assert adapter.is_connected() is False

    @pytest.mark.asyncio
    async def test_set_speaking_delegates_to_client(self, adapter, mock_voice_client):
        """set_speaking() should delegate to ws.speak()."""
        await adapter.set_speaking(True)
        mock_voice_client.ws.speak.assert_called_with(True)

        await adapter.set_speaking(False)
        mock_voice_client.ws.speak.assert_called_with(False)

    @pytest.mark.asyncio
    async def test_set_speaking_handles_errors_gracefully(
        self, adapter, mock_voice_client
    ):
        """set_speaking() should not raise on websocket errors."""
        mock_voice_client.ws.speak = AsyncMock(side_effect=Exception("WS error"))

        # Should not raise
        await adapter.set_speaking(True)

    def test_guild_property(self, adapter, mock_voice_client):
        """guild property should delegate to underlying client."""
        assert adapter.guild is mock_voice_client.guild

    def test_user_property(self, adapter, mock_voice_client):
        """user property should delegate to underlying client."""
        assert adapter.user is mock_voice_client.user


class TestMockVoiceClientAdapter:
    """Tests for MockVoiceClientAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a mock adapter."""
        return MockVoiceClientAdapter()

    def test_initial_state(self, adapter):
        """Should start with default state."""
        assert adapter.is_playing() is False
        assert adapter.recording is False
        assert adapter.is_connected() is True
        assert adapter.guild is None
        assert adapter.user is None

    def test_play_records_call(self, adapter):
        """play() should record call."""
        source = MagicMock()
        callback = MagicMock()

        adapter.play(source, after=callback)

        assert len(adapter.play_calls) == 1
        assert adapter.play_calls[0] == (source, callback)
        assert adapter.is_playing() is True

    def test_stop_records_call(self, adapter):
        """stop() should record call and update state."""
        adapter.play(MagicMock())  # Start playing
        adapter.stop()

        assert adapter.stop_calls == 1
        assert adapter.is_playing() is False

    def test_start_recording_records_call(self, adapter):
        """start_recording() should record call."""
        sink = MagicMock()
        callback = MagicMock()

        adapter.start_recording(sink, callback)

        assert len(adapter.start_recording_calls) == 1
        assert adapter.start_recording_calls[0] == (sink, callback)
        assert adapter.recording is True

    def test_stop_recording_records_call(self, adapter):
        """stop_recording() should record call."""
        adapter.start_recording(MagicMock(), MagicMock())  # Start recording
        adapter.stop_recording()

        assert adapter.stop_recording_calls == 1
        assert adapter.recording is False

    @pytest.mark.asyncio
    async def test_set_speaking_records_call(self, adapter):
        """set_speaking() should record call."""
        await adapter.set_speaking(True)
        await adapter.set_speaking(False)

        assert adapter.set_speaking_calls == [True, False]

    def test_set_connected(self, adapter):
        """set_connected() should update state."""
        adapter.set_connected(False)
        assert adapter.is_connected() is False

        adapter.set_connected(True)
        assert adapter.is_connected() is True

    def test_set_guild(self, adapter):
        """set_guild() should update property."""
        guild = MagicMock()
        adapter.set_guild(guild)
        assert adapter.guild is guild

    def test_set_user(self, adapter):
        """set_user() should update property."""
        user = MagicMock()
        adapter.set_user(user)
        assert adapter.user is user

    def test_simulate_playback_complete(self, adapter):
        """simulate_playback_complete() should call after callback."""
        callback = MagicMock()
        adapter.play(MagicMock(), after=callback)

        adapter.simulate_playback_complete()

        callback.assert_called_once_with(None)
        assert adapter.is_playing() is False

    def test_simulate_playback_complete_with_error(self, adapter):
        """simulate_playback_complete() should pass error to callback."""
        callback = MagicMock()
        error = Exception("Playback error")
        adapter.play(MagicMock(), after=callback)

        adapter.simulate_playback_complete(error)

        callback.assert_called_once_with(error)

    def test_reset_clears_state(self, adapter):
        """reset() should clear all state and calls."""
        # Set up some state
        adapter.play(MagicMock())
        adapter.start_recording(MagicMock(), MagicMock())
        adapter.stop_calls = 5
        adapter.stop_recording_calls = 3
        adapter.set_connected(False)

        adapter.reset()

        assert adapter.is_playing() is False
        assert adapter.recording is False
        assert adapter.is_connected() is True
        assert len(adapter.play_calls) == 0
        assert adapter.stop_calls == 0
        assert len(adapter.start_recording_calls) == 0
        assert adapter.stop_recording_calls == 0

    def test_multiple_plays_recorded(self, adapter):
        """Multiple play calls should all be recorded."""
        adapter.play(MagicMock())
        adapter.stop()
        adapter.play(MagicMock())

        assert len(adapter.play_calls) == 2
        assert adapter.stop_calls == 1
