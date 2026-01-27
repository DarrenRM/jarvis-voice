"""Voice client adapter for testability.

Provides an abstraction over discord.VoiceClient to enable unit testing
of VoiceSession without requiring a live Discord connection.
"""

import logging
from typing import Any, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import discord
    from discord import AudioSource, Guild, Member

logger = logging.getLogger(__name__)


@runtime_checkable
class VoiceClientAdapter(Protocol):
    """Protocol defining voice client operations.

    This protocol allows for both production (DiscordVoiceClientAdapter)
    and test (MockVoiceClientAdapter) implementations.
    """

    # Playback
    def play(
        self,
        source: "AudioSource",
        *,
        after: Optional[Callable[[Optional[Exception]], Any]] = None,
    ) -> None:
        """Play audio from a source."""
        ...

    def stop(self) -> None:
        """Stop current playback."""
        ...

    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        ...

    # Recording
    def start_recording(self, sink: Any, callback: Callable) -> None:
        """Start recording audio."""
        ...

    def stop_recording(self) -> None:
        """Stop recording audio."""
        ...

    @property
    def recording(self) -> bool:
        """Check if currently recording."""
        ...

    # Connection
    def is_connected(self) -> bool:
        """Check if connected to voice channel."""
        ...

    # Speaking indicator
    async def set_speaking(self, speaking: bool) -> None:
        """Set Discord speaking indicator."""
        ...

    # Guild access
    @property
    def guild(self) -> Optional["Guild"]:
        """Get the guild this client is connected to."""
        ...

    @property
    def user(self) -> Optional["Member"]:
        """Get the bot's member in the guild."""
        ...


class DiscordVoiceClientAdapter:
    """Production adapter wrapping discord.VoiceClient.

    Delegates all operations to the underlying Discord voice client,
    with graceful error handling for websocket operations.
    """

    def __init__(self, voice_client: "discord.VoiceClient"):
        """Initialize adapter.

        Args:
            voice_client: The Discord voice client to wrap.
        """
        self._client = voice_client

    def play(
        self,
        source: "AudioSource",
        *,
        after: Optional[Callable[[Optional[Exception]], Any]] = None,
    ) -> None:
        """Play audio from a source."""
        self._client.play(source, after=after)

    def stop(self) -> None:
        """Stop current playback."""
        self._client.stop()

    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        return self._client.is_playing()

    def start_recording(self, sink: Any, callback: Callable) -> None:
        """Start recording audio."""
        self._client.start_recording(sink, callback)

    def stop_recording(self) -> None:
        """Stop recording audio."""
        self._client.stop_recording()

    @property
    def recording(self) -> bool:
        """Check if currently recording."""
        return self._client.recording

    def is_connected(self) -> bool:
        """Check if connected to voice channel."""
        return self._client.is_connected()

    async def set_speaking(self, speaking: bool) -> None:
        """Set Discord speaking indicator.

        Handles websocket errors gracefully as the websocket may not
        be ready or may have disconnected.
        """
        try:
            await self._client.ws.speak(speaking)
        except Exception:
            # May fail if websocket not ready or disconnected
            pass

    @property
    def guild(self) -> Optional["Guild"]:
        """Get the guild this client is connected to."""
        return self._client.guild

    @property
    def user(self) -> Optional["Member"]:
        """Get the bot's member in the guild."""
        return self._client.user


class MockVoiceClientAdapter:
    """Test adapter that records all operations.

    Useful for unit testing VoiceSession without requiring a live
    Discord connection.
    """

    def __init__(self):
        """Initialize mock adapter."""
        self._playing = False
        self._recording = False
        self._connected = True
        self._speaking = False

        # Recorded calls for assertions
        self.play_calls: list[tuple[Any, Optional[Callable]]] = []
        self.stop_calls: int = 0
        self.start_recording_calls: list[tuple[Any, Callable]] = []
        self.stop_recording_calls: int = 0
        self.set_speaking_calls: list[bool] = []

        # Mock guild and user
        self._guild: Optional[Any] = None
        self._user: Optional[Any] = None

        # Callback for simulating playback completion
        self._after_callback: Optional[Callable] = None

    def play(
        self,
        source: Any,
        *,
        after: Optional[Callable[[Optional[Exception]], Any]] = None,
    ) -> None:
        """Record play call and store callback."""
        self.play_calls.append((source, after))
        self._playing = True
        self._after_callback = after

    def stop(self) -> None:
        """Record stop call."""
        self.stop_calls += 1
        self._playing = False

    def is_playing(self) -> bool:
        """Return mock playing state."""
        return self._playing

    def start_recording(self, sink: Any, callback: Callable) -> None:
        """Record start_recording call."""
        self.start_recording_calls.append((sink, callback))
        self._recording = True

    def stop_recording(self) -> None:
        """Record stop_recording call."""
        self.stop_recording_calls += 1
        self._recording = False

    @property
    def recording(self) -> bool:
        """Return mock recording state."""
        return self._recording

    def is_connected(self) -> bool:
        """Return mock connected state."""
        return self._connected

    async def set_speaking(self, speaking: bool) -> None:
        """Record set_speaking call."""
        self.set_speaking_calls.append(speaking)
        self._speaking = speaking

    @property
    def guild(self) -> Optional[Any]:
        """Return mock guild."""
        return self._guild

    @property
    def user(self) -> Optional[Any]:
        """Return mock user."""
        return self._user

    # Test helper methods

    def set_connected(self, connected: bool) -> None:
        """Set mock connected state."""
        self._connected = connected

    def set_guild(self, guild: Any) -> None:
        """Set mock guild."""
        self._guild = guild

    def set_user(self, user: Any) -> None:
        """Set mock user."""
        self._user = user

    def simulate_playback_complete(self, error: Optional[Exception] = None) -> None:
        """Simulate playback completion by calling the after callback."""
        self._playing = False
        if self._after_callback:
            self._after_callback(error)
            self._after_callback = None

    def reset(self) -> None:
        """Reset all recorded calls and state."""
        self._playing = False
        self._recording = False
        self._connected = True
        self._speaking = False
        self.play_calls.clear()
        self.stop_calls = 0
        self.start_recording_calls.clear()
        self.stop_recording_calls = 0
        self.set_speaking_calls.clear()
        self._after_callback = None
