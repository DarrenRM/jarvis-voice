"""Discord bot with voice assistant support.

Integrates voice session for wake word, STT, Claude, and TTS.
"""

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands

from ..config import Settings
from ..detection import VAD, WakeWordDetector
from ..stt import Transcriber
from ..llm import BaseLLMClient
from ..tts import TTSSynthesizer
from .voice_session import VoiceSession

logger = logging.getLogger(__name__)


class ClaudeBot(commands.Bot):
    """Discord bot with Claude voice assistant.

    Commands:
    - !join: Join your voice channel and start listening
    - !leave: Leave voice channel
    - !status: Show bot status
    """

    def __init__(
        self,
        settings: Settings,
        *,
        wake_detector: Optional[WakeWordDetector] = None,
        vad: Optional[VAD] = None,
        transcriber: Optional[Transcriber] = None,
        llm: Optional[BaseLLMClient] = None,
        tts: Optional[TTSSynthesizer] = None,
    ):
        """Initialize bot.

        Args:
            settings: Application settings
            wake_detector: Shared WakeWordDetector (optional)
            vad: Shared VAD (optional)
            transcriber: Shared Transcriber (optional)
            llm: Shared LLM client (Claude or Groq) (optional)
            tts: Shared TTSSynthesizer (optional)
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True

        super().__init__(
            command_prefix="!",
            intents=intents,
        )

        self.settings = settings

        # Shared components (can be None, session will create if needed)
        self._wake_detector = wake_detector
        self._vad = vad
        self._transcriber = transcriber
        self._llm = llm
        self._tts = tts

        # Active voice session (single channel at a time)
        self._session: Optional[VoiceSession] = None
        self._session_task: Optional[asyncio.Task] = None

        self._register_commands()

    def _register_commands(self) -> None:
        """Register bot commands."""

        @self.command(name="join")
        async def join_voice(ctx: commands.Context):
            """Switch to the user's voice channel."""
            if not ctx.author.voice:
                await ctx.send("You need to be in a voice channel!")
                return

            channel = ctx.author.voice.channel

            # Check if already in this channel
            if ctx.voice_client and ctx.voice_client.channel == channel:
                await ctx.send(f"Already in **{channel.name}**!")
                return

            # Join (or switch to) the channel
            await self._join_channel(channel, ctx.channel)

        @self.command(name="leave")
        async def leave_voice(ctx: commands.Context):
            """Leave the voice channel and stop the session."""
            if not ctx.voice_client:
                await ctx.send("Not in a voice channel!")
                return

            # Stop session
            if self._session:
                await self._session.stop()
                self._session = None

            # Cancel session task
            if self._session_task:
                self._session_task.cancel()
                try:
                    await self._session_task
                except asyncio.CancelledError:
                    pass
                self._session_task = None

            # Disconnect
            await ctx.voice_client.disconnect()
            await ctx.send("Left voice channel.")

        @self.command(name="status")
        async def show_status(ctx: commands.Context):
            """Show bot status."""
            if self._session:
                state = self._session.state
                vc = ctx.voice_client
                channel_name = vc.channel.name if vc else "Unknown"
                await ctx.send(
                    f"**Voice Session Active**\n"
                    f"Channel: {channel_name}\n"
                    f"State: {state}\n"
                    f"Wake word: {self.settings.wake_word_model.replace('_', ' ')}"
                )
            else:
                await ctx.send("No active voice session. Use !join to start.")

    async def _run_session(
        self,
        voice_client: discord.VoiceClient,
        notify_channel: Optional[discord.abc.Messageable] = None,
    ):
        """Run voice session and handle errors.

        Args:
            voice_client: Connected voice client
            notify_channel: Optional channel for error messages
        """
        try:
            await self._session.start(voice_client)
        except asyncio.CancelledError:
            logger.info("Session task cancelled")
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            if notify_channel:
                await notify_channel.send(f"Voice session error: {e}")
        finally:
            # Clean up
            self._session = None
            self._session_task = None
            if voice_client.is_connected():
                await voice_client.disconnect()

    async def _cleanup_session(self) -> None:
        """Stop current session and disconnect from voice."""
        if self._session:
            await self._session.stop()
            self._session = None

        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass
            self._session_task = None

        # Disconnect from all voice clients
        for vc in self.voice_clients:
            if vc.is_connected():
                await vc.disconnect()

    async def _join_channel(
        self,
        channel: discord.VoiceChannel,
        notify_channel: Optional[discord.abc.Messageable] = None,
    ) -> bool:
        """Join a voice channel and start session.

        Args:
            channel: Voice channel to join
            notify_channel: Optional channel for error messages

        Returns:
            True if joined successfully
        """
        # If already in a session, leave first
        if self._session is not None:
            await self._cleanup_session()

        try:
            # Create session with pre-loaded components
            self._session = VoiceSession(
                wake_detector=self._wake_detector,
                vad=self._vad,
                transcriber=self._transcriber,
                tts=self._tts,
            )

            # Initialize any remaining components
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._session._init_components)

            # Connect to voice channel
            logger.info(f"Connecting to voice channel: {channel.name}")
            voice_client = await channel.connect()

            # Start session in background task
            self._session_task = asyncio.create_task(
                self._run_session(voice_client, notify_channel)
            )
            logger.info(f"Joined voice channel: {channel.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to join voice: {e}", exc_info=True)
            self._session = None
            if notify_channel:
                await notify_channel.send(f"Failed to join: {e}")
            return False

    async def _auto_join(self) -> None:
        """Auto-join first available voice channel on startup."""
        for guild in self.guilds:
            # Find target channel
            if self.settings.auto_join_channel:
                # Find by name
                channel = discord.utils.get(
                    guild.voice_channels, name=self.settings.auto_join_channel
                )
            else:
                # First available
                channel = guild.voice_channels[0] if guild.voice_channels else None

            if channel:
                logger.info(f"Auto-joining voice channel: {channel.name} in {guild.name}")
                await self._join_channel(channel)
                return  # Only join one channel

        logger.warning("No voice channels available for auto-join")

    async def on_ready(self) -> None:
        """Handle bot ready event."""
        logger.info(f"Bot ready: {self.user.name} ({self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        logger.info(f"Wake word: {self.settings.wake_word_model.replace('_', ' ')}")

        # Auto-join voice channel if enabled
        if self.settings.auto_join_voice:
            await self._auto_join()

    async def close(self) -> None:
        """Clean up on shutdown."""
        logger.info("Bot closing...")

        # Stop active session
        if self._session:
            await self._session.stop()

        # Cancel session task
        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass

        # Disconnect from all voice channels
        for vc in self.voice_clients:
            try:
                await vc.disconnect(force=True)
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")

        await super().close()
