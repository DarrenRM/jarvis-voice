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
from ..llm import ClaudeClient
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
        claude: Optional[ClaudeClient] = None,
        tts: Optional[TTSSynthesizer] = None,
    ):
        """Initialize bot.

        Args:
            settings: Application settings
            wake_detector: Shared WakeWordDetector (optional)
            vad: Shared VAD (optional)
            transcriber: Shared Transcriber (optional)
            claude: Shared ClaudeClient (optional)
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
        self._claude = claude
        self._tts = tts

        # Active voice session (single channel at a time)
        self._session: Optional[VoiceSession] = None
        self._session_task: Optional[asyncio.Task] = None

        self._register_commands()

    def _register_commands(self) -> None:
        """Register bot commands."""

        @self.command(name="join")
        async def join_voice(ctx: commands.Context):
            """Join the user's voice channel and start listening."""
            if not ctx.author.voice:
                await ctx.send("You need to be in a voice channel!")
                return

            channel = ctx.author.voice.channel

            # Check if already in a session
            if self._session is not None:
                await ctx.send("Already in a voice session! Use !leave first.")
                return

            # Disconnect from any existing voice connection
            if ctx.voice_client:
                await ctx.voice_client.disconnect()

            try:
                # Create session with pre-loaded components
                self._session = VoiceSession(
                    wake_detector=self._wake_detector,
                    vad=self._vad,
                    transcriber=self._transcriber,
                    claude=self._claude,
                    tts=self._tts,
                )

                # Initialize any remaining components (STT clients)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._session._init_components)

                # NOW connect to voice channel
                logger.info(f"Connecting to voice channel: {channel.name}")
                voice_client = await channel.connect()

                await ctx.send(
                    f"Joined **{channel.name}**! "
                    f"Say '{self.settings.wake_word_model.replace('_', ' ')}' to activate."
                )

                # Start session in background task
                self._session_task = asyncio.create_task(
                    self._run_session(voice_client, ctx)
                )

            except Exception as e:
                logger.error(f"Failed to join voice: {e}", exc_info=True)
                await ctx.send(f"Failed to join: {e}")
                self._session = None
                # Disconnect if we managed to connect
                if ctx.voice_client:
                    await ctx.voice_client.disconnect()

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
        ctx: commands.Context,
    ):
        """Run voice session and handle errors.

        Args:
            voice_client: Connected voice client
            ctx: Command context for sending messages
        """
        try:
            await self._session.start(voice_client)
        except asyncio.CancelledError:
            logger.info("Session task cancelled")
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            await ctx.send(f"Voice session error: {e}")
        finally:
            # Clean up
            self._session = None
            self._session_task = None
            if voice_client.is_connected():
                await voice_client.disconnect()

    async def on_ready(self) -> None:
        """Handle bot ready event."""
        logger.info(f"Bot ready: {self.user.name} ({self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        logger.info(f"Wake word: {self.settings.wake_word_model.replace('_', ' ')}")

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
