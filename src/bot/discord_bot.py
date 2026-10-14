"""Discord bot with voice assistant support.

Integrates voice session for wake word, STT, Claude, and TTS.
"""

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands

from ..config import Settings
from ..detection import VAD
from ..llm import LLMService
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
        vad: Optional[VAD] = None,
        llm: Optional[LLMService] = None,
        tts: Optional[TTSSynthesizer] = None,
    ):
        """Initialize bot.

        Args:
            settings: Application settings
            vad: Shared VAD (optional)
            llm: Shared LLM client (Claude or Groq) (optional)
            tts: Shared TTSSynthesizer (optional)
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            # Text the model writes (or a spoken prompt injection) must never mass-ping:
            # only direct user mentions (tag_user's <@id>) notify anyone.
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True),
        )

        self.settings = settings

        # Shared components (can be None, session will create if needed)
        # Note: Wake word detectors are now created per-user in StreamingSink
        self._vad = vad
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
            # Note: Wake word detectors are now created per-user in StreamingSink
            self._session = VoiceSession(
                vad=self._vad,
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

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Handle voice state changes for user lifecycle management.

        Cleans up per-user wake detectors when users leave the voice channel.
        """
        # Skip if no active session or sink
        if not self._session or not self._session.sink:
            return

        # Handle bot's own disconnection
        if member.id == self.user.id:
            if before.channel and not after.channel:
                logger.warning(f"[DISCONNECT] Bot was disconnected from {before.channel.name}")
                # Clean up session
                if self._session:
                    asyncio.create_task(self._cleanup_session())
            return

        # User left voice channel entirely
        if before.channel and not after.channel:
            self._session.sink.remove_user(member.id)
            logger.debug(f"User {member.display_name} left voice, cleaned up")

        # User switched to a different channel (left our channel)
        elif before.channel != after.channel:
            # Check if they left the channel we're in
            if self._session.voice_client and before.channel == self._session.voice_client.channel:
                self._session.sink.remove_user(member.id)
                logger.debug(f"User {member.display_name} switched channels, cleaned up")

        # User joined our channel - trigger greeting
        if self._session.voice_client and after.channel == self._session.voice_client.channel:
            # They just arrived (weren't in our channel before)
            if before.channel != after.channel:
                asyncio.create_task(self._session.greet_user(member))

    async def close(self) -> None:
        """Clean up on shutdown."""
        logger.info("Bot closing...")

        # Stop active session
        if self._session:
            await self._session.stop()
            # Give recv_audio thread time to exit after stop_recording()
            # This prevents "Event loop is closed" errors during shutdown
            await asyncio.sleep(0.1)

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
                # Clear speaking state before disconnect to fix "stuck green circle"
                if vc.ws:
                    try:
                        await vc.ws.speak(False)
                    except Exception:
                        pass
                await vc.disconnect(force=True)
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")

        await super().close()
