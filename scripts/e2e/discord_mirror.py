"""--listen: play an e2e run into a real Discord voice channel so a person can hear it.

Logs in with the bot's token (stop the live bot first: one voice connection per guild),
joins the voice channel someone is in, and plays a live mix of the synthetic user
speech and everything the bot under test says. It only plays: nothing said in the
channel is recorded or fed to the test.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
from typing import Optional

import discord
import numpy as np

logger = logging.getLogger("e2e")

FRAME_BYTES = 3840  # 20 ms of 48 kHz stereo s16le
SILENCE = b"\x00" * FRAME_BYTES
PREBUFFER = 3  # frames of slack for thread timing jitter (60 ms)
MAX_LAG = 15  # drop audio that falls more than 300 ms behind


class _Stream:
    """Frames from one producer thread (the injector or the fake player)."""

    def __init__(self):
        self.frames: collections.deque[bytes] = collections.deque()
        self._primed = False
        self._waited = 0

    def push(self, frame: bytes) -> None:
        if len(frame) == FRAME_BYTES:
            self.frames.append(frame)

    def pop(self) -> Optional[bytes]:
        while len(self.frames) > MAX_LAG:
            self.frames.popleft()
        if not self._primed:
            # Start a burst once a little is buffered (or the burst is that short)
            if not self.frames:
                return None
            if len(self.frames) < PREBUFFER and self._waited < PREBUFFER:
                self._waited += 1
                return None
            self._primed, self._waited = True, 0
        if not self.frames:
            self._primed = False
            return None
        return self.frames.popleft()


class MixSource(discord.AudioSource):
    """Never ends: returns silence between bursts so one play() covers the whole run.
    DiscordMirror pauses the player while idle, so the bot doesn't show as speaking."""

    def __init__(self):
        self.user = _Stream()
        self.bot = _Stream()
        self.voiced = 0  # frames that carried audio
        self.idle_reads = 0  # silent frames in a row

    def pending(self) -> bool:
        return bool(self.user.frames or self.bot.frames)

    def read(self) -> bytes:
        parts = [f for f in (self.user.pop(), self.bot.pop()) if f]
        if not parts:
            self.idle_reads += 1
            return SILENCE
        self.idle_reads = 0
        self.voiced += 1
        if len(parts) == 1:
            return parts[0]
        mixed = sum(np.frombuffer(p, dtype=np.int16).astype(np.int32) for p in parts)
        return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()

    def is_opus(self) -> bool:
        return False


class DiscordMirror:
    """Runs the Discord client on its own thread and event loop. The harness loop blocks
    for seconds at a time (model loading), which stalls Discord's heartbeats and drops
    the voice connection if they share a loop."""

    def __init__(self):
        self._source = MixSource()
        self.user = self._source.user.push  # tap for harness.Injector
        self.bot = self._source.bot.push  # tap for harness.FakeVoiceClient
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._text_channel = None  # where the bot's chat posts are forwarded

    @classmethod
    async def join(cls, settings, timeout: float = 45.0) -> "DiscordMirror":
        mirror = cls()
        mirror._thread = threading.Thread(target=asyncio.run, args=(mirror._main(settings),),
                                          name="discord-mirror", daemon=True)
        mirror._thread.start()
        if not await asyncio.to_thread(mirror._ready.wait, timeout):
            raise TimeoutError("Discord voice join timed out")
        if mirror._error:
            raise RuntimeError(f"Discord mirror failed: {mirror._error}")
        return mirror

    async def _main(self, settings) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        client = discord.Client(intents=discord.Intents.default())
        ready = asyncio.Event()

        @client.event
        async def on_ready():
            ready.set()

        task = asyncio.create_task(client.start(settings.discord_token))
        vc = None
        try:
            waiter = asyncio.create_task(ready.wait())
            await asyncio.wait({task, waiter}, timeout=30, return_when=asyncio.FIRST_COMPLETED)
            if not ready.is_set():
                waiter.cancel()
                failed = task.done() and not task.cancelled() and task.exception()
                raise RuntimeError(f"login failed: {type(failed).__name__}" if failed
                                   else "login timed out")
            channel = self._find_channel(client, settings)
            self._text_channel = discord.utils.get(channel.guild.text_channels,
                                                   name=settings.image_channel_name)
            vc = await channel.connect()
            vc.play(self._source)
            logger.info(f"[LISTEN] Playing the run into voice channel {channel.name!r}")
            self._ready.set()
            source = self._source
            ticks = 0
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), 0.05)
                except asyncio.TimeoutError:
                    pass
                # Pause after 0.5 s of silence (speaking indicator off), resume on audio
                if vc.is_paused():
                    if source.pending():
                        source.idle_reads = 0
                        vc.resume()
                elif vc.is_playing() and source.idle_reads > 25:
                    vc.pause()
                ticks += 1
                if ticks % 100 == 0:  # every ~5 s
                    logger.debug(f"[LISTEN] {source.voiced * 0.02:.0f}s of audio so far, "
                                 f"connected={vc.is_connected()} playing={vc.is_playing()}")
                    if vc.is_connected() and not vc.is_playing() and not vc.is_paused():
                        logger.warning("[LISTEN] Discord player stopped; restarting it")
                        vc.play(source)
            await asyncio.sleep(0.5)  # let the last frames play out
            logger.info(f"[LISTEN] Played {self._source.voiced * 0.02:.0f}s of audio")
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            logger.exception("[LISTEN] Discord mirror failed")
        finally:
            self._ready.set()
            if vc is not None:
                vc.stop()
                try:
                    # Make sure "stopped speaking" goes out before the connection closes,
                    # or Discord can leave the bot's green ring on
                    await vc.ws.speak(discord.SpeakingState.none)
                    await asyncio.sleep(0.3)
                    await vc.disconnect(force=True)
                except Exception:
                    pass
            await client.close()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @staticmethod
    def _find_channel(client: discord.Client, settings):
        """The owner's voice channel if MONITOR_OWNER_ID is set, else the busiest one."""
        guilds = client.guilds
        if settings.discord_guild_id:
            guilds = [g for g in guilds if str(g.id) == str(settings.discord_guild_id)]
        owner = settings.monitor_owner_id
        best = None
        for guild in guilds:
            for channel in guild.voice_channels:
                people = [uid for uid in channel.voice_states if uid != client.user.id]
                if owner and owner in people:
                    return channel
                if people and (best is None or len(people) > best[0]):
                    best = (len(people), channel)
        if best is None:
            raise RuntimeError("Nobody is in a voice channel to listen")
        return best[1]

    async def post(self, content: str) -> None:
        """Forward a chat post from the bot under test to the real text channel."""
        if self._loop is None or self._text_channel is None:
            return
        send = self._text_channel.send(f"-# e2e test\n{content}",
                                       allowed_mentions=discord.AllowedMentions.none())
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(send, self._loop))

    async def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 15)
