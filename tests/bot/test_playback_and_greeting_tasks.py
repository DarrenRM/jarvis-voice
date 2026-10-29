"""Playback never waits forever, and background greetings are held and their errors logged."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import src.bot.voice_session as vs
from src.bot.discord_bot import ClaudeBot
from src.bot.voice_session import VoiceSession


class SilentVoiceClient:
    """Starts playing but never calls the after-callback (and never stops on its own)."""

    def __init__(self):
        self.playing = False
        self.stopped = False

    def is_playing(self):
        return self.playing

    def play(self, source, *, after=None):
        self.playing = True

    def stop(self):
        self.playing = False
        self.stopped = True


async def test_lost_playback_callback_does_not_hang_the_session(monkeypatch):
    monkeypatch.setattr(vs, "PLAYBACK_CALLBACK_GRACE_SECONDS", 0.1)
    session = VoiceSession.__new__(VoiceSession)
    session.voice_client = SilentVoiceClient()
    session.turn_observer = None
    session._loop = asyncio.get_running_loop()

    clip = np.zeros(240, dtype=np.float32)  # 10 ms at 24 kHz
    await asyncio.wait_for(session._play_audio(clip), timeout=2.0)

    assert session.voice_client.stopped


def _bot_with_session(greet_user):
    bot = ClaudeBot.__new__(ClaudeBot)
    channel = SimpleNamespace(name="game night")
    session = MagicMock()
    session.sink = MagicMock()
    session.voice_client = SimpleNamespace(channel=channel)
    session.greet_user = greet_user
    bot._session = session
    bot._greetings = set()
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=1))
    return bot, channel


async def test_greeting_task_is_held_until_done():
    started, release = asyncio.Event(), asyncio.Event()

    async def greet_user(member):
        started.set()
        await release.wait()

    bot, channel = _bot_with_session(greet_user)
    member = SimpleNamespace(id=2, display_name="Sam")
    await ClaudeBot.on_voice_state_update(
        bot, member, SimpleNamespace(channel=None), SimpleNamespace(channel=channel)
    )
    await started.wait()
    assert len(bot._greetings) == 1

    release.set()
    await asyncio.sleep(0.01)
    assert bot._greetings == set()


async def test_greeting_errors_are_logged(caplog):
    async def greet_user(member):
        raise RuntimeError("TTS down")

    bot, channel = _bot_with_session(greet_user)
    member = SimpleNamespace(id=2, display_name="Sam")
    with caplog.at_level("WARNING", logger="src.bot.discord_bot"):
        await ClaudeBot.on_voice_state_update(
            bot, member, SimpleNamespace(channel=None), SimpleNamespace(channel=channel)
        )
        await asyncio.sleep(0.01)

    assert "Greeting failed" in caplog.text and "TTS down" in caplog.text
    assert bot._greetings == set()
