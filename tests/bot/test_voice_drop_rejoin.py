"""When Discord drops the bot's voice connection, it rejoins instead of staying out."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.bot.discord_bot import ClaudeBot


def _bot(session=True, closing=False):
    bot = ClaudeBot.__new__(ClaudeBot)
    bot._session = MagicMock() if session else None
    bot._session_task = None
    bot._rejoin_task = None
    bot._closing = closing
    bot._connection = SimpleNamespace(user=SimpleNamespace(id=1))
    bot._rejoin_after_drop = AsyncMock()
    return bot


def _own_drop():
    channel = SimpleNamespace(name="micro pp")
    return SimpleNamespace(id=1), SimpleNamespace(channel=channel), SimpleNamespace(channel=None)


async def test_drop_with_live_session_rejoins():
    bot = _bot()
    member, before, after = _own_drop()
    await ClaudeBot.on_voice_state_update(bot, member, before, after)
    await asyncio.sleep(0)
    bot._rejoin_after_drop.assert_awaited_once_with(before.channel)


async def test_drop_while_closing_does_not_rejoin():
    bot = _bot(closing=True)
    await ClaudeBot.on_voice_state_update(bot, *_own_drop())
    await asyncio.sleep(0)
    bot._rejoin_after_drop.assert_not_called()


async def test_leave_command_does_not_rejoin():
    # !leave clears _session before disconnecting
    bot = _bot(session=False)
    await ClaudeBot.on_voice_state_update(bot, *_own_drop())
    await asyncio.sleep(0)
    bot._rejoin_after_drop.assert_not_called()
