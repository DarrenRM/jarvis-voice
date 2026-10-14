"""ClaudeBot creates the monitor only when enabled and routes reactions to it (no connection)."""

from types import SimpleNamespace

from src.bot.discord_bot import ClaudeBot
from src.monitor.service import APPROVE

from .helpers import OWNER_ID, make_settings


async def test_no_monitor_when_disabled():
    bot = ClaudeBot(make_settings(MONITOR_ENABLED=False))
    assert bot._monitor is None
    payload = SimpleNamespace(message_id=1, user_id=OWNER_ID, emoji=APPROVE, channel_id=2)
    await bot.on_raw_reaction_add(payload)  # nothing to do, no error


async def test_monitor_created_and_receives_reactions():
    bot = ClaudeBot(make_settings())
    assert bot._monitor is not None
    seen = []

    async def handle_reaction(message_id, user_id, emoji, fetch_message=None):
        seen.append((message_id, user_id, emoji))

    bot._monitor.handle_reaction = handle_reaction
    payload = SimpleNamespace(message_id=5, user_id=OWNER_ID, emoji=APPROVE, channel_id=2)
    await bot.on_raw_reaction_add(payload)
    assert seen == [(5, OWNER_ID, APPROVE)]
    assert bot._monitor_channel() is None  # not connected to any guild
