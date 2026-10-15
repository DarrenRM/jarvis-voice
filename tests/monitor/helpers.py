"""Fakes and builders for monitor tests (no Discord, no network)."""

from datetime import UTC, datetime
from types import SimpleNamespace

from src.monitor.turn import Clip, LiveTurn, LogEvent, ToolUse

BOT_ID = 900
OWNER_ID = 42
FAKE_SECRET = "csk-FAKEKEY0123456789abcdef"


def make_settings(**overrides):
    from src.config import Settings

    values = {
        "DISCORD_TOKEN": "fake-discord-token-for-tests",
        "CEREBRAS_API_KEY": FAKE_SECRET,
        "MONITOR_ENABLED": True,
        "MONITOR_OWNER_ID": OWNER_ID,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def good_turn(**changes) -> LiveTurn:
    """A normal answered turn: affirmation at 0.7s, two answer sentences from 3.2s."""
    turn = LiveTurn(
        started=datetime(2026, 10, 14, 21, 5),
        speaker_id=1, speaker_name="Sam",
        heard="what's the capital of Australia?", raw_heard="Hey Jarvis, what's the capital...",
        reply="[happy] It's Canberra.\nMost people guess Sydney.",
        reply_ok=True,
        clips=[Clip(start=0.7, seconds=0.8, end=1.5),
               Clip(start=3.2, seconds=1.2, end=4.4, answer=True),
               Clip(start=4.7, seconds=1.5, end=6.2, answer=True)],
        stt_s=0.3,
        llm_calls=[{"first_text_s": 0.4, "total_s": 0.9, "tool_calls": 0, "finish": "stop"}],
        tts_s=[0.6, 0.7],
        duration_s=6.4,
    )
    for key, value in changes.items():
        setattr(turn, key, value)
    return turn


def event(message: str, level: str = "ERROR",
          logger: str = "src.llm.chat_orchestrator") -> LogEvent:
    return LogEvent(at=1.0, level=level, logger=logger, message=message)


def tool(name: str, args: str = '"query"', ok: bool = True, error: str = "",
         no_results: bool = False) -> ToolUse:
    return ToolUse(name=name, args=args, ok=ok, error=error, no_results=no_results)


class FakeMessage:
    def __init__(self, message_id: int, content: str, author_id: int = BOT_ID,
                 channel_name: str = "bot-monitor"):
        self.id = message_id
        self.content = content
        self.author = SimpleNamespace(id=author_id)
        self.channel = SimpleNamespace(name=channel_name)
        self.jump_url = f"https://discord.com/channels/1/2/{message_id}"
        self.created_at = datetime(2026, 10, 14, 10, 0, tzinfo=UTC)
        self.reactions: list[str] = []
        self.edits: list[str] = []

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def edit(self, content: str) -> None:
        self.content = content
        self.edits.append(content)


class FakeChannel:
    name = "bot-monitor"

    def __init__(self):
        self.sent: list[FakeMessage] = []
        self.kwargs: list[dict] = []
        self._next_id = 1000

    async def send(self, content: str, **kwargs) -> FakeMessage:
        self._next_id += 1
        message = FakeMessage(self._next_id, content)
        self.sent.append(message)
        self.kwargs.append(kwargs)
        return message


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

