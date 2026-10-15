"""MonitorService: posting, dedupe, approvals. Fakes only: nothing reaches Discord."""

from src.llm.chat_orchestrator import ERROR_REPLY
from src.monitor import turn as tr
from src.monitor.reporting import REACTION_HINT
from src.monitor.service import (
    APPROVE,
    MAX_POSTS_PER_HOUR,
    REJECT,
    ApprovalLog,
    MonitorService,
    create_monitor,
)
from src.monitor.turn import Clip

from .helpers import (
    BOT_ID,
    FAKE_SECRET,
    OWNER_ID,
    FakeChannel,
    FakeClock,
    FakeMessage,
    event,
    good_turn,
    make_settings,
)


def error_turn(message: str = "LLM API error: Request timed out.", **changes):
    return good_turn(outcome=tr.ERROR_REPLY_OUTCOME, reply=ERROR_REPLY,
                     events=[event(message)], **changes)


def make_service(tmp_path, channel=None, clock=None, **settings):
    channel = channel if channel is not None else FakeChannel()
    service = MonitorService(
        make_settings(**settings),
        get_channel=lambda: channel,
        bot_user_id=lambda: BOT_ID,
        approvals_path=tmp_path / "approved-fixes.md",
        dismissals_path=tmp_path / "dismissed-findings.md",
        clock=clock or FakeClock(),
    )
    return service, channel


class TestDisabled:
    def test_create_monitor_off_by_default(self):
        assert create_monitor(make_settings(MONITOR_ENABLED=False)) is None

    def test_defaults(self):
        from src.config import Settings

        s = Settings(_env_file=None, DISCORD_TOKEN="x")
        assert s.monitor_enabled is False and s.monitor_channel_name == "bot-monitor"
        assert s.monitor_owner_id is None and s.monitor_retranscribe is False

    def test_blank_owner_id_is_none(self):
        assert make_settings(MONITOR_OWNER_ID="").monitor_owner_id is None

    async def test_disabled_service_posts_nothing(self, tmp_path):
        service, channel = make_service(tmp_path, MONITOR_ENABLED=False)
        assert await service.handle_turn(error_turn()) is None
        service._on_turn(error_turn())
        service.start()  # no-op when off
        assert channel.sent == []
        assert service.recorder._handler is None
        assert await service.handle_reaction(1, OWNER_ID, APPROVE) is None


class TestPosting:
    async def test_posts_with_reactions(self, tmp_path):
        service, channel = make_service(tmp_path)
        message = await service.handle_turn(error_turn())
        assert channel.sent == [message]
        assert message.reactions == [APPROVE, REJECT]
        text = message.content
        assert text.startswith("**Bot monitor: Error reply spoken**")
        assert 'Heard from Sam: "what\'s the capital of Australia?"' in text
        assert "Request timed out" in text and "Timings: first audio 0.7s" in text
        assert len(text) <= 2000
        assert channel.kwargs[0]["allowed_mentions"] is not None

    async def test_good_turn_posts_nothing(self, tmp_path):
        service, channel = make_service(tmp_path)
        assert await service.handle_turn(good_turn()) is None
        assert channel.sent == []

    async def test_secrets_are_redacted(self, tmp_path):
        service, channel = make_service(tmp_path)
        message = f"LLM API error: bad key {FAKE_SECRET} Bearer abc123def456"
        await service.handle_turn(error_turn(message))
        text = channel.sent[0].content
        assert FAKE_SECRET not in text and "abc123def456" not in text
        assert "[redacted]" in text

    async def test_user_text_cannot_ping(self, tmp_path):
        service, channel = make_service(tmp_path)
        await service.handle_turn(error_turn(heard="tell @everyone **now**"))
        text = channel.sent[0].content
        assert "@everyone" not in text and "\\*\\*now\\*\\*" in text

    async def test_missing_channel_only_logs(self, tmp_path):
        service = MonitorService(make_settings(), get_channel=lambda: None,
                                 approvals_path=tmp_path / "a.md", clock=FakeClock())
        assert await service.handle_turn(error_turn()) is None

    async def test_send_failure_is_swallowed(self, tmp_path):
        class Broken(FakeChannel):
            async def send(self, content, **kwargs):
                raise RuntimeError("Missing Permissions")

        service, _ = make_service(tmp_path, channel=Broken())
        assert await service.handle_turn(error_turn()) is None

    async def test_one_post_per_turn_lists_everything(self, tmp_path):
        service, channel = make_service(tmp_path)
        turn = error_turn(clips=[Clip(0.7, 0.8, 1.5), Clip(12.0, 2.0, 14.0, answer=True)])
        await service.handle_turn(turn)
        assert len(channel.sent) == 1
        assert "(also: slow answer)" in channel.sent[0].content


class TestDedupe:
    async def test_repeat_within_window_bumps_count(self, tmp_path):
        clock = FakeClock()
        service, channel = make_service(tmp_path, clock=clock)
        first = await service.handle_turn(error_turn("LLM API error: Request timed out after 15s"))
        clock.advance(60)
        again = await service.handle_turn(error_turn("LLM API error: Request timed out after 16s"))
        assert again is None and len(channel.sent) == 1
        assert "Seen 2 times" in first.content
        assert first.content.endswith(REACTION_HINT)
        clock.advance(60)
        await service.handle_turn(error_turn("LLM API error: Request timed out after 9.5s"))
        assert "Seen 3 times" in first.content and "Seen 2 times" not in first.content

    async def test_new_post_after_window(self, tmp_path):
        clock = FakeClock()
        service, channel = make_service(tmp_path, clock=clock, MONITOR_DEDUPE_MINUTES=30)
        await service.handle_turn(error_turn())
        clock.advance(31 * 60)
        await service.handle_turn(error_turn())
        assert len(channel.sent) == 2

    async def test_different_problem_posts(self, tmp_path):
        service, channel = make_service(tmp_path)
        await service.handle_turn(error_turn())
        await service.handle_turn(good_turn(reply="My weather feeds just ghosted me."))
        assert len(channel.sent) == 2

    async def test_hourly_cap(self, tmp_path):
        service, channel = make_service(tmp_path)
        for i in range(MAX_POSTS_PER_HOUR + 3):
            await service.handle_turn(good_turn(
                outcome=tr.CRASHED, error=f"Error{'x' * i}: boom", reply="", clips=[]))
        assert len(channel.sent) == MAX_POSTS_PER_HOUR


class TestApprovals:
    async def posted(self, tmp_path):
        service, channel = make_service(tmp_path)
        message = await service.handle_turn(error_turn())
        return service, message

    async def test_owner_check_appends_once(self, tmp_path):
        service, message = await self.posted(tmp_path)
        assert await service.handle_reaction(message.id, OWNER_ID, APPROVE) == "approved"
        text = service.approvals.path.read_text(encoding="utf-8")
        assert text.startswith("# Approved fixes")
        assert f"<!-- discord-message:{message.id} -->" in text
        assert "Error reply spoken" in text and "Request timed out" in text
        assert message.jump_url in text
        assert "  - The LLM call failed" in text
        # A check confirms something went wrong; the diagnosis still needs checking
        assert "confirmed something went wrong" in text and "monitor's guess" in text
        assert "Confirmed: added to the fix list" in message.content
        # Re-running (a second reaction event, a restart) adds nothing
        assert await service.handle_reaction(message.id, OWNER_ID, APPROVE) == "already_approved"
        assert service.approvals.path.read_text(encoding="utf-8") == text

    async def test_cross_is_recorded_as_dismissed(self, tmp_path):
        service, message = await self.posted(tmp_path)
        assert await service.handle_reaction(message.id, OWNER_ID, REJECT) == "dismissed"
        assert not service.approvals.path.exists()
        text = service.dismissals.path.read_text(encoding="utf-8")
        assert text.startswith("# Dismissed findings")
        assert f"<!-- discord-message:{message.id} -->" in text
        assert "didn't disrupt the flow" in text and "Error reply spoken" in text
        assert await service.handle_reaction(message.id, OWNER_ID, REJECT) == "already_dismissed"
        assert service.dismissals.path.read_text(encoding="utf-8") == text

    async def test_only_owner_counts(self, tmp_path):
        service, message = await self.posted(tmp_path)
        assert await service.handle_reaction(message.id, 777, APPROVE) is None
        assert await service.handle_reaction(message.id, BOT_ID, APPROVE) is None
        assert await service.handle_reaction(message.id, OWNER_ID, "\U0001F44D") is None
        assert not service.approvals.path.exists()

    async def test_no_owner_configured(self, tmp_path):
        service, channel = make_service(tmp_path, MONITOR_OWNER_ID="")
        message = await service.handle_turn(error_turn())
        assert await service.handle_reaction(message.id, OWNER_ID, APPROVE) is None

    async def test_post_from_before_restart_is_fetched(self, tmp_path):
        old, message = await self.posted(tmp_path)
        fresh, _ = make_service(tmp_path)  # new process: no memory of the post

        async def fetch():
            return message

        assert await fresh.handle_reaction(message.id, OWNER_ID, APPROVE, fetch) == "approved"
        assert f"discord-message:{message.id}" in fresh.approvals.path.read_text(encoding="utf-8")

    async def test_unrelated_message_is_not_approved(self, tmp_path):
        service, _ = make_service(tmp_path)
        others = [FakeMessage(5, "**Bot monitor: fake**", author_id=OWNER_ID),
                  FakeMessage(6, "hello there"),
                  FakeMessage(7, "**Bot monitor: x**", channel_name="general")]
        for msg in others:
            async def fetch(msg=msg):
                return msg
            assert await service.handle_reaction(msg.id, OWNER_ID, APPROVE, fetch) is None
        assert not service.approvals.path.exists()

    async def test_unknown_message_without_fetch(self, tmp_path):
        service, _ = make_service(tmp_path)
        assert await service.handle_reaction(123, OWNER_ID, APPROVE) is None


class TestApprovalLog:
    def test_keeps_crlf_files_crlf(self, tmp_path):
        path = tmp_path / "fixes.md"
        path.write_bytes(b"# Approved fixes\r\n\r\nIntro.\r\n")
        log = ApprovalLog(path)
        assert log.add(5, "## Entry\n<!-- discord-message:5 -->\n- detail\n")
        data = path.read_bytes()
        assert b"<!-- discord-message:5 -->\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
        assert not log.add(5, "## Again\n<!-- discord-message:5 -->\n")
        assert data == path.read_bytes()

    def test_creates_file_with_header(self, tmp_path):
        log = ApprovalLog(tmp_path / "sub" / "fixes.md")
        assert log.add(9, "<!-- discord-message:9 -->")
        assert log.contains(9) and not log.contains(10)


class TestRetranscribe:
    async def test_mismatch_detected_from_reply_audio(self, tmp_path):
        import numpy as np

        class FakeWhisper:
            calls = 0

            async def transcribe_audio(self, audio):
                FakeWhisper.calls += 1
                assert audio.dtype == np.float32 and len(audio) == 16000
                return "It's Canberra. It's Canberra. It's Canberra. Most people guess Sydney."

        service = MonitorService(
            make_settings(MONITOR_RETRANSCRIBE=True), get_channel=FakeChannel,
            transcriber=FakeWhisper(), approvals_path=tmp_path / "a.md", clock=FakeClock())
        assert service.recorder._retain_audio
        turn = good_turn(answer_audio=[np.zeros(24000, dtype=np.float32)])
        message = await service.handle_turn(turn)
        assert FakeWhisper.calls == 1
        assert "Spoken audio doesn't match" in message.content
        assert turn.answer_audio == []  # not kept around
