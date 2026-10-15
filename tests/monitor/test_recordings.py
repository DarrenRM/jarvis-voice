"""Audio kept for the IRL test set (MONITOR_SAVE_AUDIO). Fakes only: nothing reaches Discord."""

import json

import numpy as np

from src.monitor import turn as tr
from src.monitor.recordings import AudioArchive, read_wav, write_wav
from src.monitor.reporting import recording_path
from src.monitor.service import APPROVE, REJECT, MonitorService
from src.sink.wake_processor import WakeWordProcessor

from .helpers import BOT_ID, OWNER_ID, FakeChannel, FakeClock, event, good_turn, make_settings

TONE = (0.3 * np.sin(np.linspace(0, 400, 16000))).astype(np.float32)  # 1s


def recording_service(tmp_path, **settings):
    channel = FakeChannel()
    service = MonitorService(
        make_settings(MONITOR_SAVE_AUDIO=True, MONITOR_AUDIO_DIR=str(tmp_path / "rec"),
                      **settings),
        get_channel=lambda: channel,
        bot_user_id=lambda: BOT_ID,
        approvals_path=tmp_path / "approved.md",
        dismissals_path=tmp_path / "dismissed.md",
        clock=FakeClock(),
    )
    return service, channel


def bad_turn():
    return good_turn(outcome=tr.ERROR_REPLY_OUTCOME, question_audio=TONE,
                     events=[event("LLM API error: Request timed out.")])


class TestArchive:
    def test_wav_round_trip(self, tmp_path):
        write_wav(tmp_path / "a.wav", TONE)
        back = read_wav(tmp_path / "a.wav")
        assert len(back) == len(TONE) and np.abs(back - TONE).max() < 1e-3

    def test_turn_saved_with_sidecar(self, tmp_path):
        archive = AudioArchive(tmp_path)
        path = archive.save_turn(good_turn(question_audio=TONE), [])
        archive.close()
        assert path.parent.name == "turns" and path.name.startswith("20261014-210500_sam_")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["heard"] == "what's the capital of Australia?"
        assert meta["audio_seconds"] == 1.0 and meta["findings"] == []
        assert len(read_wav(path)) == len(TONE)

    def test_same_second_gets_unique_names(self, tmp_path):
        archive = AudioArchive(tmp_path)
        paths = {archive.save_turn(good_turn(question_audio=TONE), []) for _ in range(3)}
        archive.close()
        assert len(paths) == 3 and all(p.exists() for p in paths)

    def test_no_audio_no_file(self, tmp_path):
        archive = AudioArchive(tmp_path)
        assert archive.save_turn(good_turn(), []) is None
        archive.close()

    def test_wake_clips_skip_low_near_misses(self, tmp_path):
        archive = AudioArchive(tmp_path, wake_min_score=0.2, names={7: "Sam"})
        archive.save_wake(7, TONE, 0.91, True)
        archive.save_wake(7, TONE, 0.37, False)
        archive.save_wake(7, TONE, 0.08, False)  # everyday speech
        archive.close()
        names = sorted(p.name for p in (tmp_path / "wake").glob("*.wav"))
        assert len(names) == 2
        assert names[0].endswith("_sam_hit_0.91.wav") or names[1].endswith("_hit_0.91.wav")
        assert any(n.endswith("_sam_miss_0.37.wav") for n in names)

    def test_oldest_pruned_past_the_limit(self, tmp_path):
        archive = AudioArchive(tmp_path, max_mb=0.08)  # room for two 1s clips (~32 KB each)
        for score in (0.91, 0.92, 0.93, 0.94):
            archive.save_wake(1, TONE, score, True)
        archive.close()
        total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
        assert total <= 0.08 * 1024 * 1024
        assert any(p.name.endswith("_0.94.wav") for p in (tmp_path / "wake").glob("*.wav"))

    def test_reaction_lands_in_sidecar(self, tmp_path):
        archive = AudioArchive(tmp_path)
        path = archive.save_turn(good_turn(question_audio=TONE), [])
        archive.note_reaction(path, "confirmed", "https://discord.com/x")
        archive.close()
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["owner_reaction"] == "confirmed" and meta["discord"] == "https://discord.com/x"


class TestService:
    def test_off_by_default(self, tmp_path):
        service = MonitorService(make_settings(), get_channel=lambda: None,
                                 approvals_path=tmp_path / "a.md")
        assert service.archive is None and not service.recorder.keep_question_audio

    async def test_good_turn_saved_but_not_posted(self, tmp_path):
        service, channel = recording_service(tmp_path)
        turn = good_turn(question_audio=TONE)
        assert await service.handle_turn(turn) is None
        await service.stop()
        assert turn.question_audio is None  # not held in memory
        assert len(list((tmp_path / "rec" / "turns").glob("*.wav"))) == 1
        assert channel.sent == []

    async def test_hang_snapshot_not_saved(self, tmp_path):
        service, _ = recording_service(tmp_path)
        await service.handle_turn(good_turn(outcome=tr.HUNG, idle_s=31.0, question_audio=TONE))
        await service.stop()
        assert not (tmp_path / "rec" / "turns").exists() or \
            not list((tmp_path / "rec" / "turns").glob("*.wav"))

    async def test_post_names_recording_and_reactions_reach_sidecar(self, tmp_path):
        service, _ = recording_service(tmp_path)
        message = await service.handle_turn(bad_turn())
        rel = recording_path(message.content)
        assert rel and rel.endswith(".wav")
        assert await service.handle_reaction(message.id, OWNER_ID, APPROVE) == "approved"
        assert rel in service.approvals.path.read_text(encoding="utf-8")
        await service.stop()
        meta = json.loads((tmp_path / "rec" / "turns" / rel.rsplit("/", 1)[1])
                          .with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["owner_reaction"] == "confirmed"
        assert meta["findings"][0]["detector"] == "error_reply"

    async def test_dismissal_reaches_sidecar(self, tmp_path):
        service, _ = recording_service(tmp_path)
        message = await service.handle_turn(bad_turn())
        assert await service.handle_reaction(message.id, OWNER_ID, REJECT) == "dismissed"
        await service.stop()
        rel = recording_path(message.content)
        meta = json.loads((tmp_path / "rec" / "turns" / rel.rsplit("/", 1)[1])
                          .with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["owner_reaction"] == "dismissed"

    def test_recorder_hooks(self, tmp_path):
        service, _ = recording_service(tmp_path)
        recorder = service.recorder
        token = recorder.turn_started(1, "Sam")
        recorder.question_audio(TONE)
        assert recorder._turn.question_audio is TONE
        recorder.turn_ended(token, None)
        clips = []
        recorder.on_wake_clip = lambda *a: clips.append(a)
        recorder.wake_clip(7, TONE, 0.4, False)
        assert clips and clips[0][0] == 7 and clips[0][3] is False


class _State:
    def __init__(self, user_id, audio, detector):
        self.user_id, self._audio, self.detector = user_id, audio, detector

    def get_pre_buffer(self):
        return self._audio


class TestWakeProcessorClips:
    def test_report_clip_uses_last_hit_score(self):
        class Det:
            last_score = 0.83

        processor = WakeWordProcessor(user_manager=None)
        got = []
        processor.on_clip = lambda *a: got.append(a)
        processor.report_clip(_State(5, TONE, Det()), hit=True)
        processor.report_clip(_State(5, TONE, Det()), hit=False, score=0.31)
        assert [(g[0], g[2], g[3]) for g in got] == [(5, 0.83, True), (5, 0.31, False)]

    def test_callback_errors_are_swallowed(self):
        processor = WakeWordProcessor(user_manager=None)

        def boom(*a):
            raise RuntimeError("disk full")

        processor.on_clip = boom
        processor.report_clip(_State(5, TONE, None), hit=True, score=0.9)  # no raise

    def test_no_callback_is_a_no_op(self):
        WakeWordProcessor(user_manager=None).report_clip(_State(5, TONE, None), hit=True)
