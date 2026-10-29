"""StreamingSink keeps recording when a packet fails, and shares its models safely.

py-cord 2.8's router thread stops recording for good if sink.write() raises, so one bad
packet (or a wake model that fails to load for a new speaker) would leave the bot in the
channel, looking fine, but deaf until restarted.
"""

import threading
import time

import numpy as np

from src.bot.voice_sink import StreamingSink
from src.sink import per_user_manager

SPEAKER, NEWCOMER = 111, 222


def pcm(ms: int = 20) -> bytes:
    """48kHz stereo s16le with some signal (Discord's decoded format)."""
    samples = 48 * ms
    tone = (np.sin(np.arange(samples) / 5) * 8000).astype(np.int16)
    return np.repeat(tone, 2).tobytes()


class TestWriteNeverRaises:
    def test_a_failing_packet_is_dropped_and_the_next_one_still_counts(self, monkeypatch):
        woke = []
        sink = StreamingSink(vad=None, on_wake_word=lambda: woke.append(True))
        calls = []

        def process_audio(user_id, audio):
            calls.append(user_id)
            if len(calls) == 1:
                raise RuntimeError("bad packet")
            return True

        monkeypatch.setattr(sink._wake_processor, "process_audio", process_audio)

        sink.write(pcm(), SPEAKER)  # must not raise into py-cord's router thread
        sink.write(pcm(), SPEAKER)

        assert calls == [SPEAKER, SPEAKER]
        assert woke == [True]
        assert sink._write_errors == 1

    def test_errors_are_logged_but_not_once_per_packet(self, monkeypatch, caplog):
        sink = StreamingSink(vad=None)

        def broken(user_id, audio):
            raise RuntimeError("still broken")

        monkeypatch.setattr(sink._wake_processor, "process_audio", broken)
        with caplog.at_level("ERROR", logger="src.bot.voice_sink"):
            for _ in range(50):
                sink.write(pcm(), SPEAKER)

        assert sink._write_errors == 50
        assert len([r for r in caplog.records if "Dropped an audio packet" in r.message]) == 1


class TestWakeModelLoadFailure:
    def test_new_speaker_whose_model_fails_to_load_does_not_stop_recording(
        self, monkeypatch
    ):
        attempts = []

        class FailingDetector:
            def __init__(self, **kwargs):
                attempts.append(kwargs)
                raise OSError("model download failed")

        monkeypatch.setattr("src.detection.WakeWordDetector", FailingDetector)
        sink = StreamingSink(vad=None)

        for _ in range(5):
            sink.write(pcm(), NEWCOMER)

        # Contained inside the manager (no packet errors), and not retried every packet
        assert sink._write_errors == 0
        assert len(attempts) == 1
        assert sink._user_manager.get(NEWCOMER).detector is None

    def test_loading_is_tried_again_later(self, monkeypatch):
        loaded = []

        class FlakyDetector:
            def __init__(self, **kwargs):
                if not loaded:
                    loaded.append("failed")
                    raise OSError("model download failed")
                loaded.append("ok")

        monkeypatch.setattr("src.detection.WakeWordDetector", FlakyDetector)
        manager = per_user_manager.PerUserAudioManager()
        state = manager.get_or_create(SPEAKER)

        assert manager.ensure_detector(state) is None
        assert manager.ensure_detector(state) is None  # still backing off
        manager._detector_retry_at = time.monotonic() - 1  # the wait is over
        assert manager.ensure_detector(state) is not None
        assert loaded == ["failed", "ok"]


class FakeDetector:
    def __init__(self, **kwargs):
        pass

    def process(self, audio):
        return False

    def flush(self):
        return False

    def reset(self):
        pass


class TestSlowModelLoad:
    def test_a_slow_load_does_not_hold_the_lock_the_event_loop_waits_on(self, monkeypatch):
        """A new speaker's detector can take a while to load (seconds on a fresh install,
        while the model files download). The session's idle poll runs on the event loop
        every 200 ms and takes the sink's lock: it must not wait out the load."""
        loading, finish = threading.Event(), threading.Event()

        class SlowDetector(FakeDetector):
            def __init__(self, **kwargs):
                loading.set()
                finish.wait(5.0)

        monkeypatch.setattr("src.detection.WakeWordDetector", SlowDetector)
        sink = StreamingSink(vad=None)
        router = threading.Thread(target=sink.write, args=(pcm(), NEWCOMER))
        router.start()
        try:
            assert loading.wait(2.0)
            poll = threading.Thread(target=sink.check_idle_audio_gap)  # the idle poll
            poll.start()
            poll.join(1.0)
            assert not poll.is_alive(), "the idle poll waited for the model to load"
        finally:
            finish.set()
            router.join(5.0)

        assert isinstance(sink._user_manager.get(NEWCOMER).detector, SlowDetector)
        assert sink._write_errors == 0

    def test_the_next_packet_uses_the_loaded_detector(self, monkeypatch):
        heard = []

        class Detector(FakeDetector):
            def process(self, audio):
                heard.append(len(audio))
                return False

        monkeypatch.setattr("src.detection.WakeWordDetector", Detector)
        sink = StreamingSink(vad=None)
        sink.write(pcm(), NEWCOMER)  # asks for a detector, loaded after the lock
        sink.write(pcm(), NEWCOMER)

        assert heard and isinstance(sink._user_manager.get(NEWCOMER).detector, Detector)

    def test_models_are_downloaded_at_startup_before_the_bot_connects(self, monkeypatch):
        import src.main as main_module

        downloads = []
        monkeypatch.setattr(main_module, "download_wake_word_models",
                            lambda: downloads.append(True))
        monkeypatch.setattr(main_module, "VAD", lambda **kwargs: object())
        monkeypatch.setattr(main_module, "TTSSynthesizer", lambda **kwargs: object())

        main_module.preload_components(main_module.get_settings())

        assert downloads == [True]

    def test_a_failed_startup_download_is_logged_not_fatal(self, monkeypatch, caplog):
        import src.main as main_module

        def offline():
            raise OSError("no network")

        monkeypatch.setattr(main_module, "download_wake_word_models", offline)
        monkeypatch.setattr(main_module, "VAD", lambda **kwargs: object())
        monkeypatch.setattr(main_module, "TTSSynthesizer", lambda **kwargs: object())

        with caplog.at_level("ERROR", logger="src.main"):
            components = main_module.preload_components(main_module.get_settings())

        assert "Could not download the wake word models" in caplog.text
        assert set(components) == {"vad", "tts"}


class TestSharedModelsAreLocked:
    def test_listening_reset_waits_for_the_packet_being_processed(self):
        """vad.reset() from the event loop must not run inside detect_endpoint()."""
        overlaps = []
        in_detect = threading.Event()

        class SlowVAD:
            is_speech = False
            busy = False

            def detect_endpoint(self, audio):
                self.busy = True
                in_detect.set()
                time.sleep(0.2)
                self.busy = False
                return None

            def reset(self):
                if self.busy:
                    overlaps.append("reset during detect_endpoint")

        sink = StreamingSink(vad=SlowVAD())
        sink.start_listening(user_id=SPEAKER)
        router = threading.Thread(target=sink.write, args=(pcm(), SPEAKER))
        router.start()
        assert in_detect.wait(1.0)
        sink.start_listening(user_id=SPEAKER)  # the session starting a new turn
        router.join()

        assert overlaps == []
