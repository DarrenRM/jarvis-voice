"""StreamingSink speaker lock and wake-word watch, as used by follow-up windows."""

import numpy as np

from src.bot.voice_sink import StreamingSink

ASKER, OTHER = 111, 222


def pcm(ms: int = 20) -> bytes:
    """48kHz stereo s16le with some signal (Discord's decoded format)."""
    samples = 48 * ms
    tone = (np.sin(np.arange(samples) / 5) * 8000).astype(np.int16)
    return np.repeat(tone, 2).tobytes()


def make_sink(**kwargs) -> StreamingSink:
    return StreamingSink(vad=None, **kwargs)


def buffered_seconds(sink: StreamingSink) -> float:
    return len(sink.get_audio_buffer()) / 16000


class TestSpeakerLock:
    def test_follow_up_hears_only_the_person_it_answered(self):
        # Going idle after a turn clears the wake word user; the lock must still hold
        sink = make_sink()
        sink.start_listening(user_id=ASKER)
        sink.write(pcm(), OTHER)
        assert buffered_seconds(sink) == 0
        sink.write(pcm(), ASKER)
        assert buffered_seconds(sink) > 0

    def test_lock_released_when_listening_stops(self):
        sink = make_sink()
        sink.start_listening(user_id=ASKER)
        sink.stop_listening()
        assert sink._listen_user_id is None and not sink._watch_wake_word


class TestWakeWordWatch:
    def test_someone_elses_wake_word_is_reported(self, monkeypatch):
        woke = []
        sink = make_sink(on_wake_word=lambda: woke.append(True))
        monkeypatch.setattr(sink._wake_processor, "process_audio", lambda uid, audio: uid == OTHER)
        sink.start_listening(user_id=ASKER, watch_wake_word=True)
        sink.write(pcm(), OTHER)
        assert woke == [True]
        assert buffered_seconds(sink) == 0  # their audio still isn't part of the follow-up

    def test_not_watched_in_a_normal_turn(self, monkeypatch):
        woke = []
        sink = make_sink(on_wake_word=lambda: woke.append(True))
        monkeypatch.setattr(sink._wake_processor, "process_audio", lambda uid, audio: True)
        sink.start_listening(user_id=ASKER)
        sink.write(pcm(), OTHER)
        assert woke == []
