"""Wake word near-miss logging through the real WakeWordProcessor (scores faked)."""

import logging
import time

import numpy as np

from src.detection.wake_word import WakeWordDetector
from src.sink.per_user_manager import PerUserAudioManager
from src.sink.wake_processor import WakeWordProcessor

from .cases import SAM, SAM_WAKE_SCORES


class ScoredModel:
    """openwakeword stand-in: these scores, then 0 (silence)."""

    def __init__(self, scores):
        self._scores = list(scores)

    def predict(self, audio):
        return {"hey_jarvis": self._scores.pop(0) if self._scores else 0.0}

    def reset(self):
        pass


def processor_with(scores) -> tuple[WakeWordProcessor, object]:
    detector = WakeWordDetector.__new__(WakeWordDetector)
    detector.threshold = 0.5
    detector._model = ScoredModel(scores)
    detector._on_detected = None
    detector._buffer = np.array([], dtype=np.float32)
    detector._min_samples = 320
    detector.peak = 0.0
    manager = PerUserAudioManager()
    state = manager.get_or_create(SAM.id)
    state.detector = detector
    return WakeWordProcessor(manager), state


def speak_then_pause(processor, state, scores) -> list[bool]:
    hits = [processor.process_audio(SAM.id, np.zeros(1280, dtype=np.float32))
            for _ in scores]
    state.last_seen = time.time() - 0.5  # he paused: the idle gap check looks at him
    processor.check_idle_audio_gap()
    return hits


def near_misses(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Near miss" in r.getMessage()]


def test_a_real_wake_is_not_logged_as_a_near_miss(caplog):
    """2026-10-15 10:26: Sam's "Hey Jarvis" scored 1.00 and woke the bot, then the
    pause after it was logged as "near miss 1.00": the peak score wasn't cleared on a
    hit, so the monitor's near-miss reports were wrong."""
    processor, state = processor_with(SAM_WAKE_SCORES)
    with caplog.at_level(logging.INFO, logger="src.sink.wake_processor"):
        hits = speak_then_pause(processor, state, SAM_WAKE_SCORES)
    assert hits[-1] and processor.wake_word_user_id == SAM.id
    assert near_misses(caplog) == []


def test_a_real_miss_is_still_logged(caplog):
    """Control: the same build-up without the hit is a near miss worth logging."""
    scores = SAM_WAKE_SCORES[:-1]
    processor, state = processor_with(scores)
    with caplog.at_level(logging.INFO, logger="src.sink.wake_processor"):
        assert not any(speak_then_pause(processor, state, scores))
    assert len(near_misses(caplog)) == 1
    assert f"best score {max(scores):.2f}" in near_misses(caplog)[0]
