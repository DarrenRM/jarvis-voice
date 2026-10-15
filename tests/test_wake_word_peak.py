"""The near-miss score covers misses only: a real wake (or a reset) starts it over."""

import numpy as np

from src.detection.wake_word import WakeWordDetector


class _Model:
    def __init__(self, scores):
        self._scores = iter(scores)

    def predict(self, audio):
        return {"hey_jarvis": next(self._scores)}

    def reset(self):
        pass


def _detector(scores):
    detector = WakeWordDetector.__new__(WakeWordDetector)
    detector.threshold = 0.5
    detector._model = _Model(scores)
    detector._on_detected = None
    detector._buffer = np.array([], dtype=np.float32)
    detector._min_samples = 320
    detector.peak = 0.0
    return detector


CHUNK = np.zeros(320, dtype=np.float32)


def test_miss_is_reported():
    detector = _detector([0.1, 0.37, 0.2])
    for _ in range(3):
        assert not detector.process(CHUNK)
    assert detector.take_peak() == 0.37


def test_hit_is_not_reported_as_a_near_miss():
    detector = _detector([0.3, 1.0, 0.0])
    detector.process(CHUNK)
    assert detector.process(CHUNK)
    detector.process(CHUNK)
    assert detector.take_peak() == 0.0


def test_reset_starts_over():
    detector = _detector([0.4])
    detector.process(CHUNK)
    detector.reset()
    assert detector.take_peak() == 0.0
