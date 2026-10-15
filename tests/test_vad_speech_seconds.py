"""VAD.speech_seconds: the check that captured audio is speech before the bot starts a
question over (clicks and game sounds through a mic aren't)."""

import wave
from math import gcd

import numpy as np
import pytest
from scipy.signal import resample_poly

from src.detection.vad import VAD

YES = "assets/audio/yes.wav"  # the bot saying "Yes?" (~0.4s of speech)


@pytest.fixture(scope="module")
def vad():
    v = VAD()
    if v._model is None:
        pytest.skip("Silero VAD not available")
    return v


def load(path: str) -> np.ndarray:
    with wave.open(path) as f:
        rate, channels = f.getframerate(), f.getnchannels()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)
    pcm = pcm.reshape(-1, channels).mean(axis=1) / 32768
    g = gcd(16000, rate)
    return resample_poly(pcm, 16000 // g, rate // g).astype(np.float32)


def test_speech_counts(vad):
    assert vad.speech_seconds(load(YES)) >= 0.3


def test_clicks_and_silence_do_not(vad):
    rng = np.random.default_rng(0)
    clicks = np.zeros(32000, dtype=np.float32)
    for start in range(0, len(clicks), 9600):  # 0.2s bursts every 0.6s, like Pat's mic
        clicks[start:start + 3200] = rng.normal(0, 0.2, 3200)
    assert vad.speech_seconds(clicks) < 0.1
    assert vad.speech_seconds(np.zeros(16000, dtype=np.float32)) == 0.0


def test_fresh_state_each_time(vad):
    speech = load(YES)
    assert vad.speech_seconds(speech) == vad.speech_seconds(speech)


def test_no_model_is_none():
    v = VAD.__new__(VAD)
    v._model = None
    assert v.speech_seconds(np.zeros(16000, dtype=np.float32)) is None
