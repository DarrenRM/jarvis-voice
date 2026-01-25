"""Audio resampling utilities for Discord format conversion.

Discord audio format: 48kHz stereo s16le (signed 16-bit little-endian)
Pipeline audio format: 16kHz mono float32
TTS output format: 24kHz mono float32
"""

import numpy as np
from scipy import signal


def resample_48k_stereo_to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """Convert Discord 48kHz stereo PCM to 16kHz mono float32.

    Args:
        audio_bytes: Raw PCM bytes (48kHz, stereo, s16le)

    Returns:
        Audio as float32 numpy array (16kHz, mono)
    """
    if len(audio_bytes) == 0:
        return np.array([], dtype=np.float32)

    # Convert bytes to int16 array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    # Deinterleave stereo to get left and right channels
    # Audio comes as [L, R, L, R, ...]
    left = audio_int16[0::2]
    right = audio_int16[1::2]

    # Mix to mono (average of channels)
    mono = (left.astype(np.float32) + right.astype(np.float32)) / 2.0

    # Normalize to -1.0 to 1.0
    mono = mono / 32768.0

    # Resample from 48kHz to 16kHz (factor of 3)
    # Use resample_poly for integer ratios - 10-50x faster than FFT-based resample()
    # This reduces GIL contention that causes audio pops when user talks during playback
    if len(mono) < 3:
        return np.array([], dtype=np.float32)

    resampled = signal.resample_poly(mono, 1, 3)

    return resampled.astype(np.float32)


def resample_24k_mono_to_48k_stereo(audio_float32: np.ndarray) -> bytes:
    """Convert TTS 24kHz mono float32 to Discord 48kHz stereo PCM.

    Args:
        audio_float32: Audio as float32 numpy array (24kHz, mono, -1 to 1)

    Returns:
        Raw PCM bytes (48kHz, stereo, s16le)
    """
    if len(audio_float32) == 0:
        return b""

    # Resample from 24kHz to 48kHz (factor of 2)
    num_samples_out = len(audio_float32) * 2
    resampled = signal.resample(audio_float32, num_samples_out)

    # Clip to valid range and convert to int16
    resampled = np.clip(resampled, -1.0, 1.0)
    audio_int16 = (resampled * 32767).astype(np.int16)

    # Interleave to stereo (duplicate mono to both channels)
    # Output should be [L, R, L, R, ...]
    stereo = np.empty(len(audio_int16) * 2, dtype=np.int16)
    stereo[0::2] = audio_int16  # Left channel
    stereo[1::2] = audio_int16  # Right channel

    return stereo.tobytes()


def resample_16k_mono_to_48k_stereo(audio_float32: np.ndarray) -> bytes:
    """Convert 16kHz mono float32 to Discord 48kHz stereo PCM.

    Useful for playing back recorded audio or generated sounds.

    Args:
        audio_float32: Audio as float32 numpy array (16kHz, mono, -1 to 1)

    Returns:
        Raw PCM bytes (48kHz, stereo, s16le)
    """
    if len(audio_float32) == 0:
        return b""

    # Resample from 16kHz to 48kHz (factor of 3)
    num_samples_out = len(audio_float32) * 3
    resampled = signal.resample(audio_float32, num_samples_out)

    # Clip to valid range and convert to int16
    resampled = np.clip(resampled, -1.0, 1.0)
    audio_int16 = (resampled * 32767).astype(np.int16)

    # Interleave to stereo
    stereo = np.empty(len(audio_int16) * 2, dtype=np.int16)
    stereo[0::2] = audio_int16
    stereo[1::2] = audio_int16

    return stereo.tobytes()
