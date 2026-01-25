"""Audio processing module."""

from .resampler import (
    resample_48k_stereo_to_16k_mono,
    resample_24k_mono_to_48k_stereo,
    resample_16k_mono_to_48k_stereo,
)

# Sample rates used throughout the pipeline
SAMPLE_RATE_DISCORD = 48000  # Discord native audio (stereo)
SAMPLE_RATE_PROCESSING = 16000  # Wake word, VAD, STT (mono)
SAMPLE_RATE_TTS = 24000  # Inworld TTS output (mono)

__all__ = [
    "resample_48k_stereo_to_16k_mono",
    "resample_24k_mono_to_48k_stereo",
    "resample_16k_mono_to_48k_stereo",
    "SAMPLE_RATE_DISCORD",
    "SAMPLE_RATE_PROCESSING",
    "SAMPLE_RATE_TTS",
]
