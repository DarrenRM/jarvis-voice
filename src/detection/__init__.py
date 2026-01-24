"""Speech detection module."""

from .vad import VAD
from .wake_word import WakeWordDetector, WhisperWakeWordDetector

__all__ = ["VAD", "WakeWordDetector", "WhisperWakeWordDetector"]
