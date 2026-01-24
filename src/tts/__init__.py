"""Text-to-speech module."""

from .synthesizer import TTSSynthesizer
from .streaming import StreamingTTSBuffer, stream_tts_from_llm

__all__ = ["TTSSynthesizer", "StreamingTTSBuffer", "stream_tts_from_llm"]
