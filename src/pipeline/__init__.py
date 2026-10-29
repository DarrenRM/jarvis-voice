"""Audio processing pipeline module."""

from .base import PipelineContext, PipelineStage, StreamingPipelineStage
from .response_stage import ResponseStage
from .transcription_stage import (
    ConfigurableTranscriptionStage,
    TranscriptionResult,
    TranscriptionStage,
)
from .tts_stage import TTSStage, TTSSynthesizerProtocol
from .voice_pipeline import OpeningPhrase, VoicePipeline

__all__ = [
    "ConfigurableTranscriptionStage",
    "PipelineContext",
    "PipelineStage",
    "ResponseStage",
    "StreamingPipelineStage",
    "TranscriptionResult",
    "TranscriptionStage",
    "TTSStage",
    "TTSSynthesizerProtocol",
    "OpeningPhrase",
    "VoicePipeline",
]
