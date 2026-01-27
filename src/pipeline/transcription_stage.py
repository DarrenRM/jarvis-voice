"""Transcription stage for voice pipeline.

This module provides STT with wake word stripping.
Extracts pattern from voice_session.py:956-984.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class STTClient(Protocol):
    """Protocol for Speech-to-Text clients."""

    async def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)

        Returns:
            Transcribed text, or empty string if no speech detected
        """
        ...


@dataclass
class TranscriptionResult:
    """Result from transcription stage.

    Attributes:
        text: Transcribed and processed text (wake word stripped)
        raw_text: Original transcription before processing
        wake_word_only: True if only wake word was detected
        is_empty: True if no speech was detected
    """

    text: str
    raw_text: str
    wake_word_only: bool = False
    is_empty: bool = False

    @classmethod
    def empty(cls) -> "TranscriptionResult":
        """Create result for empty/no speech."""
        return cls(text="", raw_text="", is_empty=True)

    @classmethod
    def wake_only(cls, raw_text: str) -> "TranscriptionResult":
        """Create result for wake-word-only detection."""
        return cls(text="", raw_text=raw_text, wake_word_only=True)


class TranscriptionStage(PipelineStage):
    """STT stage with wake word stripping.

    Extracts pattern from voice_session.py:956-984.

    Flow:
    1. Transcribe audio via STT client
    2. Strip wake word prefix if present
    3. Return processed text or special result for wake-word-only

    Example:
        stage = TranscriptionStage(stt_client)
        result = await stage.process(audio, context)

        if result.is_empty:
            # No speech detected
        elif result.wake_word_only:
            # User said only wake word - trigger follow-up
        else:
            # Normal transcription in result.text
    """

    # Wake word pattern for stripping from transcription
    # Matches variations like: "hey jarvis", "hey, jarvis", "a jarvis"
    # Pattern extracted from voice_session.py:956-959
    DEFAULT_WAKE_PATTERN = re.compile(
        r"^(hey[,\s]*jarvis|a[,\s]*jarvis)[,.\s!?]*",
        re.IGNORECASE,
    )

    # Minimum meaningful text length after stripping
    MIN_TEXT_LENGTH = 3

    def __init__(
        self,
        stt_client: STTClient,
        wake_pattern: Optional[re.Pattern] = None,
        min_text_length: int = MIN_TEXT_LENGTH,
    ):
        """Initialize transcription stage.

        Args:
            stt_client: Speech-to-text client (e.g., GroqWhisperClient)
            wake_pattern: Optional custom wake word pattern (default: "hey jarvis")
            min_text_length: Minimum text length after wake word stripping
        """
        self._stt = stt_client
        self._wake_pattern = wake_pattern or self.DEFAULT_WAKE_PATTERN
        self._min_text_length = min_text_length

    @property
    def name(self) -> str:
        """Stage name."""
        return "transcription"

    async def process(
        self,
        audio: np.ndarray,
        context: PipelineContext,
    ) -> TranscriptionResult:
        """Transcribe audio and strip wake word.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
            context: Pipeline context (speaker info)

        Returns:
            TranscriptionResult with processed text and status flags
        """
        # Transcribe
        raw_text = await self._stt.transcribe_audio(audio)

        if not raw_text or not raw_text.strip():
            logger.debug("[TRANSCRIPTION] No speech detected")
            return TranscriptionResult.empty()

        logger.info(f"[TRANSCRIPTION] Raw: '{raw_text}'")

        # Strip wake word
        processed = self._strip_wake_word(raw_text)

        if processed is None:
            logger.info("[TRANSCRIPTION] Wake word only detected")
            return TranscriptionResult.wake_only(raw_text)

        logger.info(f"[TRANSCRIPTION] Processed: '{processed}'")
        return TranscriptionResult(text=processed, raw_text=raw_text)

    def _strip_wake_word(self, text: str) -> Optional[str]:
        """Strip wake word from transcription.

        Extracts pattern from voice_session.py:961-984.

        Args:
            text: Raw transcription text

        Returns:
            Text with wake word stripped, or None if only wake word was said
        """
        text_stripped = text.strip()
        match = self._wake_pattern.match(text_stripped)

        if match:
            text_stripped = text_stripped[match.end() :].strip()
            if text_stripped:
                logger.debug(
                    f"[WAKE-STRIP] Stripped wake word, remaining: '{text_stripped}'"
                )
            else:
                return None

        # If stripping left nothing meaningful
        if not text_stripped or len(text_stripped) < self._min_text_length:
            return None

        return text_stripped


class ConfigurableTranscriptionStage(TranscriptionStage):
    """Transcription stage with configurable wake words.

    Allows setting wake word pattern at runtime, useful for
    testing or supporting multiple wake words.
    """

    def __init__(
        self,
        stt_client: STTClient,
        wake_words: Optional[list[str]] = None,
        min_text_length: int = TranscriptionStage.MIN_TEXT_LENGTH,
    ):
        """Initialize with optional wake word list.

        Args:
            stt_client: Speech-to-text client
            wake_words: List of wake word phrases (default: ["hey jarvis", "a jarvis"])
            min_text_length: Minimum text length after wake word stripping
        """
        if wake_words:
            # Build pattern from word list
            escaped_words = [re.escape(w) for w in wake_words]
            pattern_str = r"^(" + "|".join(escaped_words) + r")[,.\s!?]*"
            wake_pattern = re.compile(pattern_str, re.IGNORECASE)
        else:
            wake_pattern = None

        super().__init__(
            stt_client=stt_client,
            wake_pattern=wake_pattern,
            min_text_length=min_text_length,
        )

    def set_wake_words(self, wake_words: list[str]) -> None:
        """Update wake word pattern at runtime.

        Args:
            wake_words: List of wake word phrases
        """
        if not wake_words:
            self._wake_pattern = self.DEFAULT_WAKE_PATTERN
            return

        escaped_words = [re.escape(w) for w in wake_words]
        pattern_str = r"^(" + "|".join(escaped_words) + r")[,.\s!?]*"
        self._wake_pattern = re.compile(pattern_str, re.IGNORECASE)
