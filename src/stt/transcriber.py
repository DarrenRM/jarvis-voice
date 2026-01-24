"""Speech-to-text transcription using Faster-Whisper."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    """Speech-to-text using Faster-Whisper.

    Features:
    - Fast transcription with GPU acceleration
    - Streaming support for partial results
    - Configurable model size
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        """Initialize transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Compute device (cuda, cpu)
            compute_type: Compute precision (float16, int8, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Faster-Whisper model."""
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(
                f"Loaded Whisper {self.model_size} on {self.device} "
                f"with {self.compute_type}"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data (float32, 16kHz, mono)
            language: Language code

        Returns:
            Transcribed text
        """
        if self._model is None:
            logger.error("Whisper model not loaded")
            return ""

        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,  # Use built-in VAD for better accuracy
            )

            # Collect all segments
            text = " ".join(segment.text.strip() for segment in segments)
            logger.debug(f"Transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    async def transcribe_streaming(
        self,
        audio: np.ndarray,
        language: str = "en",
    ):
        """Transcribe audio with streaming results.

        Args:
            audio: Audio data
            language: Language code

        Yields:
            Partial transcription results
        """
        if self._model is None:
            return

        try:
            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                word_timestamps=True,
            )

            for segment in segments:
                yield segment.text.strip()

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
