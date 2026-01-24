"""Groq Whisper STT client for fast transcription.

Uses Groq's hosted Whisper Large v3 for ~189x realtime transcription.
"""

import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GroqWhisperClient:
    """STT client using Groq's Whisper API.

    Features:
    - ~189x realtime transcription speed
    - Simple REST API (no WebSocket complexity)
    - Whisper Large v3 accuracy
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
    ):
        """Initialize Groq Whisper client.

        Args:
            api_key: Groq API key
            model: Whisper model (whisper-large-v3 or whisper-large-v3-turbo)
            language: Language code (en, es, fr, etc.)
        """
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client = None

    def _ensure_client(self) -> None:
        """Initialize the Groq client if not already done."""
        if self._client is not None:
            return

        try:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
            logger.info(f"Groq Whisper client initialized (model: {self.model})")

        except ImportError:
            logger.error("groq not installed. Install with: pip install groq")
            raise

    async def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio using Groq Whisper.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)

        Returns:
            Transcript text
        """
        import asyncio

        self._ensure_client()

        try:
            # Convert float32 to int16 WAV bytes
            audio_int16 = (audio * 32767).astype(np.int16)

            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            import wave

            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())

            wav_buffer.seek(0)

            # Groq API is synchronous, run in executor
            loop = asyncio.get_running_loop()

            def do_transcribe():
                return self._client.audio.transcriptions.create(
                    file=("audio.wav", wav_buffer.read(), "audio/wav"),
                    model=self.model,
                    language=self.language,
                    response_format="text",
                )

            transcript = await loop.run_in_executor(None, do_transcribe)

            logger.info(f"[GROQ] Transcribed: {transcript[:50] if transcript else 'empty'}...")
            return transcript.strip() if transcript else ""

        except Exception as e:
            logger.error(f"Groq transcription error: {e}", exc_info=True)
            return ""
