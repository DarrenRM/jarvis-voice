"""Voice Activity Detection using Silero VAD."""

import logging
from typing import Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detection using Silero VAD.

    Features:
    - Low latency speech detection
    - Endpoint detection (speech start/end)
    - Configurable sensitivity
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
    ):
        """Initialize VAD.

        Args:
            threshold: Speech detection threshold (0-1)
            sample_rate: Audio sample rate (16000 or 8000)
            min_silence_duration_ms: Silence duration to trigger speech end
            speech_pad_ms: Padding before/after speech
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        # Silero VAD requires exactly 512 samples at 16kHz (256 at 8kHz)
        self._chunk_size = 512 if sample_rate == 16000 else 256

        # Load Silero VAD model and create iterator
        self._model = None
        self._vad_iterator = None
        self._load_model()

        # Separate buffers for different use cases
        self._process_buffer = np.array([], dtype=np.float32)
        self._endpoint_buffer = np.array([], dtype=np.float32)

        # State tracking
        self._is_speech = False
        self._speech_start_time: Optional[float] = None
        self._total_samples = 0

    def _load_model(self) -> None:
        """Load Silero VAD model."""
        try:
            from silero_vad import load_silero_vad, VADIterator

            self._model = load_silero_vad()
            self._vad_iterator = VADIterator(
                self._model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
            logger.info("Silero VAD model loaded with VADIterator")
        except ImportError:
            # Fallback to torch.hub if silero_vad package not installed
            logger.info("silero_vad package not found, using torch.hub")
            self._load_model_legacy()
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def _load_model_legacy(self) -> None:
        """Load Silero VAD model using torch.hub (legacy method)."""
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            logger.info("Silero VAD model loaded (legacy)")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def speech_seconds(self, audio: np.ndarray) -> Optional[float]:
        """Seconds of speech in a whole clip, from a fresh model state (None if the
        model isn't loaded). Resets this VAD before and after: use a separate instance
        from the one following live audio."""
        if self._model is None:
            return None
        self.reset()
        try:
            speech = 0
            for i in range(0, len(audio) - self._chunk_size + 1, self._chunk_size):
                is_speech, _ = self.process(audio[i:i + self._chunk_size])
                speech += is_speech
            return speech * self._chunk_size / self.sample_rate
        finally:
            self.reset()

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speech = False
        self._speech_start_time = None
        self._total_samples = 0
        self._process_buffer = np.array([], dtype=np.float32)
        self._endpoint_buffer = np.array([], dtype=np.float32)
        if self._vad_iterator is not None:
            self._vad_iterator.reset_states()
        elif self._model is not None:
            self._model.reset_states()

    def process(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Process audio chunk and detect speech.

        Args:
            audio: Audio chunk (float32, -1 to 1)

        Returns:
            Tuple of (is_speech, confidence)
        """
        if self._model is None:
            return False, 0.0

        # Buffer audio until we have enough
        self._process_buffer = np.concatenate([self._process_buffer, audio])

        if len(self._process_buffer) < self._chunk_size:
            return self._is_speech, 0.5 if self._is_speech else 0.0

        # Process all complete chunks, keep remainder
        speech_prob = 0.0
        while len(self._process_buffer) >= self._chunk_size:
            chunk = self._process_buffer[:self._chunk_size]
            self._process_buffer = self._process_buffer[self._chunk_size:]

            # Convert to tensor
            audio_tensor = torch.from_numpy(chunk).float()

            # Get speech probability
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob >= self.threshold

        # Track state transitions
        if is_speech and not self._is_speech:
            self._is_speech = True
        elif not is_speech and self._is_speech:
            self._is_speech = False

        return is_speech, speech_prob

    def detect_endpoint(self, audio: np.ndarray) -> Optional[str]:
        """Detect speech endpoints using VADIterator.

        Args:
            audio: Audio chunk to process

        Returns:
            "start" if speech started, "end" if speech ended, None otherwise
        """
        if self._vad_iterator is None:
            # Fallback to basic processing if VADIterator not available
            return self._detect_endpoint_legacy(audio)

        # Buffer audio
        self._endpoint_buffer = np.concatenate([self._endpoint_buffer, audio])

        result = None
        chunks_processed = 0
        # Process all complete chunks
        while len(self._endpoint_buffer) >= self._chunk_size:
            chunk = self._endpoint_buffer[:self._chunk_size]
            self._endpoint_buffer = self._endpoint_buffer[self._chunk_size:]
            self._total_samples += self._chunk_size
            chunks_processed += 1

            # Convert to tensor
            audio_tensor = torch.from_numpy(chunk).float()

            # Get speech event from VADIterator
            speech_dict = self._vad_iterator(audio_tensor, return_seconds=True)

            if speech_dict:
                if "start" in speech_dict:
                    self._is_speech = True
                    self._speech_start_time = speech_dict["start"]
                    result = "start"
                    logger.info(f"VAD: Speech STARTED at {speech_dict['start']:.2f}s")
                elif "end" in speech_dict:
                    self._is_speech = False
                    result = "end"
                    logger.info(f"VAD: Speech ENDED at {speech_dict['end']:.2f}s")

        if chunks_processed > 0:
            logger.debug(f"VAD: Processed {chunks_processed} chunks, is_speech={self._is_speech}, buffer_remaining={len(self._endpoint_buffer)}")

        return result

    def _detect_endpoint_legacy(self, audio: np.ndarray) -> Optional[str]:
        """Legacy endpoint detection without VADIterator."""
        was_speaking = self._is_speech
        is_speech, _ = self.process(audio)

        if is_speech and not was_speaking:
            return "start"

        # Simple silence-based end detection
        if was_speaking and not is_speech:
            self._is_speech = False
            return "end"

        return None

    @property
    def is_speech(self) -> bool:
        """Whether speech is currently detected."""
        return self._is_speech

    @property
    def speech_duration_s(self) -> float:
        """Duration of current speech in seconds."""
        if not self._is_speech or self._speech_start_time is None:
            return 0.0
        current_time = self._total_samples / self.sample_rate
        return current_time - self._speech_start_time
