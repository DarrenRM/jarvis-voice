"""Wake word detection using OpenWakeWord."""

import logging
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Wake word detection using OpenWakeWord.

    Features:
    - Pre-trained wake words (hey_jarvis, alexa, hey_mycroft, etc.)
    - Custom wake word models
    - Low latency detection (<1ms)
    - Configurable confidence threshold
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        """Initialize wake word detector.

        Args:
            model_path: Path to custom wake word model
            model_name: Pre-trained model name (hey_jarvis, alexa, hey_mycroft, etc.)
            threshold: Detection confidence threshold
            sample_rate: Audio sample rate
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model_name = model_name
        self._model = None
        self._on_detected: Optional[Callable[[], None]] = None

        # Buffer for accumulating audio chunks
        # OpenWakeWord can handle variable chunk sizes, but prefers ~80ms
        # Using smaller minimum to avoid stalls when audio stops mid-buffer
        self._buffer = np.array([], dtype=np.float32)
        self._min_samples = int(sample_rate * 0.02)  # 20ms minimum (was 80ms)
        # Highest score since take_peak(): how close a missed wake word came
        self.peak = 0.0
        self.last_score = 0.0  # score of the latest hit

        self._load_model(model_path, model_name)

    def _load_model(self, model_path: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Load OpenWakeWord model."""
        try:
            import openwakeword
            from openwakeword import Model

            # Download default models if not present
            openwakeword.utils.download_models()

            if model_path and Path(model_path).exists():
                # Custom model
                self._model = Model(wakeword_models=[model_path])
                logger.info(f"Loaded custom wake word model: {model_path}")
            elif model_name:
                # Use specific pre-trained model
                self._model = Model(
                    wakeword_models=[model_name],
                    inference_framework="onnx",
                )
                logger.info(f"Loaded OpenWakeWord with '{model_name}' model")
            else:
                # Use all default models
                self._model = Model(
                    inference_framework="onnx",
                )
                logger.info("Loaded OpenWakeWord with default models")

        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            raise

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for wake word detection.

        Args:
            callback: Function to call when wake word detected
        """
        self._on_detected = callback

    def process(self, audio: np.ndarray) -> bool:
        """Process audio chunk for wake word.

        Args:
            audio: Audio chunk (float32, -1 to 1)

        Returns:
            True if wake word detected
        """
        if self._model is None:
            return False

        # Buffer audio until we have enough
        self._buffer = np.concatenate([self._buffer, audio])

        if len(self._buffer) < self._min_samples:
            return False

        # Process buffered audio
        audio_to_process = self._buffer
        self._buffer = np.array([], dtype=np.float32)

        # OpenWakeWord expects int16
        audio_int16 = (audio_to_process * 32767).astype(np.int16)

        # Process audio
        try:
            t0 = time.perf_counter()
            prediction = self._model.predict(audio_int16)
            inference_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            # Log but don't crash on ONNX errors (usually chunk size issues)
            logger.debug(f"OpenWakeWord prediction error: {e}")
            return False

        # Check all wake words
        for wakeword, scores in prediction.items():
            if isinstance(scores, (list, np.ndarray)):
                score = scores[-1] if len(scores) > 0 else 0
            else:
                score = scores
            self.peak = max(self.peak, float(score))

            if score >= self.threshold:
                self.peak = 0.0  # a hit isn't a near miss
                self.last_score = float(score)
                logger.info(
                    f"Wake word detected: {wakeword} (score: {score:.2f}, "
                    f"inference: {inference_ms:.1f}ms, chunk: {len(audio_int16)} samples)"
                )
                if self._on_detected:
                    self._on_detected()
                return True

        return False

    def take_peak(self) -> float:
        """Highest score since the last call (then starts over)."""
        peak, self.peak = self.peak, 0.0
        return peak

    def reset(self) -> None:
        """Reset detector state."""
        self._buffer = np.array([], dtype=np.float32)
        self.peak = 0.0
        if self._model is not None:
            self._model.reset()

    def flush(self) -> bool:
        """Process any buffered audio immediately.

        Call this periodically to prevent buffer stalls when audio stops mid-buffer.

        Returns:
            True if wake word detected
        """
        if self._model is None or len(self._buffer) == 0:
            return False

        # Process whatever we have, even if it's small
        audio_to_process = self._buffer
        self._buffer = np.array([], dtype=np.float32)

        # OpenWakeWord expects int16
        audio_int16 = (audio_to_process * 32767).astype(np.int16)

        try:
            prediction = self._model.predict(audio_int16)
        except Exception as e:
            logger.debug(f"OpenWakeWord flush error: {e}")
            return False

        for wakeword, scores in prediction.items():
            if isinstance(scores, (list, np.ndarray)):
                score = scores[-1] if len(scores) > 0 else 0
            else:
                score = scores

            if score >= self.threshold:
                self.last_score = float(score)
                logger.info(f"Wake word detected (flush): {wakeword} (score: {score:.2f})")
                if self._on_detected:
                    self._on_detected()
                return True

        return False

