"""Custom streaming sink for real-time voice processing.

This sink receives Discord audio (48kHz stereo) and processes it in real-time
for wake word detection and VAD, rather than waiting for recording to stop.
"""

import logging
import time
from typing import Callable, Optional

import discord
import numpy as np

from ..audio import resample_48k_stereo_to_16k_mono

logger = logging.getLogger(__name__)


class StreamingSink(discord.sinks.Sink):
    """Sink that processes audio in real-time for voice assistant pipeline.

    This sink:
    - Receives 48kHz stereo PCM from Discord
    - Resamples to 16kHz mono for processing
    - Feeds audio to wake word detector and VAD
    - Buffers audio when in listening state
    - Emits callbacks for wake word and speech end events
    """

    def __init__(
        self,
        *,
        wake_detector=None,
        vad=None,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[np.ndarray, Optional[int]], None]] = None,
        max_speech_duration: float = 30.0,
        filters=None,
    ):
        """Initialize streaming sink.

        Args:
            wake_detector: WakeWordDetector instance
            vad: VAD instance
            on_wake_word: Callback when wake word detected
            on_speech_end: Callback when speech ends, receives (audio buffer, user_id)
            max_speech_duration: Max seconds of speech before forced end (default 30s)
            filters: Optional pycord audio filters
        """
        super().__init__(filters=filters)

        self.wake_detector = wake_detector
        self.vad = vad
        self._on_wake_word = on_wake_word
        self._on_speech_end = on_speech_end

        # State
        self._state = "idle"  # idle, listening
        self._audio_buffer: list[np.ndarray] = []
        self._speech_started = False
        self._max_speech_duration = max_speech_duration

        # Track user for logging
        self._last_user_id = None

        # Timing stats
        self._write_count = 0
        self._last_stats_time = time.perf_counter()
        self._total_audio_bytes = 0

        # Push-to-talk detection: track last audio received
        self._last_audio_time: Optional[float] = None
        self._silence_timeout = 0.5  # Seconds of no audio = speech end (for PTT)
        self._audio_timeout_detected = False  # Once detected, commit to it

        # Skip initial audio after wake word to avoid capturing "jarvis" tail
        self._listen_start_time: Optional[float] = None
        self._skip_initial_audio = False
        self._initial_skip_duration = 0.3  # Skip first 300ms

        # Track if we've already flushed wake detector for current gap
        self._idle_flush_done = False

    @property
    def state(self) -> str:
        """Current state of the sink."""
        return self._state

    @state.setter
    def state(self, value: str):
        """Set state and handle transitions."""
        if value not in ("idle", "listening"):
            raise ValueError(f"Invalid state: {value}")

        old_state = self._state
        self._state = value

        if old_state != value:
            logger.debug(f"Sink state: {old_state} -> {value}")

            if value == "listening":
                # Reset for new listening session
                self._audio_buffer = []
                self._speech_started = False
                if self.vad:
                    self.vad.reset()

            elif value == "idle":
                # Clear buffer when going idle
                self._audio_buffer = []
                if self.wake_detector:
                    self.wake_detector.reset()

    def start_listening(self):
        """Transition to listening state."""
        logger.info("[SINK] start_listening() called")
        # Reset all listening state for fresh start
        self._audio_buffer = []
        self._speech_started = False
        self._last_audio_time = None  # Reset PTT tracking
        self._audio_timeout_detected = False  # Reset timeout flag
        # Skip first ~300ms of audio to avoid capturing wake word tail
        self._listen_start_time = time.perf_counter()
        self._skip_initial_audio = True
        if self.vad:
            self.vad.reset()
        self._state = "listening"

    def stop_listening(self):
        """Transition to idle state."""
        logger.info("[SINK] stop_listening() called")
        self.state = "idle"

    def get_audio_buffer(self) -> np.ndarray:
        """Get accumulated audio buffer.

        Returns:
            Concatenated audio as float32 numpy array (16kHz mono)
        """
        if not self._audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._audio_buffer)

    def check_audio_timeout(self) -> bool:
        """Check if audio has stopped arriving (push-to-talk released).

        Returns:
            True if audio stopped and we should end speech, False otherwise
        """
        # Once we detect timeout, keep returning True (commit to the decision)
        if self._audio_timeout_detected:
            return True

        if not self._speech_started or self._state != "listening":
            return False

        if self._last_audio_time is None:
            return False

        silence_duration = time.perf_counter() - self._last_audio_time
        if silence_duration >= self._silence_timeout:
            buffer_samples = sum(len(a) for a in self._audio_buffer)
            buffer_duration = buffer_samples / 16000
            logger.info(
                f"[SINK] Audio stopped for {silence_duration:.2f}s "
                f"(PTT released?), buffer={buffer_duration:.1f}s"
            )
            self._audio_timeout_detected = True  # Commit to this decision
            return True
        return False

    def check_idle_audio_gap(self) -> bool:
        """Check if audio has stopped while idle and flush wake word detector.

        This handles the case where user says wake word and stops talking
        before more audio arrives to trigger detection.

        Returns:
            True if wake word was detected from flush, False otherwise
        """
        if self._state != "idle":
            return False

        if self._last_audio_time is None:
            return False

        # Only flush once per gap (flag is reset when new audio arrives)
        if self._idle_flush_done:
            return False

        gap = time.perf_counter() - self._last_audio_time
        # Only flush if there's been a significant gap (similar to resume detection)
        if gap >= 0.25 and self.wake_detector and hasattr(self.wake_detector, 'flush'):
            self._idle_flush_done = True  # Mark as flushed for this gap
            if self.wake_detector.flush():
                logger.info(f"[SINK] Wake word detected from idle gap flush ({gap:.2f}s gap)")
                return True
        return False

    def init(self, vc) -> None:
        """Initialize sink with voice client reference.

        Called by pycord when recording starts.

        Args:
            vc: VoiceClient instance
        """
        super().init(vc)
        self.vc = vc
        logger.debug("StreamingSink initialized with VoiceClient")

    def write(self, data, user) -> None:
        """Process incoming audio data.

        This is called by pycord for each audio packet received.

        Args:
            data: Raw PCM audio bytes (48kHz stereo s16le)
            user: Discord user who sent the audio
        """
        # Track stats
        self._write_count += 1
        self._total_audio_bytes += len(data) if data else 0

        # Log stats every 5 seconds
        now = time.perf_counter()
        if now - self._last_stats_time >= 5.0:
            elapsed = now - self._last_stats_time
            rate = self._write_count / elapsed
            bytes_per_sec = self._total_audio_bytes / elapsed
            logger.info(
                f"[SINK STATS] {self._write_count} writes in {elapsed:.1f}s "
                f"({rate:.1f}/s, {bytes_per_sec/1024:.1f} KB/s), state={self._state}"
            )
            self._write_count = 0
            self._total_audio_bytes = 0
            self._last_stats_time = now

        # Get user id for logging (user might be an int or User object)
        # Only update user ID when idle - this locks in the wake word speaker
        # and prevents someone else's noise from stealing attribution
        user_id = user if isinstance(user, int) else getattr(user, "id", user)
        if self._state == "idle" and self._last_user_id != user_id:
            self._last_user_id = user_id
            logger.info(f"[SINK] Receiving audio from user: {user_id}")

        # Handle None or empty data
        if not data:
            return

        # Track when we last received audio (for PTT detection)
        now = time.perf_counter()
        if self._last_audio_time is not None:
            gap = now - self._last_audio_time
            # Log if there was a significant gap in audio (Discord wasn't sending)
            if gap > 0.2 and self._state == "idle":
                logger.info(f"[SINK] Audio resumed after {gap:.1f}s gap")
                # Reset idle flush flag since new audio arrived
                self._idle_flush_done = False
                # Flush wake word detector's buffer - it may have stale audio from before the gap
                if self.wake_detector and hasattr(self.wake_detector, 'flush'):
                    if self.wake_detector.flush():
                        logger.info("[SINK] Wake word detected from flushed buffer!")
                        if self._on_wake_word:
                            self._on_wake_word()
                            return  # Don't process current audio if wake word found
        self._last_audio_time = now
        # Reset idle flush flag when audio arrives (allows periodic flush after next gap)
        self._idle_flush_done = False

        # Convert to 16kHz mono float32
        audio_16k = resample_48k_stereo_to_16k_mono(data)

        if len(audio_16k) == 0:
            return

        if self._state == "idle":
            # Check for wake word
            if self.wake_detector:
                detected = self.wake_detector.process(audio_16k)
                if detected and self._on_wake_word:
                    # Log time since last audio packet (shows Discord transmission lag)
                    if self._last_audio_time:
                        lag = time.perf_counter() - self._last_audio_time
                        logger.info(f"[SINK] Wake word detected! (packet lag: {lag*1000:.0f}ms)")
                    else:
                        logger.info("[SINK] Wake word detected!")
                    self._on_wake_word()

        elif self._state == "listening":
            # Skip initial audio to avoid capturing wake word tail
            if self._skip_initial_audio:
                elapsed = time.perf_counter() - self._listen_start_time
                if elapsed < self._initial_skip_duration:
                    return  # Don't buffer this audio
                else:
                    self._skip_initial_audio = False
                    logger.debug(f"[SINK] Finished skipping initial {elapsed:.2f}s of audio")

            # Buffer audio and check VAD
            self._audio_buffer.append(audio_16k.copy())
            buffer_samples = sum(len(a) for a in self._audio_buffer)

            if self.vad:
                endpoint = self.vad.detect_endpoint(audio_16k)

                # Log VAD state periodically (every ~1 second of audio = 16000 samples)
                if buffer_samples % 16000 < len(audio_16k):
                    buffer_sec = buffer_samples / 16000
                    logger.info(
                        f"[SINK VAD] buffer={buffer_sec:.1f}s, "
                        f"speech_started={self._speech_started}, "
                        f"vad.is_speech={self.vad.is_speech}"
                    )

                if endpoint == "start" and not self._speech_started:
                    self._speech_started = True
                    logger.info("[SINK] VAD detected speech START")

                elif endpoint == "end" and self._speech_started:
                    buffer_duration = buffer_samples / 16000
                    logger.info(f"[SPEECH_END] VAD silence threshold reached, buffer={buffer_duration:.2f}s")
                    # Reset state before callback
                    self._speech_started = False
                    if self._on_speech_end:
                        audio = self.get_audio_buffer()
                        self._on_speech_end(audio, self._last_user_id)
                    else:
                        logger.warning("[SINK] No _on_speech_end callback set!")
                    # Clear buffer after speech end
                    self._audio_buffer = []

                # Force end if speech has been going too long (background noise protection)
                elif self._speech_started:
                    speech_duration = buffer_samples / 16000
                    if speech_duration >= self._max_speech_duration:
                        logger.warning(
                            f"[SPEECH_END] Max duration ({self._max_speech_duration}s) reached, "
                            f"buffer={speech_duration:.1f}s - forcing end"
                        )
                        # Reset state BEFORE callback to prevent spam
                        self._speech_started = False
                        if self._on_speech_end:
                            audio = self.get_audio_buffer()
                            self._on_speech_end(audio, self._last_user_id)
                        # Clear buffer after forced end
                        self._audio_buffer = []

    def cleanup(self) -> None:
        """Clean up resources when recording stops."""
        logger.debug("StreamingSink cleanup")
        self._audio_buffer = []
        self._state = "idle"
