"""Custom streaming sink for real-time voice processing.

This sink receives Discord audio (48kHz stereo) and processes it in real-time
for wake word detection and VAD, rather than waiting for recording to stop.
"""

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

import discord
import numpy as np

from ..audio import resample_48k_stereo_to_16k_mono

if TYPE_CHECKING:
    from ..detection import SpeakingDetector, SpeakingState

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
        speaking_detector: Optional["SpeakingDetector"] = None,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[np.ndarray, Optional[int]], None]] = None,
        on_speaking_change: Optional[Callable[["SpeakingState"], None]] = None,
        max_speech_duration: float = 30.0,
        speaker_isolation_enabled: bool = True,
        filters=None,
    ):
        """Initialize streaming sink.

        Args:
            wake_detector: WakeWordDetector instance
            vad: VAD instance
            speaking_detector: Optional SpeakingDetector for multi-user speech detection
            on_wake_word: Callback when wake word detected
            on_speech_end: Callback when speech ends, receives (audio buffer, user_id)
            on_speaking_change: Callback when channel speaking state changes
            max_speech_duration: Max seconds of speech before forced end (default 30s)
            speaker_isolation_enabled: Filter audio from non-wake-word speakers (default True)
            filters: Optional pycord audio filters
        """
        super().__init__(filters=filters)

        self.wake_detector = wake_detector
        self.vad = vad
        self.speaking_detector = speaking_detector
        self._on_wake_word = on_wake_word
        self._on_speech_end = on_speech_end
        self._on_speaking_change = on_speaking_change
        self._last_speaking_state: Optional["SpeakingState"] = None
        self._speaker_isolation_enabled = speaker_isolation_enabled

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

        # Track if we've already flushed wake detector for current gap
        self._idle_flush_done = False
        self._silence_attempts = 0

        # Per-user pre-buffers: capture audio before wake word for question context
        # This captures the start of questions like "what's the weather" when
        # user says "hey jarvis what's the weather" without pausing
        # Each user gets their own buffer so we only use the wake word speaker's audio
        self._per_user_pre_buffers: dict[int, list[np.ndarray]] = {}
        self._pre_buffer_max_samples = int(16000 * 1.5)  # 1.5s at 16kHz

        # Speaker isolation: track filtered users for logging
        self._filtered_users: set[int] = set()

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

        # Include ONLY the wake word speaker's pre-buffer (speaker isolation)
        # This captures the start of questions when user speaks continuously
        if self._last_user_id and self._last_user_id in self._per_user_pre_buffers:
            pre_buffer = self._per_user_pre_buffers[self._last_user_id]
            pre_duration = sum(len(a) for a in pre_buffer) / 16000
            logger.info(f"[SINK] Including {pre_duration:.2f}s pre-buffer from user {self._last_user_id}")
            self._audio_buffer = pre_buffer.copy()
        else:
            self._audio_buffer = []

        # Clear all pre-buffers and reset filtered user tracking
        self._per_user_pre_buffers.clear()
        self._filtered_users.clear()

        self._speech_started = False
        self._last_audio_time = None  # Reset PTT tracking
        self._audio_timeout_detected = False  # Reset timeout flag

        # NOTE: Removed skip logic - pre-buffer replaces it
        # The wake word audio is in the pre-buffer but LLM handles it gracefully

        if self.vad:
            self.vad.reset()
        self._state = "listening"

    def stop_listening(self):
        """Transition to idle state."""
        logger.info("[SINK] stop_listening() called")
        # Log speaker isolation stats if any users were filtered
        if self._filtered_users:
            logger.info(f"[SINK ISOLATION] Session filtered audio from {len(self._filtered_users)} other user(s): {self._filtered_users}")
            self._filtered_users.clear()
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

    def is_speech_active(self) -> bool:
        """Check if VAD has detected active speech.

        Used by session to extend timeout while user is speaking.
        """
        return self._speech_started and self._state == "listening"

    def is_anyone_speaking(self) -> bool:
        """Check if anyone in the channel is speaking (with 3s debounce).

        Uses the speaking detector if available, otherwise returns False.
        The 3-second debounce means this returns True for 3s after speech ends.
        Use this for deciding whether to hold off on bot responses.
        """
        if self.speaking_detector:
            from ..detection import SpeakingState

            return self.speaking_detector.get_state() == SpeakingState.SOMEONE_SPEAKING
        return False

    def is_anyone_actively_speaking(self) -> bool:
        """Check if anyone is actively speaking RIGHT NOW (no debounce).

        Returns True only while speech is detected, False immediately when speech stops.
        Use this for quick checks like acknowledgment timing.
        """
        if self.speaking_detector:
            return self.speaking_detector.is_anyone_actively_speaking()
        return False

    def check_idle_audio_gap(self) -> bool:
        """Check if audio has stopped while idle and complete wake word detection.

        This handles the case where user says wake word and stops talking
        before more audio arrives to trigger detection. OpenWakeWord is a
        streaming model - it may have the wake word "loaded" in its internal
        state but needs more audio to complete the detection.

        Returns:
            True if wake word was detected, False otherwise
        """
        if self._state != "idle":
            return False

        if self._last_audio_time is None:
            return False

        gap = time.perf_counter() - self._last_audio_time

        # Poll speaking detector during gaps to trigger staleness check
        # This ensures state changes to SILENCE even when no audio arrives
        if gap >= 0.5 and self.speaking_detector:
            state = self.speaking_detector.get_state()
            if state != self._last_speaking_state:
                self._last_speaking_state = state
                if self._on_speaking_change:
                    self._on_speaking_change(state)

        # Only check if there's been a significant gap (250ms+)
        # Allow up to 5 attempts (2.5s of silence total) before giving up
        if gap >= 0.25 and self.wake_detector:
            if self._silence_attempts >= 5:
                return False  # Give up after 5 attempts

            self._silence_attempts += 1
            logger.debug(f"[SINK] Feeding silence attempt {self._silence_attempts}/5 (gap={gap:.2f}s)")

            # Feed silence to the wake detector to complete any pending detection.
            # OpenWakeWord's internal state may have the wake word loaded but needs
            # more audio frames to trigger the final detection.
            # Use 500ms of silence (increased from 100ms) to ensure detection completes.
            silence = np.zeros(8000, dtype=np.float32)  # 500ms of silence at 16kHz
            detected = self.wake_detector.process(silence)
            if detected:
                logger.info(f"[SINK] Wake word detected from silence padding ({gap:.2f}s gap)")
                return True

            # Also try flush in case there's buffered audio
            if hasattr(self.wake_detector, 'flush') and self.wake_detector.flush():
                logger.info(f"[SINK] Wake word detected from flush ({gap:.2f}s gap)")
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
        gap_since_last = (now - self._last_audio_time) if self._last_audio_time else 0
        if self._last_audio_time is not None:
            gap = now - self._last_audio_time
            # Log if there was a significant gap in audio (Discord wasn't sending)
            if gap > 0.2 and self._state == "idle":
                logger.info(f"[SINK] Audio resumed after {gap:.1f}s gap")
                # Reset idle flush flag since new audio arrived
                self._idle_flush_done = False
                self._silence_attempts = 0
                # Clear all pre-buffers on large gaps (stale audio from before PTT)
                if gap > 1.0:
                    self._per_user_pre_buffers.clear()
                # Flush wake word detector's buffer - it may have stale audio from before the gap
                if self.wake_detector and hasattr(self.wake_detector, 'flush'):
                    if self.wake_detector.flush():
                        logger.info("[SINK] Wake word detected from flushed buffer!")
                        if self._on_wake_word:
                            self._on_wake_word()
                            return  # Don't process current audio if wake word found
        # Update last audio time:
        # - In idle state: track all audio (for gap detection)
        # - In listening state with isolation: only track locked user (for PTT timeout)
        if self._state == "idle":
            self._last_audio_time = now
        elif not self._speaker_isolation_enabled or self._last_user_id is None or user_id == self._last_user_id:
            self._last_audio_time = now
        # else: listening with isolation and wrong user - don't update (will trigger PTT timeout)

        # Reset idle flush flag when audio arrives (allows periodic flush after next gap)
        self._idle_flush_done = False
        self._silence_attempts = 0

        # Convert to 16kHz mono float32
        audio_16k = resample_48k_stereo_to_16k_mono(data)

        if len(audio_16k) == 0:
            return

        # Process through speaking detector (tracks all users' speech state)
        if self.speaking_detector:
            state = self.speaking_detector.process_audio(audio_16k, user_id)
            if state != self._last_speaking_state:
                self._last_speaking_state = state
                if self._on_speaking_change:
                    self._on_speaking_change(state)

        # === DIAGNOSTIC LOGGING (both states) ===
        chunk_size = len(data) if data else 0
        gap_ms = gap_since_last * 1000
        energy = np.sqrt(np.mean(audio_16k**2)) if len(audio_16k) > 0 else 0
        logger.debug(
            f"[DIAG] state={self._state}, gap={gap_ms:.0f}ms, chunk={chunk_size}B, "
            f"energy={energy:.4f}, samples={len(audio_16k)}"
        )
        # === END DIAGNOSTIC ===

        if self._state == "idle":
            # Maintain per-user rolling pre-buffer (captures question start before wake word)
            # Each user gets their own buffer so we only use the wake word speaker's audio
            if user_id not in self._per_user_pre_buffers:
                self._per_user_pre_buffers[user_id] = []

            self._per_user_pre_buffers[user_id].append(audio_16k.copy())

            # Trim this user's buffer to max duration
            user_buffer = self._per_user_pre_buffers[user_id]
            total = sum(len(a) for a in user_buffer)
            while total > self._pre_buffer_max_samples and user_buffer:
                removed = user_buffer.pop(0)
                total -= len(removed)

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
            # SPEAKER ISOLATION: Only buffer audio from the wake word speaker
            if self._speaker_isolation_enabled and self._last_user_id is not None and user_id != self._last_user_id:
                # Track interference for debugging (log once per user per session)
                if user_id not in self._filtered_users:
                    logger.info(f"[SINK ISOLATION] Filtering audio from user {user_id} (locked to {self._last_user_id})")
                    self._filtered_users.add(user_id)
                # Don't buffer this audio - PTT timing already handled above
                # Still process through speaking detector for awareness
                return

            # Buffer audio and check VAD
            # NOTE: Skip logic removed - pre-buffer handles wake word audio
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
        if self.speaking_detector:
            self.speaking_detector.reset()
