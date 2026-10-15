"""Custom streaming sink for real-time voice processing.

This sink receives Discord audio (48kHz stereo) and processes it in real-time
for wake word detection and VAD, rather than waiting for recording to stop.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import discord
import numpy as np

from ..audio import resample_48k_stereo_to_16k_mono, SAMPLE_RATE_PROCESSING
from ..config import get_settings

# Phase 3: Per-user audio management (feature flagged)
from ..sink import PerUserAudioManager, WakeWordProcessor

if TYPE_CHECKING:
    from ..detection import SpeakingDetector, SpeakingState, WakeWordDetector

logger = logging.getLogger(__name__)

# Captured speech shorter than this (a cough, a stray "mm") doesn't count as carrying on
CARRY_ON_MIN_SPEECH_SECONDS = 0.4
# Discord packets are 20 ms: about 60 s of capture at most
_CAPTURE_MAX_CHUNKS = 3000


class StreamingSink(discord.sinks.Sink):
    """Sink that processes audio in real-time for voice assistant pipeline.

    This sink:
    - Receives 48kHz stereo PCM from Discord
    - Resamples to 16kHz mono for processing
    - Feeds audio to wake word detector and VAD
    - Buffers audio when in listening state
    - Emits callbacks for wake word and speech end events
    """

    # py-cord 2.8's receive path (needed for Discord's DAVE encryption) expects these from a
    # sink, but its own discord.sinks.Sink doesn't define them yet. No event listeners or
    # child sinks; we want decoded PCM, not opus.
    __sink_listeners__: tuple = ()

    def walk_children(self, with_self: bool = False):
        return [self] if with_self else []

    def is_opus(self) -> bool:
        return False

    def __init__(
        self,
        *,
        wake_model_name: str = "hey_jarvis",
        wake_threshold: float = 0.5,
        wake_sample_rate: int = 16000,
        vad=None,
        speaking_detector: Optional["SpeakingDetector"] = None,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[np.ndarray, Optional[int]], None]] = None,
        on_speaking_change: Optional[Callable[["SpeakingState"], None]] = None,
        max_speech_duration: float = 30.0,
        speaker_isolation_enabled: bool = True,
        filters=None,
        on_wake_clip: Optional[Callable[[int, np.ndarray, float, bool], None]] = None,
    ):
        """Initialize streaming sink.

        Args:
            wake_model_name: Wake word model name (e.g., "hey_jarvis")
            wake_threshold: Detection confidence threshold
            wake_sample_rate: Audio sample rate for wake detection
            vad: VAD instance
            speaking_detector: Optional SpeakingDetector for multi-user speech detection
            on_wake_word: Callback when wake word detected
            on_speech_end: Callback when speech ends, receives (audio buffer, user_id)
            on_speaking_change: Callback when channel speaking state changes
            max_speech_duration: Max seconds of speech before forced end (default 30s)
            speaker_isolation_enabled: Filter audio from non-wake-word speakers (default True)
            filters: Optional pycord audio filters
            on_wake_clip: Gets (user_id, last ~1.5s of audio, score, hit) for wake word
                hits and near misses (audio thread; the monitor saves them)
        """
        super().__init__(filters=filters)

        # Wake detection config (used by PerUserAudioManager)
        self._wake_model_name = wake_model_name
        self._wake_threshold = wake_threshold
        self._wake_sample_rate = wake_sample_rate

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
        # Who start_listening() locked onto, and whether other users' wake words still
        # count (follow-up windows, where the locked speaker didn't say the wake word)
        self._listen_user_id: Optional[int] = None
        self._watch_wake_word = False
        self._user_log_cooldown = 5.0  # Only log "receiving from user" after 5s of silence from them

        # Timing stats
        self._write_count = 0
        self._last_stats_time = time.perf_counter()
        self._total_audio_bytes = 0

        # Push-to-talk detection: track last audio received
        self._last_audio_time: Optional[float] = None
        # Seconds of no audio = speech end (PTT released, or Discord's voice gate closed)
        self._silence_timeout = get_settings().min_silence_ms / 1000
        self._audio_timeout_detected = False  # Once detected, commit to it

        # Track if we've already flushed wake detector for current gap
        self._idle_flush_done = False

        # Speaker isolation: track filtered users for logging
        self._filtered_users: set[int] = set()

        # Multi-speaker tolerance: delay speech end when others are speaking
        # This prevents premature truncation when Discord packet timing is unreliable
        self._pending_speech_end_time: Optional[float] = None
        self._multi_speaker_silence_extension_ms = 500  # Extra silence required when others speaking

        # Carry-on capture: while the session works on a turn (sink idle), keep the
        # speaker's audio so whatever they add isn't lost. start_listening() for the same
        # speaker picks it up; VoiceSession also checks it to start a turn over.
        self._capture_lock = threading.Lock()
        self._capture_user_id: Optional[int] = None
        self._capture_fold = True  # start_listening() for this speaker starts from it
        self._capture_buffer: list[np.ndarray] = []
        self._capture_quiet_buffer: list[np.ndarray] = []  # only while the bot was silent
        self._capture_voiced = 0.0  # seconds of it that were speech
        self._capture_voiced_quiet = 0.0  # ...said while the bot wasn't playing anything
        self._capture_last_audio: Optional[float] = None
        self._capture_last_voiced: Optional[float] = None
        # Listening that started from captured speech: ends once the speaker has been
        # quiet for _silence_timeout even if Discord keeps sending (VAD never restarted)
        self._seeded = False
        self._last_voiced_time: Optional[float] = None

        # Wake word refractory period: after detection, suppress further detections
        # This prevents the same "hey jarvis" audio from triggering multiple times
        self._wake_word_refractory_until: Optional[float] = None
        self._wake_word_refractory_seconds = 5.0  # Suppress for 5 seconds after detection

        # Per-user audio management (handles wake detection, pre-buffers, etc.)
        self._settings = get_settings()
        self._user_manager = PerUserAudioManager(
            wake_model_name=wake_model_name,
            wake_threshold=wake_threshold,
            wake_sample_rate=wake_sample_rate,
        )
        self._wake_processor = WakeWordProcessor(self._user_manager)
        self._wake_processor.on_clip = on_wake_clip

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
                self._wake_processor.on_idle_transition()

    @property
    def last_audio_time(self) -> Optional[float]:
        """perf_counter time of the last audio packet tracked (end of the question)."""
        return self._last_audio_time

    @property
    def wake_user_id(self) -> Optional[int]:
        """Get the user who triggered the wake word, falling back to last audio sender.

        Use this for speaker attribution instead of _last_user_id, which can be
        overwritten by other users' audio packets after wake word detection.
        """
        return self._wake_processor.wake_word_user_id or self._last_user_id

    def remove_user(self, user_id: int) -> None:
        """Clean up when user leaves voice channel.

        Args:
            user_id: Discord user ID
        """
        self._user_manager.cleanup_user(user_id)
        logger.debug(f"[WAKE] Removed user {user_id}")

    def start_listening(
        self,
        user_id: Optional[int] = None,
        watch_wake_word: bool = False,
        prefix: Optional[np.ndarray] = None,
    ):
        """Transition to listening state.

        Args:
            user_id: Speaker to listen to (follow-ups: the person the bot answered).
                Default: the wake word speaker.
            watch_wake_word: Keep detecting the wake word from everyone else; a hit
                calls on_wake_word (lets someone else cut a follow-up window short)
            prefix: Audio that goes first in the buffer (a turn being started over)
        """
        logger.info("[SINK] start_listening() called")

        # Use the user who triggered wake word (captured at detection time, not now)
        # This prevents race conditions where another user's packet arrives between detection and start_listening
        wake_user = user_id or self._wake_processor.wake_word_user_id or self._last_user_id

        # What the speaker said while the bot was busy (carry-on capture) comes first;
        # otherwise ONLY the wake word speaker's pre-buffer (speaker isolation), which
        # captures the start of questions when the user speaks continuously
        captured, captured_last, captured_voiced, captured_last_voiced = (
            self._end_capture(wake_user)
        )
        seeded = prefix is not None or captured_voiced >= CARRY_ON_MIN_SPEECH_SECONDS
        state = self._user_manager.get(wake_user) if wake_user else None
        if seeded:
            parts = [a for a in (prefix, captured) if a is not None and len(a) > 0]
            logger.info(
                f"[SINK] Starting from {sum(len(a) for a in parts) / SAMPLE_RATE_PROCESSING:.2f}s "
                f"already said by user {wake_user} ({captured_voiced:.1f}s while the bot was busy)"
            )
            self._audio_buffer = parts
        elif state:
            pre_buffer = state.get_pre_buffer()
            pre_duration = len(pre_buffer) / SAMPLE_RATE_PROCESSING
            logger.info(f"[SINK] Including {pre_duration:.2f}s pre-buffer from user {wake_user}")
            self._audio_buffer = [pre_buffer] if len(pre_buffer) > 0 else []
        else:
            self._audio_buffer = []
        # Clear all pre-buffers
        self._user_manager.clear_all_pre_buffers()

        # Lock speaker isolation to the wake word speaker
        self._last_user_id = wake_user
        self._listen_user_id = wake_user
        self._watch_wake_word = watch_wake_word

        # Reset filtered user tracking
        self._filtered_users.clear()

        self._speech_started = seeded
        self._seeded = seeded
        self._pending_speech_end_time = None  # Reset multi-speaker tolerance
        # PTT tracking: seeded listening counts from the speaker's last audio
        now = time.perf_counter()
        self._last_audio_time = (captured_last or now) if seeded else None
        self._last_voiced_time = (captured_last_voiced or now) if seeded else None
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
        self._listen_user_id = None
        self._watch_wake_word = False
        self.state = "idle"

    def get_audio_buffer(self) -> np.ndarray:
        """Get accumulated audio buffer.

        Returns:
            Concatenated audio as float32 numpy array (16kHz mono)
        """
        if not self._audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._audio_buffer)

    def start_capture(self, user_id: Optional[int], fold: bool = True) -> None:
        """Keep this speaker's audio while the sink is idle (see _capture_user_id).

        fold: start_listening() for this speaker begins with it. False only measures
        (how much they said), so a later wake word turn doesn't start with old chatter.
        """
        with self._capture_lock:
            self._capture_user_id = user_id
            self._capture_fold = fold
            self._capture_buffer = []
            self._capture_quiet_buffer = []
            self._capture_voiced = 0.0
            self._capture_voiced_quiet = 0.0
            self._capture_last_audio = None
            self._capture_last_voiced = None

    def stop_capture(self, only_fold: bool = False) -> None:
        """Stop capturing (only_fold: leave a measuring-only capture running)."""
        with self._capture_lock:
            if only_fold and not self._capture_fold:
                return
        self.start_capture(None)

    def is_capturing(self, user_id: Optional[int], fold: bool) -> bool:
        """This capture is still running (nothing has started listening since)."""
        with self._capture_lock:
            return self._capture_user_id == user_id and self._capture_fold == fold

    def captured_speech(self, while_bot_quiet: bool = False) -> float:
        """Seconds of speech captured so far (optionally only while the bot was silent,
        which leaves out the bot's own voice echoing through someone's mic)."""
        with self._capture_lock:
            return self._capture_voiced_quiet if while_bot_quiet else self._capture_voiced

    def captured_audio(self, while_bot_quiet: bool = False) -> np.ndarray:
        """The audio captured so far (optionally only what came while the bot was silent),
        for checking with a real VAD that it's speech and not clicks or game sounds."""
        with self._capture_lock:
            chunks = self._capture_quiet_buffer if while_bot_quiet else self._capture_buffer
            return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    def _end_capture(
        self, user_id: Optional[int]
    ) -> tuple[Optional[np.ndarray], Optional[float], float, Optional[float]]:
        """Stop capturing. Returns (audio, last audio time, seconds of speech, last speech
        time) if the capture was for user_id, else (None, None, 0, None)."""
        with self._capture_lock:
            mine = (user_id is not None and self._capture_user_id == user_id
                    and self._capture_fold)
            audio = np.concatenate(self._capture_buffer) if mine and self._capture_buffer else None
            result = (audio, self._capture_last_audio, self._capture_voiced,
                      self._capture_last_voiced) if mine else (None, None, 0.0, None)
        self.stop_capture()
        return result

    def _capture(self, user_id: int, audio_16k: np.ndarray, now: float) -> None:
        if user_id != self._capture_user_id:
            return
        voiced = self._user_is_speaking(user_id, audio_16k)
        seconds = len(audio_16k) / SAMPLE_RATE_PROCESSING
        bot_playing = self._bot_is_playing()
        with self._capture_lock:
            if user_id != self._capture_user_id:
                return
            self._capture_buffer.append(audio_16k.copy())
            if len(self._capture_buffer) > _CAPTURE_MAX_CHUNKS:
                self._capture_buffer.pop(0)  # nobody took it; keep the newest only
            if not bot_playing:
                self._capture_quiet_buffer.append(self._capture_buffer[-1])
                if len(self._capture_quiet_buffer) > _CAPTURE_MAX_CHUNKS:
                    self._capture_quiet_buffer.pop(0)
            self._capture_last_audio = now
            if voiced:
                self._capture_voiced += seconds
                self._capture_last_voiced = now
                if not bot_playing:
                    self._capture_voiced_quiet += seconds

    def _user_is_speaking(self, user_id: int, audio_16k: np.ndarray) -> bool:
        """This packet is speech (speaking detector's per-user state, else energy)."""
        if self.speaking_detector:
            user = self.speaking_detector.get_user_state(user_id)
            return bool(user and user.is_speaking)
        return float(np.sqrt(np.mean(audio_16k**2))) >= 0.01

    def _bot_is_playing(self) -> bool:
        vc = getattr(self, "vc", None)
        try:
            return bool(vc and vc.is_playing())
        except Exception:
            return False

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
        if self._seeded and self._last_voiced_time is not None:
            # Started from captured speech: Discord may keep sending (noise) with no
            # VAD restart, so quiet from the speaker also counts
            silence_duration = max(silence_duration, time.perf_counter() - self._last_voiced_time)
        if silence_duration >= self._silence_timeout:
            buffer_samples = sum(len(a) for a in self._audio_buffer)
            buffer_duration = buffer_samples / SAMPLE_RATE_PROCESSING
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

        # Check refractory period - skip detection if recently triggered
        in_refractory = (
            self._wake_word_refractory_until is not None
            and time.perf_counter() < self._wake_word_refractory_until
        )
        if in_refractory:
            return False

        # Check for wake word in audio gap using WakeWordProcessor
        user_id = self._wake_processor.check_idle_audio_gap()
        if user_id is not None:
            self._wake_word_refractory_until = time.perf_counter() + self._wake_word_refractory_seconds
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
            data: Raw PCM audio bytes (48kHz stereo s16le), or py-cord 2.8's VoiceData
                (PCM in .pcm)
            user: Discord user who sent the audio
        """
        if not isinstance(data, (bytes, bytearray)) and hasattr(data, "pcm"):
            data = data.pcm
            if user is None:
                return  # py-cord doesn't know whose audio this is yet (no SSRC mapping)
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
        # Save the pre-gap user before updating (needed for wake word attribution from flush)
        pre_gap_user_id = self._last_user_id

        if self._state == "idle":
            # Only log if we haven't heard from this user recently (prevents spam with multiple users)
            user_state = self._user_manager.get_or_create(user_id)
            last_seen = user_state.last_seen
            if now - last_seen >= self._user_log_cooldown:
                logger.info(f"[SINK] Receiving audio from user: {user_id}")
            self._last_user_id = user_id

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

                # Notify wake processor of audio gap
                self._wake_processor.on_audio_gap()
                # Clear all pre-buffers on large gaps (stale audio from before PTT)
                if gap > 1.0:
                    logger.info(f"[SINK] Clearing pre-buffers due to {gap:.1f}s audio gap")
                    self._user_manager.clear_all_pre_buffers()

                # Handle per-user wake detectors based on gap duration
                # Short gaps (< 2.5s): flush to complete pending detections
                # Long gaps (>= 2.5s): reset to clear stale neural network state (prevents false positives)
                MAX_FLUSH_GAP = 2.5  # Match wake_processor.MAX_GAP_SECONDS

                # Check refractory period for flush detection
                in_refractory = (
                    self._wake_word_refractory_until is not None
                    and time.perf_counter() < self._wake_word_refractory_until
                )

                if gap >= MAX_FLUSH_GAP:
                    # Long gap - reset detectors (stale state causes false positives)
                    logger.info(f"[SINK] Resetting wake detectors due to {gap:.1f}s gap (> {MAX_FLUSH_GAP}s)")
                    self._user_manager.reset_all_detectors()
                elif not in_refractory:
                    # Short gap - flush to complete pending detections (skip if in refractory)
                    for state in self._user_manager:
                        detector = state.detector
                        if detector and hasattr(detector, 'flush') and detector.flush():
                            # Wake word was in pre-gap audio - attribute to that detector's user
                            attributed_user = pre_gap_user_id if state.user_id == pre_gap_user_id else state.user_id
                            self._wake_processor._wake_word_user_id = attributed_user
                            self._wake_word_refractory_until = time.perf_counter() + self._wake_word_refractory_seconds
                            logger.info(f"[SINK] Wake word detected from flushed buffer! (user: {attributed_user})")
                            self._wake_processor.report_clip(state, hit=True)
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
        # Silence attempts are handled internally by WakeWordProcessor.process_audio

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
            # Check refractory period - skip wake word detection if recently triggered
            in_refractory = (
                self._wake_word_refractory_until is not None
                and time.perf_counter() < self._wake_word_refractory_until
            )

            # Get or create user state and add to pre-buffer
            state = self._user_manager.get_or_create(user_id)
            state.add_to_pre_buffer(audio_16k)
            state.update_last_seen()
            self._capture(user_id, audio_16k, now)

            # Check for wake word using wake processor (skip if in refractory)
            if not in_refractory and self._wake_processor.process_audio(user_id, audio_16k):
                # Wake word detected! Set refractory period
                self._wake_word_refractory_until = time.perf_counter() + self._wake_word_refractory_seconds
                if self._last_audio_time:
                    lag = time.perf_counter() - self._last_audio_time
                    logger.info(f"[SINK] Wake word detected! (packet lag: {lag*1000:.0f}ms, user: {user_id})")
                else:
                    logger.info(f"[SINK] Wake word detected! (user: {user_id})")
                if self._on_wake_word:
                    self._on_wake_word()

        elif self._state == "listening":
            # SPEAKER ISOLATION: Only buffer audio from the wake word speaker
            # Use wake_word_user_id (captured at detection time) to avoid race conditions
            # where packets from other users arrive between detection and start_listening()
            # (follow-ups: the wake word user is cleared by then, so use who we locked onto)
            wake_user = self._listen_user_id or self._wake_processor.wake_word_user_id
            if self._speaker_isolation_enabled and wake_user is not None and user_id != wake_user:
                # Track interference for debugging (log once per user per session)
                if user_id not in self._filtered_users:
                    logger.info(f"[SINK ISOLATION] Filtering audio from user {user_id} (locked to {wake_user})")
                    self._filtered_users.add(user_id)
                if self._watch_wake_word:
                    self._watch_for_wake_word(user_id, audio_16k)
                # Don't buffer this audio - PTT timing already handled above
                # Still process through speaking detector for awareness
                return

            # Buffer audio and check VAD
            # NOTE: Skip logic removed - pre-buffer handles wake word audio
            self._audio_buffer.append(audio_16k.copy())
            if self._seeded and self._user_is_speaking(user_id, audio_16k):
                self._last_voiced_time = now
            buffer_samples = sum(len(a) for a in self._audio_buffer)

            if self.vad:
                endpoint = self.vad.detect_endpoint(audio_16k)

                # Log VAD state periodically (every ~1 second of audio = 16000 samples)
                if buffer_samples % SAMPLE_RATE_PROCESSING < len(audio_16k):
                    buffer_sec = buffer_samples / SAMPLE_RATE_PROCESSING
                    logger.info(
                        f"[SINK VAD] buffer={buffer_sec:.1f}s, "
                        f"speech_started={self._speech_started}, "
                        f"vad.is_speech={self.vad.is_speech}"
                    )

                if endpoint == "start" and self._seeded:
                    self._seeded = False  # VAD follows the speech from here
                    self._pending_speech_end_time = None
                    logger.info("[SINK] VAD detected speech START (continuing)")
                elif endpoint == "start" and not self._speech_started:
                    self._speech_started = True
                    self._pending_speech_end_time = None  # Cancel any pending end
                    logger.info("[SINK] VAD detected speech START")

                elif endpoint == "end" and self._speech_started:
                    buffer_duration = buffer_samples / SAMPLE_RATE_PROCESSING
                    others_speaking = self.speaking_detector and self.speaking_detector.is_anyone_actively_speaking()

                    if others_speaking:
                        # Multi-speaker scenario: require extra silence before committing
                        if self._pending_speech_end_time is None:
                            self._pending_speech_end_time = time.perf_counter()
                            logger.info(
                                f"[SINK] VAD detected end but others speaking - "
                                f"waiting {self._multi_speaker_silence_extension_ms}ms to confirm"
                            )
                        elif (time.perf_counter() - self._pending_speech_end_time) * 1000 >= self._multi_speaker_silence_extension_ms:
                            # Confirmed: silence persisted despite others speaking
                            logger.info(f"[SPEECH_END] Multi-speaker silence confirmed, buffer={buffer_duration:.2f}s")
                            self._speech_started = False
                            self._pending_speech_end_time = None
                            if self._on_speech_end:
                                audio = self.get_audio_buffer()
                                self._on_speech_end(audio, self._last_user_id)
                            else:
                                logger.warning("[SINK] No _on_speech_end callback set!")
                            self._audio_buffer = []
                        # else: still waiting for confirmation
                    else:
                        # Single speaker: end immediately (original behavior)
                        logger.info(f"[SPEECH_END] VAD silence threshold reached, buffer={buffer_duration:.2f}s")
                        self._speech_started = False
                        self._pending_speech_end_time = None
                        if self._on_speech_end:
                            audio = self.get_audio_buffer()
                            self._on_speech_end(audio, self._last_user_id)
                        else:
                            logger.warning("[SINK] No _on_speech_end callback set!")
                        self._audio_buffer = []

                elif self._speech_started and self._pending_speech_end_time is not None:
                    # We were waiting for confirmation but VAD didn't say "end" this frame
                    # This means VAD detected activity - cancel the pending end
                    logger.info("[SINK] Pending speech end cancelled - VAD detected activity")
                    self._pending_speech_end_time = None

                # Force end if speech has been going too long (background noise protection)
                if self._speech_started:
                    speech_duration = buffer_samples / SAMPLE_RATE_PROCESSING
                    if speech_duration >= self._max_speech_duration:
                        logger.warning(
                            f"[SPEECH_END] Max duration ({self._max_speech_duration}s) reached, "
                            f"buffer={speech_duration:.1f}s - forcing end"
                        )
                        # Reset state BEFORE callback to prevent spam
                        self._speech_started = False
                        self._pending_speech_end_time = None
                        if self._on_speech_end:
                            audio = self.get_audio_buffer()
                            self._on_speech_end(audio, self._last_user_id)
                        # Clear buffer after forced end
                        self._audio_buffer = []

    def _watch_for_wake_word(self, user_id: int, audio_16k: np.ndarray) -> None:
        """Follow-up window: someone other than the locked speaker says the wake word."""
        state = self._user_manager.get_or_create(user_id)
        state.add_to_pre_buffer(audio_16k)  # so their question's start isn't lost
        state.update_last_seen()
        if self._wake_processor.process_audio(user_id, audio_16k):
            self._watch_wake_word = False
            self._wake_word_refractory_until = time.perf_counter() + self._wake_word_refractory_seconds
            logger.info(f"[SINK] Wake word from user {user_id} during a follow-up window")
            if self._on_wake_word:
                self._on_wake_word()

    def cleanup(self) -> None:
        """Clean up resources when recording stops."""
        logger.debug("StreamingSink cleanup")
        self._audio_buffer = []
        self._state = "idle"
        if self.speaking_detector:
            self.speaking_detector.reset()
        self._user_manager.cleanup_all()
