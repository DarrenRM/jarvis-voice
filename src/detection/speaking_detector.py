"""Speaking detection for multi-user voice channels.

Detects when anyone in the voice channel is speaking using a hybrid approach:
1. Packet tracking - know when audio arrives per-user
2. Energy gate - fast noise rejection
3. VAD confirmation - Silero VAD for definitive speech detection
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeakingState(Enum):
    """Aggregate speaking state for the voice channel."""

    SOMEONE_SPEAKING = "speaking"
    SILENCE = "silence"


@dataclass
class UserAudioState:
    """Per-user audio tracking state."""

    user_id: int
    last_packet_time: Optional[float] = None
    last_energy: float = 0.0
    is_speaking: bool = False
    vad_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    speech_start_time: Optional[float] = None
    # Track when low-energy/no-speech started (for per-user debounce)
    silence_start_time: Optional[float] = None


class SpeakingDetector:
    """Detects when anyone in the voice channel is speaking.

    Uses hybrid detection:
    1. Packet tracking - know when audio arrives
    2. Energy gate - reject noise (RMS < threshold)
    3. VAD confirmation - verify actual speech

    Reports:
    - SOMEONE_SPEAKING: At least one user is actively speaking
    - SILENCE: No one has spoken for `silence_debounce_ms`

    Provides both aggregate and per-user interfaces:
    - get_state() -> SpeakingState (anyone speaking?)
    - get_speaking_users() -> list[int] (who is speaking?)
    - get_user_state(user_id) -> UserAudioState (single user details)
    """

    # Silero VAD requires exactly 512 samples at 16kHz
    VAD_CHUNK_SIZE = 512

    def __init__(
        self,
        *,
        energy_threshold: float = 0.01,
        silence_debounce_ms: int = 3000,
        vad_threshold: float = 0.5,
        sample_rate: int = 16000,
        excluded_user_ids: Optional[set[int]] = None,
        on_state_change: Optional[Callable[[SpeakingState], None]] = None,
    ):
        """Initialize speaking detector.

        Args:
            energy_threshold: RMS threshold below which audio is considered noise
            silence_debounce_ms: Milliseconds of silence before reporting SILENCE
            vad_threshold: Silero VAD sensitivity (0-1)
            sample_rate: Audio sample rate (16000 Hz expected)
            excluded_user_ids: User IDs to ignore (e.g., bot's own ID)
            on_state_change: Optional callback when state changes
        """
        self.energy_threshold = energy_threshold
        self.silence_debounce_ms = silence_debounce_ms
        self.vad_threshold = vad_threshold
        self.sample_rate = sample_rate
        self.excluded_user_ids = excluded_user_ids or set()
        self.on_state_change = on_state_change

        # Per-user state tracking
        self._user_states: dict[int, UserAudioState] = {}

        # Aggregate state
        self._last_speech_time: Optional[float] = None
        self._current_state = SpeakingState.SILENCE

        # Load shared VAD model
        self._vad_model = None
        self._load_vad_model()

        logger.info(
            f"SpeakingDetector initialized: energy_threshold={energy_threshold}, "
            f"silence_debounce_ms={silence_debounce_ms}, vad_threshold={vad_threshold}"
        )

    def _load_vad_model(self) -> None:
        """Load Silero VAD model."""
        try:
            from silero_vad import load_silero_vad

            self._vad_model = load_silero_vad()
            logger.info("SpeakingDetector: Silero VAD model loaded")
        except ImportError:
            # Fallback to torch.hub
            logger.info("silero_vad package not found, using torch.hub")
            try:
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self._vad_model = model
                logger.info("SpeakingDetector: Silero VAD model loaded (torch.hub)")
            except Exception as e:
                logger.error(f"Failed to load Silero VAD: {e}")
                raise

    def _get_or_create_user_state(self, user_id: int) -> UserAudioState:
        """Get existing user state or create new one."""
        if user_id not in self._user_states:
            self._user_states[user_id] = UserAudioState(user_id=user_id)
            logger.debug(f"SpeakingDetector: Created state for user {user_id}")
        return self._user_states[user_id]

    def _compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio chunk."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio**2)))

    def _run_vad(self, user: UserAudioState) -> bool:
        """Run VAD on user's buffered audio.

        Returns True if speech is detected in any complete chunk.
        """
        if self._vad_model is None:
            return False

        detected_speech = False

        while len(user.vad_buffer) >= self.VAD_CHUNK_SIZE:
            chunk = user.vad_buffer[: self.VAD_CHUNK_SIZE]
            user.vad_buffer = user.vad_buffer[self.VAD_CHUNK_SIZE :]

            # Convert to tensor and run inference
            audio_tensor = torch.from_numpy(chunk).float()
            with torch.no_grad():
                prob = self._vad_model(audio_tensor, self.sample_rate).item()

            if prob >= self.vad_threshold:
                detected_speech = True

        return detected_speech

    def process_audio(self, audio_16k: np.ndarray, user_id: int) -> SpeakingState:
        """Process audio from a single user.

        Args:
            audio_16k: Resampled audio (16kHz mono float32)
            user_id: Discord user ID

        Returns:
            Current channel-wide speaking state
        """
        # Skip excluded users (e.g., bot's own audio)
        if user_id in self.excluded_user_ids:
            return self.get_state()

        now = time.perf_counter()
        user = self._get_or_create_user_state(user_id)

        # Update packet tracking
        user.last_packet_time = now

        # Energy gate (fast rejection)
        energy = self._compute_rms(audio_16k)
        user.last_energy = energy

        # Per-user silence debounce (300ms) - don't clear is_speaking on brief pauses
        USER_SILENCE_DEBOUNCE_MS = 300

        if energy < self.energy_threshold:
            # Below energy threshold - likely noise or pause
            if user.is_speaking:
                # Start silence timer if not already started
                if user.silence_start_time is None:
                    user.silence_start_time = now
                # Check if silence has been long enough
                silence_ms = (now - user.silence_start_time) * 1000
                if silence_ms >= USER_SILENCE_DEBOUNCE_MS:
                    user.is_speaking = False
                    user.speech_start_time = None
                    user.silence_start_time = None
                    logger.debug(f"User {user_id}: Silence debounce expired, marked not speaking")
        else:
            # Above energy threshold - run VAD for confirmation
            user.vad_buffer = np.concatenate([user.vad_buffer, audio_16k])
            is_speech = self._run_vad(user)

            if is_speech:
                # Speech detected - reset silence timer
                user.silence_start_time = None
                if not user.is_speaking:
                    user.is_speaking = True
                    user.speech_start_time = now
                    logger.debug(f"User {user_id}: VAD detected speech start")
            elif user.is_speaking:
                # High energy but VAD says no speech - could be noise or brief pause
                # Start silence timer if not already started
                if user.silence_start_time is None:
                    user.silence_start_time = now
                # Check if silence has been long enough
                silence_ms = (now - user.silence_start_time) * 1000
                if silence_ms >= USER_SILENCE_DEBOUNCE_MS:
                    user.is_speaking = False
                    user.speech_start_time = None
                    user.silence_start_time = None
                    logger.debug(f"User {user_id}: VAD silence debounce expired, marked not speaking")

        # Compute aggregate state
        return self._compute_aggregate_state(now)

    def _compute_aggregate_state(self, now: float) -> SpeakingState:
        """Compute aggregate speaking state across all users."""
        any_speaking = any(user.is_speaking for user in self._user_states.values())

        if any_speaking:
            self._last_speech_time = now
            new_state = SpeakingState.SOMEONE_SPEAKING
        elif self._last_speech_time is None:
            # Never had speech
            new_state = SpeakingState.SILENCE
        else:
            # Check debounce
            silence_duration_ms = (now - self._last_speech_time) * 1000
            if silence_duration_ms >= self.silence_debounce_ms:
                new_state = SpeakingState.SILENCE
            else:
                # Still in debounce window
                new_state = SpeakingState.SOMEONE_SPEAKING

        # Fire callback on state change
        if new_state != self._current_state:
            old_state = self._current_state
            self._current_state = new_state
            logger.info(f"SpeakingDetector: State changed {old_state.value} -> {new_state.value}")
            if self.on_state_change:
                self.on_state_change(new_state)

        return self._current_state

    def get_state(self) -> SpeakingState:
        """Get current speaking state without processing new audio.

        Useful for polling in the event loop.
        """
        now = time.perf_counter()
        return self._compute_aggregate_state(now)

    def get_speaking_users(self) -> list[int]:
        """Get list of user IDs currently speaking."""
        return [user.user_id for user in self._user_states.values() if user.is_speaking]

    def is_anyone_actively_speaking(self) -> bool:
        """Check if anyone is actively speaking RIGHT NOW (no debounce).

        This is faster than get_state() for quick checks and doesn't apply
        the 3-second silence debounce. Use this for:
        - Deciding whether to play an acknowledgment immediately
        - Checking if someone is mid-sentence

        For "wait until conversation ends", use get_state() which applies debounce.
        """
        return any(user.is_speaking for user in self._user_states.values())

    def get_user_state(self, user_id: int) -> Optional[UserAudioState]:
        """Get state for a specific user."""
        return self._user_states.get(user_id)

    def remove_user(self, user_id: int) -> None:
        """Clean up when user leaves voice channel."""
        if user_id in self._user_states:
            del self._user_states[user_id]
            logger.debug(f"SpeakingDetector: Removed state for user {user_id}")

    def reset(self) -> None:
        """Reset all state."""
        self._user_states.clear()
        self._last_speech_time = None
        self._current_state = SpeakingState.SILENCE
        logger.debug("SpeakingDetector: Reset all state")
