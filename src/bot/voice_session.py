"""Voice session manager for Discord voice assistant.

Manages the voice pipeline state machine and coordinates all components
for a single Discord voice channel.
"""

import asyncio
import logging
import random
import re
import time
from typing import Optional

import discord
import numpy as np

from ..audio import SAMPLE_RATE_PROCESSING, SAMPLE_RATE_TTS
from ..config import get_settings
from ..detection import VAD, SpeakingDetector
from ..stt import GroqWhisperClient
from ..llm import BaseLLMClient, create_llm_client
from ..tts import TTSSynthesizer
from .discord_service import DiscordService
from .speaker_manager import SpeakerManager, SpeakerInfo
from .voice_sink import StreamingSink
from .voice_source import TTSAudioSource

logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages voice assistant session for a Discord voice channel.

    States:
    - idle: Waiting for wake word
    - listening: Recording user speech
    - processing: Transcribing and generating response
    """

    # Acknowledgment phrases (played when wake word detected)
    ACK_PHRASES = [
        "[happy] Yes?",
        "[surprised] Hmm?",
        "What's up?",
        "[happy] I'm listening...",
        "Go ahead...",
        "[breathe] I'm busy sorry...",
        "Yea, no, yea. [clear_throat] No.",
        "Yo?",
        "Sup?",
        "[happy] I'm here!"
    ]

    # Thinking phrases (randomly selected)
    # Generic phrases (used when speaker unknown)
    THINKING_PHRASES = [
        "[happy] One second...",
        "One moment...",
        "Let me check...",
        "Google it yourself... [breathe] Nah, I'll do it...",
        "Hmm, let me see...",
        "[happy] Just a moment...",
    ]

    # Personalized phrases with {name} placeholder (used when speaker is known)
    THINKING_PHRASES_PERSONAL = [
        "[happy] One second, {name}...",
        "One moment, {name}...",
        "Let me check, {name}...",
        "[laughing] Sure {name}, I'll do all the work...",
        "Google it yourself, {name}... [breathe] Nah, I'll do it...",
        "Hmm, let me see, {name}...",
        "[happy] Just a moment, {name}...",
        "Hold on, {name}...",
        "Give me a sec, {name}...",
    ]

    def __init__(
        self,
        *,
        vad: Optional[VAD] = None,
        speaking_detector: Optional[SpeakingDetector] = None,
        groq_client: Optional[GroqWhisperClient] = None,
        llm: Optional[BaseLLMClient] = None,
        tts: Optional[TTSSynthesizer] = None,
    ):
        """Initialize voice session.

        Components can be passed in (shared) or will be created.

        Args:
            vad: VAD instance
            speaking_detector: SpeakingDetector instance for multi-user awareness
            groq_client: GroqWhisperClient instance (STT)
            llm: LLM client instance (Claude or Groq)
            tts: TTSSynthesizer instance
        """
        self.settings = get_settings()

        # Speaker identification
        self.speaker_manager = SpeakerManager(self.settings.user_names)

        # Discord operations (image posting, user tagging)
        self.discord_service = DiscordService(
            channel_name=self.settings.image_channel_name,
            find_user_fn=self._find_user_by_name,
            get_known_users_fn=lambda: list(self.settings.user_names.values()),
        )

        # Components (will be initialized if not provided)
        # Note: Wake word detection is now per-user, created lazily in StreamingSink
        self.vad = vad
        self.speaking_detector = speaking_detector
        self.groq_client = groq_client
        self.llm = llm
        self.tts = tts

        # Discord voice
        self.voice_client: Optional[discord.VoiceClient] = None
        self.sink: Optional[StreamingSink] = None

        # State
        self._state = "idle"
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_buffer: list[np.ndarray] = []
        self._speech_started = False
        self._followup_window = 6.0  # Seconds to wait for follow-up

        # Events for thread-safe communication
        self._wake_event = asyncio.Event()
        self._speech_end_event = asyncio.Event()
        self._pending_audio: Optional[np.ndarray] = None

        # Parallel TTS pipeline
        # Queue holds (sentence_num, audio) tuples for ordered playback, or None to signal end
        self._audio_queue: asyncio.Queue[Optional[tuple[int, np.ndarray]]] = asyncio.Queue()
        self._playback_task: Optional[asyncio.Task] = None
        self._tts_tasks: list[asyncio.Task] = []
        self._thinking_task: Optional[asyncio.Task] = None
        self._thinking_synth_time: float = 0.0
        self._ack_played: bool = False  # Track if ack played this session
        self._listen_start: float = 0.0  # Track when listening started

        # Current speaker tracking
        self._current_speaker_id: Optional[int] = None
        self._current_speaker_name: Optional[str] = None
        self._current_speaker_activity: Optional[str] = None

        # Greeting cooldowns (user_id -> last_greeting_timestamp)
        self._greeting_cooldowns: dict[int, float] = {}

    @property
    def state(self) -> str:
        """Current session state."""
        return self._state

    @state.setter
    def state(self, value: str):
        """Set state and log transitions."""
        if value not in ("idle", "listening", "processing"):
            raise ValueError(f"Invalid state: {value}")

        old_state = self._state
        self._state = value

        if old_state != value:
            logger.info(f"Session state: {old_state} -> {value}")

    def _init_components(self):
        """Initialize ML components if not provided.

        Note: Wake word detectors are now created per-user in StreamingSink.
        """
        if self.vad is None:
            logger.info("Loading VAD...")
            self.vad = VAD(
                threshold=self.settings.vad_sensitivity,
                sample_rate=SAMPLE_RATE_PROCESSING,
                speech_pad_ms=self.settings.speech_pad_ms,
                min_silence_duration_ms=self.settings.min_silence_ms,
            )

        # Initialize speaking detector for multi-user awareness (if enabled)
        if self.speaking_detector is None and self.settings.speaking_detection_enabled:
            logger.info("Loading speaking detector...")
            # Exclude bot's own user ID to avoid detecting our own audio
            excluded_ids = set()
            if self.voice_client and self.voice_client.user:
                excluded_ids.add(self.voice_client.user.id)
            self.speaking_detector = SpeakingDetector(
                energy_threshold=self.settings.speaking_energy_threshold,
                silence_debounce_ms=self.settings.speaking_silence_debounce_ms,
                excluded_user_ids=excluded_ids,
            )

        # Initialize Groq Whisper STT
        if self.groq_client is None:
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required for STT")
            logger.info("Initializing Groq Whisper STT...")
            self.groq_client = GroqWhisperClient(
                api_key=self.settings.groq_api_key,
                model="whisper-large-v3-turbo",
                language="en",
            )

        if self.llm is None:
            logger.info("Setting up LLM client...")
            self.llm = create_llm_client(
                self.settings,
                on_image_generated=self._post_image_to_discord,
                on_user_tagged=self._post_tag_to_discord,
                on_message_posted=self._post_message_to_discord,
            )

        if self.tts is None:
            logger.info("Loading TTS...")
            self.tts = TTSSynthesizer(
                model=self.settings.tts_model,
                device=self.settings.tts_device,
                voice=self.settings.tts_voice,
                api_key=self.settings.inworld_api_key,
            )

    def _on_wake_word(self):
        """Callback when wake word detected (from sink thread)."""
        logger.info(f"[CALLBACK] _on_wake_word called, state={self._state}")
        if self._state == "idle" and self._loop:
            logger.info("[CALLBACK] Setting wake_event")
            self._loop.call_soon_threadsafe(self._wake_event.set)
        else:
            logger.warning(f"[CALLBACK] NOT setting event - state={self._state}, loop={self._loop is not None}")

    async def _post_image_to_discord(self, image_bytes: bytes, prompt: str, mime_type: str = "image/png") -> str:
        """Post a generated image to Discord. Delegates to DiscordService."""
        requester = self._current_speaker_name or "Someone"
        return await self.discord_service.post_image(
            self.voice_client, image_bytes, requester, mime_type
        )

    async def _post_tag_to_discord(self, target_name: str, message: str) -> str:
        """Post a tagged message to Discord. Delegates to DiscordService."""
        requester = self._current_speaker_name or "Someone"
        return await self.discord_service.tag_user(
            self.voice_client, target_name, message, requester
        )

    async def _post_message_to_discord(self, message: str) -> str:
        """Post a message to Discord. Delegates to DiscordService."""
        requester = self._current_speaker_name or "Someone"
        return await self.discord_service.post_message(
            self.voice_client, message, requester
        )

    def _find_user_by_name(self, name: str) -> Optional[tuple[int, str]]:
        """Find Discord user by fuzzy name match.

        Args:
            name: Name to search for (case-insensitive, partial match)

        Returns:
            Tuple of (user_id, display_name) if found, None otherwise
        """
        guild = self.voice_client.guild if self.voice_client else None
        return self.speaker_manager.find_user_by_name(name, guild)

    def _resolve_current_speaker(self, user_id: Optional[int]) -> None:
        """Resolve speaker info from user ID using SpeakerManager.

        Sets:
            self._current_speaker_id
            self._current_speaker_name
            self._current_speaker_activity
        """
        guild = self.voice_client.guild if self.voice_client else None
        info = self.speaker_manager.resolve_speaker(user_id, guild)
        self._current_speaker_id = info.user_id
        self._current_speaker_name = info.name
        self._current_speaker_activity = info.activity

    def _on_speech_end(self, audio: np.ndarray, user_id: Optional[int] = None):
        """Callback when speech ends (from sink thread)."""
        audio_duration = len(audio) / SAMPLE_RATE_PROCESSING if audio is not None else 0
        logger.info(f"[CALLBACK] _on_speech_end called with {audio_duration:.2f}s audio from user {user_id}, state={self._state}")
        if self._state == "listening" and self._loop:
            self._pending_audio = audio
            # Resolve speaker info
            self._resolve_current_speaker(user_id)
            logger.info(f"[CALLBACK] Speaker: {self._current_speaker_name} (ID: {user_id}), Activity: {self._current_speaker_activity}")
            logger.info("[CALLBACK] Setting speech_end_event")
            self._loop.call_soon_threadsafe(self._speech_end_event.set)
        else:
            logger.warning(f"[CALLBACK] NOT setting event - state={self._state}, loop={self._loop is not None}")

    async def start(self, voice_client: discord.VoiceClient):
        """Start the voice session.

        Args:
            voice_client: Connected Discord voice client
        """
        self.voice_client = voice_client
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Initialize components
        self._init_components()

        # Create the streaming sink with per-user wake detection config
        self.sink = StreamingSink(
            wake_model_name=self.settings.wake_word_model,
            wake_threshold=self.settings.wake_word_confidence,
            wake_sample_rate=SAMPLE_RATE_PROCESSING,
            vad=self.vad,
            speaking_detector=self.speaking_detector,
            on_wake_word=self._on_wake_word,
            on_speech_end=self._on_speech_end,
            max_speech_duration=self.settings.max_question_duration,
            speaker_isolation_enabled=self.settings.speaker_isolation_enabled,
        )

        # Start recording with the sink
        logger.info("Starting voice recording...")

        # Define a dummy callback since pycord requires one
        async def _recording_callback(sink: discord.sinks.Sink, *args):
            logger.debug("Recording callback triggered")

        self.voice_client.start_recording(self.sink, _recording_callback)

        # Clear any speaking state that might be set from joining
        # This fixes the "stuck green ring" on startup
        try:
            await self.voice_client.ws.speak(False)
        except Exception:
            pass  # May fail if websocket not ready

        # Run the main loop
        await self._run_loop()

    async def stop(self):
        """Stop the voice session."""
        logger.info("Stopping voice session...")
        self._running = False

        # Cancel thinking task
        if self._thinking_task and not self._thinking_task.done():
            self._thinking_task.cancel()
            try:
                await self._thinking_task
            except asyncio.CancelledError:
                pass
            self._thinking_task = None

        # Cancel playback consumer
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

        # Cancel TTS synthesis tasks
        for task in self._tts_tasks:
            if not task.done():
                task.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
            self._tts_tasks.clear()

        # Stop recording (may fail if socket already dead, that's OK)
        if self.voice_client:
            try:
                if self.voice_client.recording:
                    self.voice_client.stop_recording()
            except Exception as e:
                logger.debug(f"stop_recording failed (socket may be dead): {e}")

        # Stop any playback
        if self.voice_client:
            try:
                if self.voice_client.is_playing():
                    self.voice_client.stop()
            except Exception as e:
                logger.debug(f"stop playback failed: {e}")

        # Clear speaking state to fix "stuck green circle"
        if self.voice_client:
            try:
                await self.voice_client.ws.speak(False)
            except Exception as e:
                logger.debug(f"speak(False) failed: {e}")

        # Reset image editing context (Gemini chat session)
        if self.llm and hasattr(self.llm, '_image_service') and self.llm._image_service:
            self.llm._image_service.reset_chat()
            logger.debug("Reset image chat context on session stop")

    async def _run_loop(self):
        """Main session loop."""
        logger.info("Voice session ready - say wake word to activate")

        while self._running:
            try:
                # Reset to idle state
                self.state = "idle"
                self._wake_event.clear()
                self._speech_end_event.clear()
                self.sink.state = "idle"

                # Wait for wake word with periodic flush check
                # This handles the PTT edge case where user says wake word and stops
                # talking before more audio arrives to trigger detection
                logger.debug("Waiting for wake word...")
                idle_start = time.perf_counter()
                while not self._wake_event.is_set():
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=0.2)
                    except asyncio.TimeoutError:
                        # Check if wake word is in buffer but not yet detected
                        if self.sink and self.sink.check_idle_audio_gap():
                            logger.info("[STATE] Wake word detected from periodic flush")
                            self._wake_event.set()
                            break
                idle_time = time.perf_counter() - idle_start
                logger.info(f"[TIMING] Wake word detected after {idle_time:.1f}s idle")

                if not self._running:
                    break

                # Transition to listening
                self._listen_start = time.perf_counter()
                self.state = "listening"
                self._speech_started = False
                self._ack_played = False  # Reset for new session
                self.sink.start_listening()
                logger.info("[STATE] Now LISTENING for user speech (extends while speaking, 30s max)")

                # Play acknowledgment in background - waits for silence then plays
                # _ack_played flag tracks if it played, so _handle_wake_word_only() can skip
                ack_task = asyncio.create_task(self._play_ack())

                # Wait for speech to end (with timeout)
                # Poll for both VAD end AND push-to-talk audio stop
                try:
                    logger.info("[STATE] Waiting for speech_end_event...")
                    initial_timeout = 10.0  # Timeout for silence
                    max_timeout = self.settings.max_question_duration  # 30s safety cap

                    deadline = time.perf_counter() + initial_timeout
                    final_deadline = time.perf_counter() + max_timeout
                    speech_end_reason = None  # Track why speech ended

                    while time.perf_counter() < deadline:
                        # Check if VAD detected speech end
                        try:
                            await asyncio.wait_for(
                                self._speech_end_event.wait(),
                                timeout=0.2  # Poll every 200ms
                            )
                            # Event was set - speech ended via VAD
                            buffer_duration = len(self.sink.get_audio_buffer()) / SAMPLE_RATE_PROCESSING
                            speech_end_reason = f"VAD detected silence (buffer={buffer_duration:.2f}s)"
                            logger.info(f"[SPEECH_END] {speech_end_reason}")
                            break
                        except asyncio.TimeoutError:
                            pass

                        # Extend deadline while speech is active (up to max)
                        if self.sink.is_speech_active():
                            new_deadline = time.perf_counter() + initial_timeout
                            if new_deadline > deadline and new_deadline <= final_deadline:
                                deadline = new_deadline
                                logger.debug("[STATE] Extended deadline, speech active")

                        # Check if audio stopped (push-to-talk released)
                        if self.sink.check_audio_timeout():
                            self._pending_audio = self.sink.get_audio_buffer()
                            buffer_duration = len(self._pending_audio) / SAMPLE_RATE_PROCESSING if self._pending_audio is not None else 0
                            speech_end_reason = f"PTT released / no audio packets (buffer={buffer_duration:.2f}s)"
                            logger.info(f"[SPEECH_END] {speech_end_reason}")
                            # Resolve speaker info
                            self._resolve_current_speaker(self.sink._last_user_id)
                            logger.info(f"[STATE] Speaker: {self._current_speaker_name} (ID: {self._current_speaker_id}), Activity: {self._current_speaker_activity}")
                            break
                    else:
                        # Timeout - but still process any captured audio (don't abandon)
                        listen_time = time.perf_counter() - self._listen_start
                        buffer = self.sink.get_audio_buffer()
                        buffer_duration = len(buffer) / SAMPLE_RATE_PROCESSING if len(buffer) > 0 else 0
                        logger.warning(
                            f"[STATE] TIMEOUT after {listen_time:.2f}s - "
                            f"processing {buffer_duration:.2f}s of captured audio"
                        )
                        ack_task.cancel()
                        # Capture pending audio before falling through
                        self._pending_audio = buffer
                        # Resolve speaker info
                        self._resolve_current_speaker(self.sink._last_user_id)
                        # DON'T continue - fall through to process the audio

                    listen_time = time.perf_counter() - self._listen_start
                    # Stop the sink from buffering more audio
                    self.sink.stop_listening()
                    logger.info(f"[TIMING] Speech captured in {listen_time:.2f}s")
                except asyncio.CancelledError:
                    raise

                # Cancel ack if still running (may have already played, that's fine)
                ack_task.cancel()

                # Get the audio buffer
                audio = self._pending_audio
                if audio is None or len(audio) == 0:
                    logger.warning("[STATE] speech_end_event fired but no audio in buffer!")
                    continue

                # Process the speech
                await self._process_speech(audio)

                # Skip follow-up - require wake word each time
                # await self._followup_loop()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Prevent tight loop on errors

        logger.info("Voice session loop ended")

    async def _wait_for_silence(
        self,
        max_wait: float = 30.0,
        use_debounce: bool = True,
        poll_interval: float = 0.1,
    ) -> bool:
        """Wait until no one is speaking.

        Args:
            max_wait: Maximum seconds to wait before giving up
            use_debounce: If True, wait for full channel silence (3s debounce).
                         If False, return as soon as active speech stops.
            poll_interval: How often to check (seconds)

        Returns:
            True if silence detected, False if timed out
        """
        if not self.sink or not self.settings.speaking_detection_enabled:
            return True  # No detector, proceed immediately

        start = time.perf_counter()
        while time.perf_counter() - start < max_wait:
            if use_debounce:
                # Full channel silence (3s debounce) - for responses
                if not self.sink.is_anyone_speaking():
                    return True
            else:
                # Immediate check (no debounce) - for ack
                if not self.sink.is_anyone_actively_speaking():
                    return True
            await asyncio.sleep(poll_interval)

        return False  # Timed out

    async def _play_ack(self):
        """Play acknowledgment audio.

        Waits for the user to finish speaking before playing, so we don't
        interrupt questions like "hey jarvis what is the weather?".
        """
        try:
            # Wait for user to finish speaking (immediate mode, no 3s debounce)
            # This handles "hey jarvis what is the weather?" without interrupting
            if self.settings.speaking_detection_enabled:
                silence = await self._wait_for_silence(max_wait=5.0, use_debounce=False)
                if not silence:
                    # User still speaking after 5s - they're asking a long question
                    # Skip the ack, they'll hear the response later
                    logger.info("[ACK] Skipping - user still speaking after 5s")
                    return

            t0 = time.perf_counter()
            ack_phrase = random.choice(self.ACK_PHRASES)
            self._ack_played = True  # Mark early - prevents race with fast transcription
            ack_audio = await self.tts.synthesize(ack_phrase)
            synth_time = time.perf_counter() - t0
            logger.info(f"[TIMING] Ack TTS '{ack_phrase}': {synth_time:.2f}s")

            if len(ack_audio) > 0:
                t0 = time.perf_counter()
                await self._play_audio(ack_audio)
                play_time = time.perf_counter() - t0
                logger.info(f"[TIMING] Ack playback: {play_time:.2f}s")
        except asyncio.CancelledError:
            logger.debug("Ack playback cancelled")
            # Stop any ongoing playback when cancelled
            if self.voice_client and self.voice_client.is_playing():
                self.voice_client.stop()
        except Exception as e:
            logger.warning(f"Failed to play acknowledgment: {e}")

    async def _handle_wake_word_only(self):
        """Handle case where user only said the wake word.

        Plays acknowledgment (if not already played) and listens for follow-up question.
        If user asks something, processes it. If silence, returns to idle.
        """
        try:
            # Check how long we were listening - if ack played and user had time to respond, skip follow-up
            listen_duration = time.perf_counter() - self._listen_start
            if self._ack_played and listen_duration > 5.0:
                logger.info(f"[WAKE-ONLY] Ack played and listened for {listen_duration:.1f}s - user had time, returning to idle")
                self.state = "idle"
                return

            # Only play ack if one isn't already in progress
            # (_ack_played is set early in _play_ack to prevent race conditions)
            if not self._ack_played:
                ack_phrase = random.choice(self.ACK_PHRASES)
                logger.info(f"[WAKE-ONLY] Playing ack: '{ack_phrase}'")
                ack_audio = await self.tts.synthesize(ack_phrase)
                if len(ack_audio) > 0:
                    await self._play_audio(ack_audio)
            else:
                logger.info("[WAKE-ONLY] Ack already in progress, skipping duplicate")

            # Listen for follow-up question (user said wake word quickly, may be waiting for ack)
            logger.info("[WAKE-ONLY] Listening for follow-up question...")
            self.state = "listening"
            self.sink.start_listening()
            self._speech_end_event.clear()

            # Wait for speech with shorter timeout (user already has our attention)
            follow_up_timeout = 8.0  # 8 seconds to ask follow-up
            deadline = time.perf_counter() + follow_up_timeout

            while time.perf_counter() < deadline:
                try:
                    await asyncio.wait_for(
                        self._speech_end_event.wait(),
                        timeout=0.2
                    )
                    # VAD detected speech end
                    break
                except asyncio.TimeoutError:
                    pass

                # Extend deadline while speech is active
                if self.sink.is_speech_active():
                    new_deadline = time.perf_counter() + follow_up_timeout
                    max_deadline = time.perf_counter() + 30.0
                    if new_deadline > deadline and new_deadline <= max_deadline:
                        deadline = new_deadline

                # Check PTT timeout
                if self.sink.check_audio_timeout():
                    self._pending_audio = self.sink.get_audio_buffer()
                    break
            else:
                # Timeout with no speech - return to idle
                logger.info("[WAKE-ONLY] No follow-up question, returning to idle")
                self.sink.stop_listening()
                self.state = "idle"
                return

            # Got follow-up speech - process it
            self.sink.stop_listening()
            audio = self._pending_audio if self._pending_audio is not None else self.sink.get_audio_buffer()
            self._pending_audio = None

            if len(audio) < 1600:  # Less than 0.1s
                logger.info("[WAKE-ONLY] Follow-up audio too short, returning to idle")
                self.state = "idle"
                return

            # Process the follow-up (this will call _process_speech recursively)
            self.state = "processing"
            await self._process_speech(audio)

        except Exception as e:
            logger.error(f"[WAKE-ONLY] Error handling wake-word-only: {e}")
            self.state = "idle"

    async def _play_thinking_phrase(self, phrase: str) -> None:
        """Play thinking phrase in background.

        This runs as a task so LLM request can start in parallel.
        Tracks timing for logging.

        Args:
            phrase: The thinking phrase to synthesize and play
        """
        try:
            t0 = time.perf_counter()
            logger.info(f"[TIMING] Synthesizing thinking phrase: '{phrase}'")
            thinking_audio = await self.tts.synthesize(phrase)
            thinking_synth_time = time.perf_counter() - t0
            logger.info(f"[TIMING] Thinking phrase TTS: {thinking_synth_time:.2f}s")

            # Store synthesis time for final timing log
            self._thinking_synth_time = thinking_synth_time

            if len(thinking_audio) > 0:
                t0 = time.perf_counter()
                await self._play_audio(thinking_audio)
                thinking_play_time = time.perf_counter() - t0
                logger.info(f"[TIMING] Thinking phrase playback: {thinking_play_time:.2f}s")
        except asyncio.CancelledError:
            logger.debug("[THINKING] Playback cancelled")
            # Stop any ongoing playback when cancelled
            if self.voice_client and self.voice_client.is_playing():
                self.voice_client.stop()
            raise  # Re-raise to propagate cancellation
        except Exception as e:
            logger.warning(f"[THINKING] Failed to play thinking phrase: {e}")
            self._thinking_synth_time = 0.0  # Fallback for timing log

    async def greet_user(self, member: discord.Member) -> None:
        """Greet a user who just joined the voice channel.

        Only greets if:
        - Greeting is enabled in settings
        - Bot is idle (not listening/processing)
        - User hasn't been greeted within cooldown period
        - Silence is detected within timeout period

        Args:
            member: Discord member who joined
        """
        if not self.settings.greeting_enabled:
            return

        # Skip if bot is busy
        if self.state != "idle":
            logger.info(f"[GREETING] Skipping - bot is busy ({self.state})")
            return

        # Check cooldown (5-minute window by default)
        now = time.time()
        last_greeting = self._greeting_cooldowns.get(member.id, 0)
        if now - last_greeting < self.settings.greeting_cooldown_seconds:
            logger.info(f"[GREETING] Skipping {member.display_name} - cooldown active")
            return

        # Wait for silence (max 10 seconds by default, no debounce)
        found_silence = await self._wait_for_silence(
            max_wait=self.settings.greeting_timeout_seconds,
            use_debounce=False,
            poll_interval=0.1
        )

        if not found_silence:
            logger.info(
                f"[GREETING] Skipping {member.display_name} - "
                f"no silence in {self.settings.greeting_timeout_seconds}s"
            )
            return

        # Re-check state (might have changed during wait)
        if self.state != "idle":
            logger.info("[GREETING] Skipping - bot became busy during wait")
            return

        # Generate greeting via LLM
        greeting = await self._generate_greeting(member.display_name)
        if not greeting:
            return

        # Update cooldown
        self._greeting_cooldowns[member.id] = now

        # Synthesize and play
        logger.info(f"[GREETING] Greeting {member.display_name}: {greeting}")
        audio = await self.tts.synthesize(greeting)
        if len(audio) > 0:
            await self._play_audio(audio)

    async def _generate_greeting(self, name: str) -> Optional[str]:
        """Ask LLM to generate a short greeting for the user.

        Uses chat() not quick_response() because quick_response() has a simplified
        system prompt without the personality/roasts. chat() uses the full
        _build_system_prompt() which includes roast info for known users.

        Args:
            name: Display name of user to greet

        Returns:
            Greeting text or None if generation failed
        """
        prompt = (
            f"Generate a very short greeting (20 words max) for {name} "
            f"who just joined the voice channel."
        )

        try:
            # Use chat() to get full personality (roasts are in system prompt)
            response = await self.llm.chat(prompt)
            # Ensure it's short for TTS
            if len(response) > 100:
                response = response[:100].rsplit(' ', 1)[0]
            return response.strip()
        except Exception as e:
            logger.error(f"[GREETING] LLM error: {e}")
            return None

    async def _play_audio(self, audio: np.ndarray):
        """Play audio through Discord voice.

        Args:
            audio: Audio as float32 numpy array (24kHz mono)
        """
        if not self.voice_client or len(audio) == 0:
            logger.debug("[TIMING] _play_audio: no audio or no voice client")
            return

        audio_duration = len(audio) / SAMPLE_RATE_TTS  # 24kHz
        logger.debug(f"[TIMING] _play_audio: {audio_duration:.2f}s of audio")

        # Create audio source
        t0 = time.perf_counter()
        source = TTSAudioSource(audio)
        source_time = time.perf_counter() - t0
        logger.debug(f"[TIMING] TTSAudioSource creation: {source_time*1000:.1f}ms")

        # Play and wait for completion
        playback_done = asyncio.Event()
        playback_start = time.perf_counter()

        def after_playback(error):
            if error:
                logger.error(f"Playback error: {error}")
            elapsed = time.perf_counter() - playback_start
            logger.debug(f"[TIMING] Discord playback callback after: {elapsed:.2f}s")
            self._loop.call_soon_threadsafe(playback_done.set)

        # Stop any current playback
        if self.voice_client.is_playing():
            self.voice_client.stop()

        self.voice_client.play(source, after=after_playback)

        # Wait for playback to complete
        await playback_done.wait()

        # Clean up source and natural pause between sentences
        source.cleanup()
        await asyncio.sleep(0.15)  # 150ms gap for natural sentence pacing

    async def _playback_consumer(self):
        """Background task that plays audio from the queue seamlessly.

        Receives (sentence_num, audio) tuples and plays in order.
        Waits for:
        1. Thinking phrase to complete (if playing)
        2. Channel silence before playing the first sentence
        """
        logger.debug("[PLAYBACK] Consumer started")
        next_to_play = 1
        pending: dict[int, np.ndarray] = {}
        waited_for_thinking = False
        waited_for_silence = False

        while True:
            try:
                item = await self._audio_queue.get()

                # None signals end of stream
                if item is None:
                    # Play any remaining buffered audio in order
                    while next_to_play in pending:
                        audio = pending.pop(next_to_play)
                        if len(audio) > 0:
                            await self._play_audio(audio)
                        next_to_play += 1
                    logger.debug("[PLAYBACK] Received end signal")
                    break

                sentence_num, audio = item
                pending[sentence_num] = audio
                logger.debug(f"[PLAYBACK] Received sentence {sentence_num}, next_to_play={next_to_play}")

                # Play all consecutive available sentences
                while next_to_play in pending:
                    audio = pending.pop(next_to_play)
                    if len(audio) > 0:
                        # Wait for thinking phrase before FIRST sentence
                        if not waited_for_thinking and self._thinking_task is not None:
                            logger.info("[PLAYBACK] Waiting for thinking phrase to complete...")
                            try:
                                await self._thinking_task
                            except asyncio.CancelledError:
                                logger.debug("[PLAYBACK] Thinking task was cancelled")
                            except Exception as e:
                                logger.warning(f"[PLAYBACK] Thinking task error: {e}")
                            waited_for_thinking = True
                            self._thinking_task = None

                        # Wait for silence before FIRST sentence only
                        # This handles conversations while waiting for LLM response
                        if not waited_for_silence and self.settings.speaking_detection_enabled:
                            logger.info("[PLAYBACK] Waiting for channel silence before response...")
                            silence = await self._wait_for_silence(max_wait=30.0, use_debounce=True)
                            waited_for_silence = True
                            if not silence:
                                logger.warning("[PLAYBACK] Timeout waiting for silence, playing anyway")

                        logger.debug(f"[PLAYBACK] Playing sentence {next_to_play}")
                        await self._play_audio(audio)
                    next_to_play += 1

                self._audio_queue.task_done()
            except asyncio.CancelledError:
                logger.debug("[PLAYBACK] Consumer cancelled")
                break
            except Exception as e:
                logger.error(f"[PLAYBACK] Error: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        logger.debug("[PLAYBACK] Consumer stopped")

    async def _synthesize_and_queue(self, text: str, sentence_num: int):
        """Synthesize text using streaming and add to audio queue with ordering info."""
        try:
            t0 = time.perf_counter()
            chunks = []
            first_chunk_time = None

            # Use streaming synthesis - yields chunks as they arrive
            async for chunk in self.tts.synthesize_stream(text):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    ttfb = (first_chunk_time - t0) * 1000
                    logger.debug(f"[TTS] Sentence {sentence_num} first chunk in {ttfb:.0f}ms")
                chunks.append(chunk)

            elapsed = time.perf_counter() - t0
            if chunks:
                audio = np.concatenate(chunks)
                logger.debug(f"[TTS] Sentence {sentence_num}: {len(chunks)} chunks in {elapsed:.2f}s")
            else:
                audio = np.array([], dtype=np.float32)
                logger.debug(f"[TTS] Sentence {sentence_num}: no audio in {elapsed:.2f}s")

            # Queue with sentence number for ordered playback
            await self._audio_queue.put((sentence_num, audio))
        except Exception as e:
            logger.error(f"[TTS] Error synthesizing sentence {sentence_num}: {e}")
            # Queue empty audio to not block ordering
            await self._audio_queue.put((sentence_num, np.array([], dtype=np.float32)))

    # Wake word pattern for stripping from transcription
    _WAKE_PATTERN = re.compile(
        r'^(hey[,\s]*jarvis|a[,\s]*jarvis)[,.\s!?]*',
        re.IGNORECASE
    )

    def _strip_wake_word(self, text: str) -> Optional[str]:
        """Strip wake word from transcription.

        Args:
            text: Raw transcription text.

        Returns:
            Text with wake word stripped, or None if only wake word was said.
        """
        text_stripped = text.strip()
        match = self._WAKE_PATTERN.match(text_stripped)

        if match:
            text_stripped = text_stripped[match.end():].strip()
            if text_stripped:
                logger.info(f"[WAKE-STRIP] Stripped wake word, remaining: '{text_stripped}'")
            else:
                return None

        # If stripping left nothing meaningful
        if not text_stripped or len(text_stripped) < 3:
            return None

        return text_stripped

    def _build_prompt(self, text: str) -> str:
        """Build LLM prompt with speaker context.

        Args:
            text: User's transcribed text.

        Returns:
            Prompt string with speaker info if known.
        """
        speaker = self._current_speaker_name
        activity = self._current_speaker_activity

        if speaker:
            logger.info(f"{speaker} said: {text}" + (f" (Activity: {activity})" if activity else ""))
            if activity:
                return f"[{speaker} speaking, currently {activity}]: {text}"
            else:
                return f"[{speaker} speaking]: {text}"
        else:
            logger.info(f"User said: {text}")
            return text

    async def _process_speech(self, audio: np.ndarray):
        """Process captured speech: transcribe -> LLM -> stream TTS.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
        """
        self.state = "processing"
        total_start = time.perf_counter()

        # Log audio stats
        audio_duration = len(audio) / SAMPLE_RATE_PROCESSING
        logger.info(f"[TIMING] Audio captured: {audio_duration:.2f}s ({len(audio)} samples)")

        # Transcribe with Groq Whisper
        t0 = time.perf_counter()
        logger.info("[TIMING] Starting transcription...")
        text = await self.groq_client.transcribe_audio(audio)

        transcribe_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Transcription: {transcribe_time:.2f}s")

        if not text.strip():
            logger.info("No speech detected")
            return

        # Strip wake word
        text = self._strip_wake_word(text)
        if text is None:
            logger.info("[WAKE-ONLY] User only said wake word - playing ack and listening for follow-up")
            await self._handle_wake_word_only()
            return

        # Build prompt with speaker context
        prompt_text = self._build_prompt(text)

        # Play thinking phrase (personalized if we know who's speaking)
        # Skip if someone is still talking (they might be in conversation)
        should_play_thinking = True
        if self.settings.speaking_detection_enabled:
            silence = await self._wait_for_silence(max_wait=3.0, use_debounce=False)
            if not silence:
                logger.info("[THINKING] Skipping thinking phrase - someone still speaking")
                should_play_thinking = False

        # Reset thinking timing
        self._thinking_synth_time = 0.0

        if should_play_thinking:
            if self._current_speaker_name:
                phrase = random.choice(self.THINKING_PHRASES_PERSONAL).format(name=self._current_speaker_name)
            else:
                phrase = random.choice(self.THINKING_PHRASES)

            # Start thinking phrase in background (doesn't block LLM request)
            self._thinking_task = asyncio.create_task(self._play_thinking_phrase(phrase))
        else:
            self._thinking_task = None

        # Get LLM response with PARALLEL sentence-level streaming TTS
        # LLM starts immediately - runs in parallel with thinking phrase playback
        t0 = time.perf_counter()
        logger.info("[TIMING] Starting LLM request (parallel with thinking phrase)...")

        buffer = ""
        first_chunk_time = None
        sentences_queued = 0
        total_response = ""

        # Clear any previous state
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._tts_tasks.clear()

        # Start playback consumer (plays audio from queue back-to-back)
        self._playback_task = asyncio.create_task(self._playback_consumer())

        try:
            async for chunk in self.llm.chat_stream(prompt_text):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - t0
                    logger.info(f"[TIMING] LLM first chunk: {first_chunk_time:.2f}s")

                buffer += chunk
                total_response += chunk

                # Claude outputs one sentence per line - split on newlines for streaming
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line:
                        sentences_queued += 1
                        logger.info(f"[STREAMING] Queuing line {sentences_queued}: {line[:50]}...")

                        # Spawn TTS task (don't wait - runs in parallel)
                        task = asyncio.create_task(
                            self._synthesize_and_queue(line, sentences_queued)
                        )
                        self._tts_tasks.append(task)

                # Also queue if buffer ends with punctuation (sentence complete, no newline yet)
                # This handles "I'll look that up." before a long tool call
                stripped = buffer.strip()
                if stripped and stripped[-1] in '.!?':
                    sentences_queued += 1
                    logger.info(f"[STREAMING] Queuing sentence {sentences_queued}: {stripped[:50]}...")

                    task = asyncio.create_task(
                        self._synthesize_and_queue(stripped, sentences_queued)
                    )
                    self._tts_tasks.append(task)
                    buffer = ""  # Clear the buffer since we queued it

            # Queue any remaining text
            remaining = buffer.strip()
            if remaining:
                sentences_queued += 1
                logger.info(f"[STREAMING] Queuing remaining: {remaining[:50]}...")
                task = asyncio.create_task(
                    self._synthesize_and_queue(remaining, sentences_queued)
                )
                self._tts_tasks.append(task)

        finally:
            # Ensure cleanup even on errors
            # Wait for all TTS tasks to complete (audio is now in queue)
            if self._tts_tasks:
                await asyncio.gather(*self._tts_tasks, return_exceptions=True)
                self._tts_tasks.clear()

            # Signal end of stream and wait for playback to finish
            await self._audio_queue.put(None)
            if self._playback_task:
                await self._playback_task
                self._playback_task = None

            # Clean up thinking task if still running
            if self._thinking_task and not self._thinking_task.done():
                self._thinking_task.cancel()
                try:
                    await self._thinking_task
                except asyncio.CancelledError:
                    pass
                self._thinking_task = None

        llm_total_time = time.perf_counter() - t0
        logger.info(f"[TIMING] LLM total: {llm_total_time:.2f}s ({len(total_response)} chars, {sentences_queued} sentences streamed)")
        logger.info(f"LLM response: {total_response}")

        # Total summary
        total_time = time.perf_counter() - total_start
        logger.info(
            f"[TIMING] === TOTAL: {total_time:.2f}s === "
            f"(transcribe={transcribe_time:.1f}s, "
            f"think_tts={self._thinking_synth_time:.1f}s, "
            f"llm+streaming_tts={llm_total_time:.1f}s)"
        )

    async def _followup_loop(self):
        """Handle follow-up questions without wake word."""
        while self._running:
            logger.info(f"Listening for follow-up ({self._followup_window}s window)...")

            # Reset for follow-up listening
            self.state = "listening"
            self._speech_end_event.clear()
            self._pending_audio = None
            self.sink.start_listening()

            # Wait for speech with follow-up timeout (poll for PTT too)
            deadline = time.perf_counter() + self._followup_window
            got_speech = False
            while time.perf_counter() < deadline:
                try:
                    await asyncio.wait_for(
                        self._speech_end_event.wait(),
                        timeout=0.2
                    )
                    got_speech = True
                    break
                except asyncio.TimeoutError:
                    pass

                # Check for PTT release
                if self.sink.check_audio_timeout():
                    logger.info("[STATE] Audio stopped in follow-up (PTT released)")
                    self._pending_audio = self.sink.get_audio_buffer()
                    got_speech = True
                    break

            if not got_speech:
                logger.info("No follow-up detected, returning to wake word mode")
                break

            # Get the audio
            audio = self._pending_audio
            if audio is None or len(audio) == 0:
                logger.info("No audio in follow-up")
                break

            # Process follow-up
            await self._process_speech(audio)
