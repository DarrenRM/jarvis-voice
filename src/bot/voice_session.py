"""Voice session manager for Discord voice assistant.

Manages the voice pipeline state machine and coordinates all components
for a single Discord voice channel.
"""

import asyncio
import contextlib
import logging
import re
import time
from typing import TYPE_CHECKING, Optional

import discord
import numpy as np

from ..audio import SAMPLE_RATE_PROCESSING, SAMPLE_RATE_TTS
from ..config import get_settings
from ..detection import VAD, SpeakingDetector
from ..stt import GroqWhisperClient
from ..llm import LLMService, create_llm_client
from ..llm.chat_post import ChatPoster, JevPostJudge, QwenPostWriter
from ..llm.game_notes import GameNotes
from ..llm.addressee_judge import (
    AddresseeJudge,
    FollowUp,
    JevAddresseeJudge,
    QwenAddresseeJudge,
    is_question,
)
from ..pipeline import (
    OpeningPhrase,
    PipelineContext,
    ResponseStage,
    TranscriptionResult,
    TranscriptionStage,
    TTSStage,
    VoicePipeline,
)
from ..tts import TTSSynthesizer
from .discord_service import DiscordService
from .speaker_manager import SpeakerManager
from . import pycord_dave_patch
from .voice_sink import CARRY_ON_MIN_SPEECH_SECONDS, StreamingSink
from .voice_source import TTSAudioSource

from ..session.speaker_tracker import SpeakerTracker
from ..session.greeting_service import GreetingService
from ..session.phrases import get_random_ack, get_thinking_phrase, get_affirmation_phrase
from ..session.thinking_phrase_generator import ThinkingPhraseGenerator

# _process_speech_with_pipeline result: the speaker carried on talking, start over
CARRY_ON = "carry_on"
# After a question's reply window closes unanswered, how long to watch for a late reply
LATE_REPLY_WATCH_SECONDS = 5.0
# py-cord calls a clip's after-callback when it ends; if it never does, stop waiting this
# long past the clip's own length instead of hanging the session
PLAYBACK_CALLBACK_GRACE_SECONDS = 5.0

if TYPE_CHECKING:
    from ..monitor import TurnRecorder

logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages voice assistant session for a Discord voice channel.

    States:
    - idle: Waiting for wake word
    - listening: Recording user speech
    - processing: Transcribing and generating response
    """

    def __init__(
        self,
        *,
        vad: Optional[VAD] = None,
        speaking_detector: Optional[SpeakingDetector] = None,
        groq_client: Optional[GroqWhisperClient] = None,
        llm: Optional[LLMService] = None,
        tts: Optional[TTSSynthesizer] = None,
        turn_observer: Optional["TurnRecorder"] = None,
    ):
        """Initialize voice session.

        Components can be passed in (shared) or will be created.

        Args:
            vad: VAD instance
            speaking_detector: SpeakingDetector instance for multi-user awareness
            groq_client: GroqWhisperClient instance (STT)
            llm: LLM service (Qwen on Cerebras)
            tts: TTSSynthesizer instance
            turn_observer: Live monitor's recorder (MONITOR_ENABLED); notified of each
                turn's timings and results, never changes what the session does
        """
        self.settings = get_settings()
        self.turn_observer = turn_observer

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

        # Transcription -> response -> TTS (built in _init_components)
        self._voice_pipeline: Optional[VoicePipeline] = None

        # Discord voice
        self.voice_client: Optional[discord.VoiceClient] = None
        self.sink: Optional[StreamingSink] = None

        # State
        self._state = "idle"
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._speech_started = False
        self._addressee_judge: Optional[AddresseeJudge] = None  # Follow-up gate
        self._chat_poster: Optional[ChatPoster] = None  # Reference answers to chat
        self._followup_open = False
        # Someone else said the wake word during a follow-up window: their user ID
        self._wake_from_followup: Optional[int] = None
        # Everything the speaker said when a turn is started over (CARRY_ON)
        self._carry_on_audio: Optional[np.ndarray] = None
        # Silero for checking captured speech (_speech_confirmed); loaded on first use
        self._speech_check_vad: Optional[VAD] = None
        self._speech_check_lock = asyncio.Lock()
        self._followup_others_talked = 0.0  # last follow-up window: others' talk, seconds
        self._late_reply_task: Optional[asyncio.Task] = None

        # Events for thread-safe communication
        self._wake_event = asyncio.Event()
        self._speech_end_event = asyncio.Event()
        self._pending_audio: Optional[np.ndarray] = None

        self._warmup_task: Optional[asyncio.Task] = None
        self._ack_played: bool = False  # Track if ack played this session
        self._listen_start: float = 0.0  # Track when listening started

        # Current speaker tracking
        self._current_speaker_id: Optional[int] = None
        self._current_speaker_name: Optional[str] = None
        self._current_speaker_activity: Optional[str] = None

        # Keep-alive tracking (prevents Discord disconnect after ~1 hour idle)
        self._last_keep_alive_time: float = 0.0

        # Session components (initialized here or in _init_components)
        self._speaker_tracker = SpeakerTracker(self.speaker_manager)
        self._greeting_service: Optional[GreetingService] = None
        self._thinking_generator: Optional[ThinkingPhraseGenerator] = None
        # Note: GreetingService is initialized in _init_components after TTS is ready

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
            self._owns_tts = True
            self.tts = TTSSynthesizer(
                model=self.settings.tts_model,
                voice=self.settings.tts_voice,
                api_key=self.settings.inworld_api_key,
                temperature=self.settings.tts_temperature,
                speed=self.settings.tts_speed,
            )

        # Initialize GreetingService
        if self._greeting_service is None:
            self._greeting_service = GreetingService(
                llm_client=self.llm,
                tts_synthesizer=self.tts,
                settings=self.settings,
            )

        # Initialize dynamic thinking phrase generator (if enabled)
        if self._thinking_generator is None and self.settings.dynamic_thinking_enabled:
            if self.settings.cerebras_api_key:
                logger.info("Initializing dynamic thinking phrase generator...")
                self._thinking_generator = ThinkingPhraseGenerator(
                    api_key=self.settings.cerebras_api_key,
                    model=self.settings.cerebras_model,
                )
            else:
                logger.warning("DYNAMIC_THINKING_ENABLED but CEREBRAS_API_KEY not set, disabling")

        # Follow-ups without the wake word need a check that the speech was for the bot
        if self._addressee_judge is None and self.settings.followup_enabled:
            if self.settings.typesafe_api_key:
                self._addressee_judge = JevAddresseeJudge(self.settings.typesafe_api_key)
                logger.info("Follow-up check: TypeSafe Jev")
            elif self.settings.cerebras_api_key:
                self._addressee_judge = QwenAddresseeJudge(
                    self.settings.cerebras_api_key, self.settings.cerebras_model
                )
                logger.info("Follow-up check: Qwen on Cerebras")

        # Recipes, steps and stats also get posted to chat (Jev decides, Qwen writes)
        if self._chat_poster is None and self.settings.chat_posts_enabled:
            if self.settings.typesafe_api_key and self.settings.cerebras_api_key:
                self._chat_poster = ChatPoster(
                    JevPostJudge(self.settings.typesafe_api_key),
                    QwenPostWriter(self.settings.cerebras_api_key, self.settings.cerebras_model),
                    post=self._post_message_to_discord,
                    speak=self.tts.synthesize_cached,
                )
                logger.info("Chat posts: on (TypeSafe Jev + Qwen)")
            else:
                logger.info("Chat posts: off (needs TYPESAFE_API_KEY and CEREBRAS_API_KEY)")

        # Initialize modular VoicePipeline
        if self._voice_pipeline is None:
            game_notes = GameNotes.load()
            transcription_stage = TranscriptionStage(
                self.groq_client, corrections=game_notes.corrections()
            )
            response_stage = ResponseStage(
                self.llm.chat_orchestrator,
                self.llm.build_system_prompt,
                max_history_messages=self.settings.llm_history_max_messages,
                game_notes=game_notes,
            )
            tts_stage = TTSStage(self.tts)

            self._voice_pipeline = VoicePipeline(
                transcription=transcription_stage,
                response=response_stage,
                tts=tts_stage,
            )
            self._voice_pipeline.observer = self.turn_observer
            self._voice_pipeline.poster = self._chat_poster

    def _on_wake_word(self):
        """Callback when wake word detected (from sink thread)."""
        logger.info(f"[CALLBACK] _on_wake_word called, state={self._state}")
        if self._state == "idle" and self._loop:
            logger.info("[CALLBACK] Setting wake_event")
            self._loop.call_soon_threadsafe(self._wake_event.set)
        elif self._followup_open and self.sink is not None:
            # Read now: going idle clears the wake word user
            self._wake_from_followup = self.sink.wake_user_id
            logger.info(f"[CALLBACK] Wake word from {self._wake_from_followup} ends the follow-up window")
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
        """Resolve speaker info from user ID using SpeakerTracker.

        Sets:
            self._current_speaker_id
            self._current_speaker_name
            self._current_speaker_activity
        """
        self._speaker_tracker.set_speaker(user_id, self.voice_client)
        self._current_speaker_id = self._speaker_tracker.current_speaker_id
        self._current_speaker_name = self._speaker_tracker.current_speaker_name
        self._current_speaker_activity = self._speaker_tracker.current_speaker_activity

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

        # Open API connections in the background so the first question isn't slow
        self._warmup_task = asyncio.create_task(self._warm_up())

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
            on_wake_clip=getattr(self.turn_observer, "wake_clip", None),
        )

        # Start recording with the sink
        logger.info("Starting voice recording...")

        # py-cord 2.8 calls this with the error (if any) when recording stops. While the
        # session is running that means the bot can no longer hear anyone: say so loudly.
        def _recording_callback(*args):
            if self._running:
                logger.error(f"[RECORDING] Voice receive stopped during the session: {args}")
            else:
                logger.debug(f"Recording stopped: {args}")

        # py-cord 2.8 no longer calls sink.init(); its decoder reads the voice client off the sink
        self.sink.init(self.voice_client)
        pycord_dave_patch.apply()
        self.voice_client.start_recording(self.sink, _recording_callback)

        # Clear any speaking state that might be set from joining
        # This fixes the "stuck green ring" on startup
        try:
            await self.voice_client.ws.speak(False)
        except Exception:
            pass  # May fail if websocket not ready
        self._reset_speaking_ring()

        # Run the main loop
        await self._run_loop()

    async def _warm_up(self) -> None:
        """Warm STT, LLM and thinking-phrase connections concurrently (failures only log)."""
        try:
            jobs = [self.groq_client.warm_up(), self.llm.warm_up()]
            if self._thinking_generator:
                jobs.append(self._thinking_generator.warm_up())
            if self._chat_poster:
                # Cache "I'll post this in chat too!" so it plays without a TTS round trip
                jobs.append(self._chat_poster.announcement())
            await asyncio.gather(*jobs, return_exceptions=True)
        except Exception as e:
            logger.warning(f"[WARMUP] Skipped: {e}")

    async def stop(self):
        """Stop the voice session."""
        logger.info("Stopping voice session...")
        self._running = False
        if self._late_reply_task is not None:
            self._late_reply_task.cancel()

        # A shared TTS is closed by its owner (main); close one made here
        if getattr(self, "_owns_tts", False) and self.tts is not None:
            await self.tts.close()

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
        # Let posts still being written land, then close the checker's connection
        if self._chat_poster is not None:
            try:
                await asyncio.wait_for(self._chat_poster.close(), 25)
            except Exception as e:
                logger.debug(f"Chat poster close failed: {e}")

        if self.llm and self.llm.image_service:
            self.llm.image_service.reset_chat()
            logger.debug("Reset image chat context on session stop")

    def _reset_speaking_ring(self) -> None:
        """Play 0.1 s of silence on a real voice connection.

        A connection that dropped while speaking (killed process, a test client) can
        leave the bot's green ring on, and a bare speak(False) from the new connection
        doesn't clear it. py-cord's player sends speaking on, then off, which does.
        """
        import io

        if not isinstance(self.voice_client, discord.VoiceClient):
            return  # test doubles
        try:
            if not self.voice_client.is_playing():
                self.voice_client.play(discord.PCMAudio(io.BytesIO(b"\x00" * 3840 * 5)))
        except Exception as e:
            logger.debug(f"Speaking ring reset failed: {e}")

    async def _run_loop(self):
        """Main session loop."""
        logger.info("Voice session ready - say wake word to activate")

        while self._running:
            try:
                # Check for silent disconnect
                if not self.voice_client or not self.voice_client.is_connected():
                    logger.warning("[DISCONNECT] Voice client disconnected, ending session")
                    break

                # Reset to idle state
                self.state = "idle"
                self._wake_event.clear()
                self._speech_end_event.clear()
                self.sink.state = "idle"
                # Someone said the wake word during a follow-up window: their turn now
                wake_user = self._wake_from_followup
                self._wake_from_followup = None
                if wake_user is not None:
                    self._wake_event.set()

                # Wait for wake word with periodic flush check
                # This handles the PTT edge case where user says wake word and stops
                # talking before more audio arrives to trigger detection
                logger.debug("Waiting for wake word...")
                idle_start = time.perf_counter()
                self._last_keep_alive_time = idle_start  # Reset keep-alive timer
                while not self._wake_event.is_set():
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=0.2)
                    except asyncio.TimeoutError:
                        # Check for silent disconnect during idle
                        if not self.voice_client or not self.voice_client.is_connected():
                            logger.warning("[DISCONNECT] Voice client disconnected during idle")
                            self._running = False
                            break

                        # Keep-alive check: say something before Discord disconnects us
                        now = time.perf_counter()
                        if (self.settings.keep_alive_enabled and
                                now - self._last_keep_alive_time >= self.settings.keep_alive_interval_seconds):
                            await self._play_keep_alive()
                            self._last_keep_alive_time = now

                        # Check if wake word is in buffer but not yet detected
                        if self.sink and self.sink.check_idle_audio_gap():
                            logger.info("[STATE] Wake word detected from periodic flush")
                            self._wake_event.set()
                            break
                if not self._running:
                    break

                idle_time = time.perf_counter() - idle_start
                logger.info(f"[TIMING] Wake word detected after {idle_time:.1f}s idle")

                # Transition to listening
                self._listen_start = time.perf_counter()
                self.state = "listening"
                self._speech_started = False
                self._ack_played = False  # Reset for new session
                self.sink.start_listening(user_id=wake_user)
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
                            self._resolve_current_speaker(self.sink.wake_user_id)
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
                        self._resolve_current_speaker(self.sink.wake_user_id)
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
                result = await self._process_speech(audio)

                # If the bot asked something, the asker can answer without the wake word
                await self._followup_after(result)
                # Whatever the speaker said after that isn't part of a turn any more
                self.sink.stop_capture(only_fold=True)

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

    async def _play_keep_alive(self) -> bool:
        """Play keep-alive message to prevent Discord disconnect.

        Generates a message via LLM using the configurable prompt, then
        synthesizes and plays it. Used when idle for too long (~50 min).

        Returns:
            True if keep-alive was played, False on error
        """
        if not self.settings.keep_alive_enabled:
            return False

        try:
            logger.info("[KEEP-ALIVE] Generating keep-alive message to prevent disconnect")

            # Generate message via LLM
            response = await self.llm.chat(self.settings.keep_alive_prompt)
            if not response or len(response.strip()) == 0:
                logger.warning("[KEEP-ALIVE] LLM returned empty response")
                return False

            # Truncate if too long
            message = response.strip()
            if len(message) > 200:
                message = message[:200].rsplit(' ', 1)[0]

            logger.info(f"[KEEP-ALIVE] Playing: {message}")

            # Synthesize and play
            audio = await self.tts.synthesize(message)
            if len(audio) > 0:
                await self._play_audio(audio)
                return True

            return False

        except Exception as e:
            logger.error(f"[KEEP-ALIVE] Error: {e}")
            return False

    async def _play_ack(self):
        """Play acknowledgment audio.

        Waits for the user to finish speaking before playing, so we don't
        interrupt questions like "hey jarvis what is the weather?".

        Uses two-phase checking:
        1. After 300ms delay, check if sink VAD detected speech (fast)
        2. After synthesis (~800ms), check again before playing (catches continuous speech)
        """
        try:
            # Wait for user to finish speaking (immediate mode, no 3s debounce)
            # This handles "hey jarvis what is the weather?" without interrupting
            if self.settings.speaking_detection_enabled:
                # Initial delay to let VAD process incoming audio.
                # Without this, we check too early and think no one is speaking
                # when user is mid-sentence (race condition with audio processing).
                await asyncio.sleep(0.3)

                # Check sink's VAD first - it's faster than SpeakingDetector
                # because it's already processing the buffer in listening state
                if self.sink.is_speech_active():
                    logger.info("[ACK] Skipping - sink VAD detected active speech")
                    return

                # Also check SpeakingDetector for multi-user scenarios
                silence = await self._wait_for_silence(max_wait=5.0, use_debounce=False)
                if not silence:
                    # User still speaking after 5s - they're asking a long question
                    # Skip the ack, they'll hear the response later
                    logger.info("[ACK] Skipping - user still speaking after 5s")
                    return

            t0 = time.perf_counter()
            ack_phrase = get_random_ack()
            self._ack_played = True  # Mark early - prevents race with fast transcription
            ack_audio = await self.tts.synthesize_cached(ack_phrase)
            synth_time = time.perf_counter() - t0
            logger.info(f"[TIMING] Ack TTS '{ack_phrase}': {synth_time:.2f}s")

            if len(ack_audio) > 0:
                # Check again after synthesis - user may have continued speaking
                # during the ~800ms synthesis time. This catches continuous questions
                # that weren't detected in the initial 300ms check.
                if self.settings.speaking_detection_enabled and self.sink.is_speech_active():
                    logger.info("[ACK] Skipping playback - VAD detected speech during synthesis")
                    return

                t0 = time.perf_counter()
                await self._play_audio(ack_audio)
                play_time = time.perf_counter() - t0
                logger.info(f"[TIMING] Ack playback: {play_time:.2f}s")
        except asyncio.CancelledError:
            # Let ack continue playing - it will be stopped naturally when
            # the next audio (thinking phrase or response) starts playing.
            # This prevents the "brief chirp" issue where ack is cut off.
            logger.debug("Ack task cancelled, letting audio finish naturally")
        except Exception as e:
            logger.warning(f"Failed to play acknowledgment: {e}")

    async def _handle_wake_word_only(self, opening: Optional[OpeningPhrase] = None):
        """Handle case where user only said the wake word.

        Plays acknowledgment (if not already played) and listens for follow-up question.
        If user asks something, processes it. If silence, returns to idle.

        Args:
            opening: Phrase started at speech end; if already audible it serves as the ack
        """
        try:
            # The affirmation started at speech end may already be playing; let it
            # stand in for the ack rather than talking over it
            opening_heard = (opening is not None and not opening.cancel_if_not_playing()
                             and opening.playing)

            # Check how long we were listening - if ack played and user had time to respond, skip follow-up
            listen_duration = time.perf_counter() - self._listen_start
            if self._ack_played and listen_duration > 5.0:
                logger.info(f"[WAKE-ONLY] Ack played and listened for {listen_duration:.1f}s - user had time, returning to idle")
                self.state = "idle"
                return

            if opening_heard:
                logger.info("[WAKE-ONLY] Opening phrase already playing, using it as the ack")
                with contextlib.suppress(asyncio.CancelledError):
                    await opening.wait()
            # Only play ack if one isn't already in progress
            # (_ack_played is set early in _play_ack to prevent race conditions)
            elif not self._ack_played:
                ack_phrase = get_random_ack()
                logger.info(f"[WAKE-ONLY] Playing ack: '{ack_phrase}'")
                ack_audio = await self.tts.synthesize_cached(ack_phrase)
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

    async def greet_user(self, member: discord.Member) -> None:
        """Greet a user who just joined the voice channel.

        Delegates to GreetingService which handles:
        - Cooldown checking
        - State validation
        - Silence detection
        - Greeting generation and playback

        Args:
            member: Discord member who joined
        """
        await self._greeting_service.greet(
            member=member,
            get_current_state=lambda: self.state,
            wait_for_silence=lambda: self._wait_for_silence(
                max_wait=self.settings.greeting_timeout_seconds,
                use_debounce=False,
                poll_interval=0.1,
            ),
            play_audio=self._play_audio,
        )

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

        clip = self.turn_observer.clip_started(audio) if self.turn_observer else None
        self.voice_client.play(source, after=after_playback)

        # Wait for playback to complete
        try:
            await asyncio.wait_for(
                playback_done.wait(), timeout=audio_duration + PLAYBACK_CALLBACK_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[PLAYBACK] No end-of-playback callback for a {audio_duration:.1f}s clip "
                f"after {audio_duration + PLAYBACK_CALLBACK_GRACE_SECONDS:.1f}s; moving on"
            )
            if self.voice_client.is_playing():
                self.voice_client.stop()
        if self.turn_observer:
            self.turn_observer.clip_ended(clip)

        # Clean up source and natural pause between sentences
        source.cleanup()
        await asyncio.sleep(0.15)  # 150ms gap for natural sentence pacing

    async def _process_speech(
        self, audio: np.ndarray, followup: Optional[FollowUp] = None
    ) -> Optional[str]:
        """Process captured speech using the modular VoicePipeline.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
            followup: Set when this speech came without the wake word right after the
                bot spoke (no opening phrase; dropped unless it was for the bot)

        Returns:
            The pipeline result (None for a normal reply)
        """
        self.state = "processing"
        total_start = time.perf_counter()
        restarted = False

        while True:
            # Log audio stats
            audio_duration = len(audio) / SAMPLE_RATE_PROCESSING
            logger.info(f"[TIMING] Audio captured: {audio_duration:.2f}s ({len(audio)} samples)")

            monitor_token = None
            if self.turn_observer:
                monitor_token = self.turn_observer.turn_started(
                    self._current_speaker_id, self._current_speaker_name,
                    speech_end=self.sink.last_audio_time if self.sink else None,
                )
                question_audio = getattr(self.turn_observer, "question_audio", None)
                if question_audio is not None:
                    question_audio(audio)
            if restarted:
                # Inside the new turn so the monitor reports it
                logger.warning(
                    f"[CARRY-ON] {self._current_speaker_name} kept talking after the question "
                    f"seemed finished; started over with all of it ({audio_duration:.1f}s)"
                )
            error: Optional[BaseException] = None
            result: Optional[str] = None
            try:
                result = await self._process_speech_with_pipeline(audio, followup, restarted)
            except BaseException as e:
                error = e
                raise
            finally:
                if self.turn_observer:
                    self.turn_observer.turn_ended(monitor_token, error)
            if result != CARRY_ON:
                break
            audio, self._carry_on_audio = self._carry_on_audio, None
            restarted = True

        total_time = time.perf_counter() - total_start
        logger.info(f"[TIMING] === TOTAL: {total_time:.2f}s ===")
        return result

    async def _process_speech_with_pipeline(
        self, audio: np.ndarray, followup: Optional[FollowUp] = None, restarted: bool = False
    ) -> Optional[str]:
        """Process speech using the new modular VoicePipeline.

        This method delegates to the VoicePipeline which handles:
        - Transcription with wake word stripping
        - LLM response generation
        - Parallel TTS synthesis with ordered playback

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
        """
        # Build pipeline context with speaker info
        context = PipelineContext(
            speaker_id=self._current_speaker_id,
            speaker_name=self._current_speaker_name,
            speaker_activity=self._current_speaker_activity,
            channel_activities=self._channel_activities(),
        )

        # Opening phrase: dynamic mode says the affirmation ("Sure, {name}...") and
        # the generator adds a content phrase later; static mode says a full
        # thinking phrase. Either starts now, while transcription runs, unless
        # someone is still talking.
        thinking_generator: Optional[ThinkingPhraseGenerator] = self._thinking_generator
        if thinking_generator:
            opening_text = get_affirmation_phrase(self._current_speaker_name)
        else:
            opening_text = get_thinking_phrase(self._current_speaker_name)

        async def room_is_quiet() -> bool:
            if not self.settings.speaking_detection_enabled:
                return True
            return await self._wait_for_silence(max_wait=3.0, use_debounce=False)

        opening: Optional[OpeningPhrase] = None
        accept = None
        if followup is None and not restarted:
            opening = OpeningPhrase(
                synthesize=lambda: self.tts.synthesize_cached(opening_text),
                play_audio=self._play_audio,
                should_play=room_is_quiet,
            )
            logger.debug(f"[PIPELINE] Opening phrase started: {opening_text}")
        elif followup is not None:
            # Carrying on without the wake word: no "Sure, Darren", and only if it was for us
            async def accept(transcription: TranscriptionResult) -> bool:
                return await self._is_addressed(followup, transcription)
        # (started over: "Sure, Darren" was already said)

        # Run the pipeline, still listening to the speaker: if they carry on before the
        # answer starts, start over with everything they said
        process = asyncio.create_task(self._voice_pipeline.process(
            audio=audio,
            context=context,
            play_audio=self._play_audio,
            wait_for_silence=lambda: self._wait_for_silence(max_wait=3.0, use_debounce=True),
            thinking_phrase=None if thinking_generator else opening_text,
            thinking_generator=thinking_generator,
            tts_synthesizer=self.tts if thinking_generator else None,
            opening=opening,
            accept=accept,
        ))
        if await self._carried_on(process, audio):
            if self.turn_observer:
                self.turn_observer.pipeline_result("cancelled")
            return CARRY_ON
        result = process.result()
        if self.turn_observer:
            self.turn_observer.pipeline_result(result)
        if (result is None and self.sink is not None
                and self.sink.captured_speech() >= CARRY_ON_MIN_SPEECH_SECONDS):
            # Not a problem by itself (often "yeah" while listening): the follow-up
            # judge decides whether it needs a reply, so the monitor doesn't post it
            logger.info(
                f"[CARRY-ON] {self._current_speaker_name} talked during the answer "
                f"({self.sink.captured_speech():.1f}s); keeping it for their follow-up"
            )

        # Handle special results
        if result == "wake_word_only":
            logger.info("[PIPELINE] Wake word only detected - triggering follow-up flow")
            await self._handle_wake_word_only(opening)
        elif result == "cancelled":
            logger.info("[PIPELINE] Processing was cancelled")
        elif result == "empty":
            logger.info("[PIPELINE] No speech detected")
        elif result == "not_addressed":
            logger.info("[PIPELINE] Follow-up wasn't for the bot, staying quiet")
        else:
            logger.debug("[PIPELINE] Processing completed normally")
        return result

    async def _carried_on(self, process: asyncio.Task, said: np.ndarray) -> bool:
        """Wait for the pipeline while keeping the speaker's audio. If they say more
        before the answer starts playing (an "um... so..." pause ended their turn),
        cancel it, hear them out, and return True (_carry_on_audio holds all of it).
        Speech while the bot plays a filler doesn't count: it may be the bot's own
        voice coming back through their mic."""
        speaker = self._current_speaker_id
        if self.sink is None or speaker is None:
            await asyncio.wait({process})
            return False
        self.sink.start_capture(speaker)
        checked = 0.0  # speaking detector's count at the last Silero check
        try:
            while not process.done():
                await asyncio.wait({process}, timeout=0.1)
                if process.done() or self._voice_pipeline.answer_started:
                    continue
                voiced = self.sink.captured_speech(while_bot_quiet=True)
                if voiced < max(CARRY_ON_MIN_SPEECH_SECONDS, checked + 0.2):
                    continue
                checked = voiced
                if (await self._speech_confirmed(while_bot_quiet=True)
                        and not process.done() and not self._voice_pipeline.answer_started):
                    break
            else:
                return False
        except asyncio.CancelledError:
            process.cancel()
            raise

        logger.info(f"[CARRY-ON] {self._current_speaker_name} is still talking, starting over")
        process.cancel()
        await asyncio.wait({process})
        await self._voice_pipeline.cancel()
        # The LLM may have finished before the answer played: nobody heard it
        forget = getattr(self._voice_pipeline, "forget_last_turn", None)
        if forget is not None and forget():
            logger.info("[CARRY-ON] Dropped the unheard answer from history")
        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.stop()
        self._carry_on_audio = await self._hear_the_rest(speaker, said)
        return True

    def _channel_activities(self) -> list[str]:
        """What the others in the voice channel are doing (for the game notes)."""
        manager = getattr(self, "speaker_manager", None)
        if manager is None:
            return []
        try:
            return manager.channel_activities(getattr(self.voice_client, "channel", None),
                                              self._current_speaker_id)
        except Exception as e:
            logger.debug(f"Could not read channel activities: {e}")
            return []

    async def _speech_confirmed(self, while_bot_quiet: bool) -> bool:
        """The capture holds at least CARRY_ON_MIN_SPEECH_SECONDS of speech by Silero.
        The speaking detector's count isn't enough on its own: clicks or game sounds
        through someone's mic come in short bursts that it holds as speech for 300ms
        each (2026-10-15: Pat's question restarted twice with no new words)."""
        audio = self.sink.captured_audio(while_bot_quiet=while_bot_quiet)
        async with self._speech_check_lock:
            try:
                if self._speech_check_vad is None:
                    self._speech_check_vad = await asyncio.to_thread(VAD)
                seconds = await asyncio.to_thread(self._speech_check_vad.speech_seconds, audio)
            except Exception as e:
                logger.warning(f"[CARRY-ON] Speech check failed, trusting the detector: {e}")
                return True
        if seconds is None:
            return True  # no model: trust the speaking detector
        if seconds < CARRY_ON_MIN_SPEECH_SECONDS:
            logger.info(f"[CARRY-ON] Not speech ({seconds:.2f}s by VAD of "
                        f"{len(audio) / SAMPLE_RATE_PROCESSING:.1f}s captured), ignoring")
            return False
        return True

    async def _hear_the_rest(self, speaker_id: int, said: np.ndarray) -> np.ndarray:
        """Listen to the speaker until they finish. Returns all of it, `said` first."""
        self.state = "listening"
        self._speech_end_event.clear()
        self._pending_audio = None
        self.sink.start_listening(user_id=speaker_id, prefix=said)
        deadline = time.perf_counter() + self.settings.max_question_duration
        audio: Optional[np.ndarray] = None
        while self._running and time.perf_counter() < deadline:
            try:
                await asyncio.wait_for(self._speech_end_event.wait(), timeout=0.2)
                audio = self._pending_audio
                break
            except asyncio.TimeoutError:
                pass
            if self.sink.check_audio_timeout():
                break
        if audio is None:
            audio = self.sink.get_audio_buffer()
        self.sink.stop_listening()
        self.state = "processing"
        return audio if len(audio) > 0 else said

    def _others_talking(self, speaker_id: Optional[int]) -> bool:
        """Someone other than speaker_id is talking right now."""
        detector = self.speaking_detector
        if detector is None or not self.settings.speaking_detection_enabled:
            return False
        # The audio thread adds people to the detector as they start talking
        with self.sink.lock if self.sink is not None else contextlib.nullcontext():
            users = detector.get_speaking_users()
        return any(user != speaker_id for user in users)

    def _watch_for_late_reply(self, speaker_id: Optional[int], question: str) -> None:
        """The bot asked a question and the reply window closed unanswered. If the person
        starts talking within a few seconds anyway, the window was probably too short:
        tell the monitor. (Measuring only; their next turn doesn't start with this.)"""
        if self.sink is None or speaker_id is None or self.turn_observer is None:
            return
        name = self._current_speaker_name
        others = self._followup_others_talked
        self.sink.start_capture(speaker_id, fold=False)

        async def check() -> None:
            await asyncio.sleep(LATE_REPLY_WATCH_SECONDS)
            if not self.sink.is_capturing(speaker_id, fold=False):
                return  # a new turn started (wake word): not a missed reply
            said = self.sink.captured_speech(while_bot_quiet=True)
            spoke = (said >= CARRY_ON_MIN_SPEECH_SECONDS
                     and await self._speech_confirmed(while_bot_quiet=True))
            if self.sink.is_capturing(speaker_id, fold=False):
                self.sink.stop_capture()
            if not spoke:
                return
            note = (
                f"{name} started talking within {LATE_REPLY_WATCH_SECONDS:.0f}s after the "
                f"window to answer the bot's question closed ({said:.1f}s of speech, no wake "
                f"word), so it was probably their reply. Others talked {others:.1f}s while "
                "the window was open."
            )
            logger.warning(f"[FOLLOWUP] Late reply: {note}")
            report = getattr(self.turn_observer, "incident", None)
            if report is not None:
                report(speaker_id, name, "late_reply", note, reply=question)

        if self._late_reply_task is not None:
            self._late_reply_task.cancel()
        self._late_reply_task = asyncio.create_task(check())

    async def _followup_after(self, result: Optional[str]) -> None:
        """After the bot speaks, the person it answered can carry on without the wake
        word: a short window, only their audio, and only if the addressee check says
        they were talking to the bot (after a plain answer: asking it something new)."""
        if not self.settings.followup_enabled or self._addressee_judge is None:
            return
        speaker_id = self._current_speaker_id
        reopened = False
        while self._running and result is None:
            reply = self._last_reply()
            if not reply:
                return
            followup = FollowUp(bot_said=reply, bot_asked=is_question(reply))
            if followup.bot_asked:
                window = self.settings.followup_window_seconds
            elif self.settings.followup_after_answer:
                window = self.settings.followup_after_answer_seconds
            else:
                return
            audio = await self._listen_for_followup(speaker_id, window, followup.bot_asked)
            if audio is None:
                if followup.bot_asked and self._wake_from_followup is None:
                    self._watch_for_late_reply(speaker_id, reply)
                return
            said_during_answer = getattr(self, "_followup_said_during_answer", False)
            result = await self._process_speech(audio, followup=followup)
            if result == "not_addressed" and said_during_answer and not reopened:
                # 2026-10-15: Pat's "mm-hmm" during the answer was all the window
                # heard, so his real follow-up had no window left. Give it one.
                logger.info("[FOLLOWUP] What they said during the answer wasn't for the "
                            "bot; listening again for their reply")
                reopened = True
                result = None

    def _last_reply(self) -> Optional[str]:
        """What the bot just said (tags removed), if the turn finished normally."""
        response = getattr(self._voice_pipeline, "_response", None)
        turn = getattr(response, "last_turn", None)
        if turn is None or not turn.ok:
            return None
        return re.sub(r"\[[^\]]*\]\s*", "", turn.text).strip()

    async def _listen_for_followup(
        self, speaker_id: Optional[int], window: float, bot_asked: bool
    ) -> Optional[np.ndarray]:
        """Listen to one speaker for `window` seconds of quiet. Returns their speech, or
        None if they didn't start talking in time. Speech already under way may finish
        (including anything they said during the answer).

        The window only counts down while nobody else is talking, and when someone else
        stops, the speaker gets at least followup_grace_seconds: people wait for a gap
        before replying rather than jumping in at the first pause."""
        logger.info(
            f"[FOLLOWUP] Bot {'asked a question' if bot_asked else 'answered'}; listening "
            f"{window:.0f}s for {self._current_speaker_name} without the wake word"
        )
        self.state = "listening"
        self._speech_end_event.clear()
        self._pending_audio = None
        self._wake_from_followup = None
        self._followup_open = True
        self.sink.start_listening(user_id=speaker_id, watch_wake_word=True)
        self._followup_said_during_answer = bool(getattr(self.sink, "seed_finished", False))

        start = last = time.perf_counter()
        give_up = start + self.settings.followup_max_wait_seconds
        quiet_left = window
        others_talking = False
        others_talked = 0.0
        speech_since: Optional[float] = None
        audio: Optional[np.ndarray] = None
        while self._running:
            try:
                await asyncio.wait_for(self._speech_end_event.wait(), timeout=0.2)
                audio = self._pending_audio
                break
            except asyncio.TimeoutError:
                pass
            if self.sink.check_audio_timeout():  # push-to-talk released
                audio = self.sink.get_audio_buffer()
                break
            if self._wake_from_followup is not None:  # someone else's turn
                break
            now = time.perf_counter()
            step, last = now - last, now
            if self.sink.is_speech_active():
                speech_since = speech_since or now
                if now - speech_since >= self.settings.max_question_duration:
                    audio = self.sink.get_audio_buffer()
                    break
                continue
            if self._others_talking(speaker_id):
                others_talking = True
                others_talked += step
            else:
                if others_talking:
                    # Grace counts from here (this step was mostly them still talking)
                    others_talking = False
                    quiet_left = max(quiet_left, self.settings.followup_grace_seconds)
                else:
                    quiet_left -= step
                if quiet_left <= 0:
                    break
            if now >= give_up:
                break
        self._followup_others_talked = others_talked

        self._followup_open = False
        self.sink.stop_listening()
        self.state = "idle"
        if self._wake_from_followup is not None:
            return None  # the main loop starts their turn
        if audio is None or len(audio) == 0:
            logger.info(
                "[FOLLOWUP] No reply, back to waiting for the wake word"
                + (f" (others talked {others_talked:.1f}s)" if others_talked else "")
            )
            return None
        return audio

    async def _is_addressed(
        self, followup: FollowUp, transcription: TranscriptionResult
    ) -> bool:
        """Follow-up gate: was this speech meant for the bot?"""
        started = time.perf_counter()
        said = (transcription.raw_text or transcription.text).strip()
        verdict = await self._addressee_judge.judge(followup, said, self._current_speaker_name)
        logger.info(
            f"[FOLLOWUP] {'For the bot' if verdict.addressed else 'Not for the bot'} "
            f"({verdict.reason}, {time.perf_counter() - started:.2f}s): {said[:80]}"
        )
        return verdict.addressed
