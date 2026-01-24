"""Voice session manager for Discord voice assistant.

Manages the voice pipeline state machine and coordinates all components
for a single Discord voice channel.
"""

import asyncio
import io
import logging
import random
import time
from typing import Optional

import discord
import numpy as np

from ..config import get_settings
from ..detection import VAD, WakeWordDetector
from ..stt import Transcriber, DeepgramFluxClient, GroqWhisperClient
from ..llm import BaseLLMClient, create_llm_client
from ..tts import TTSSynthesizer
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

    # Discord user ID -> real name mapping (for personalized responses)
    USER_NAMES = {
        100000000000000001: "Darren",
        100000000000000003: "Pat",
        100000000000000002: "Sam",
        100000000000000004: "Jordan",
        # Add more user IDs here
    }

    # Acknowledgment phrases (played when wake word detected)
    ACK_PHRASES = [
        "Yes?",
        "Hmm?",
        "What's up?",
        "I'm listening...",
        "Go ahead...",
        "I'm busy sorry...",
        "Yea, no, yea. No.",
    ]

    # Thinking phrases (randomly selected)
    # Generic phrases (used when speaker unknown)
    THINKING_PHRASES = [
        "One second...",
        "One moment...",
        "Let me check...",
        "Google it yourself... Nah, I'll do it...",
        "Hmm, let me see...",
        "Just a moment...",
    ]

    # Personalized phrases with {name} placeholder (used when speaker is known)
    THINKING_PHRASES_PERSONAL = [
        "One second, {name}...",
        "One moment, {name}...",
        "Let me check, {name}...",
        "Sure {name}, I'll do all the work...",
        "Google it yourself, {name}... Nah, I'll do it...",
        "Hmm, let me see, {name}...",
        "Just a moment, {name}...",
        "Hold on, {name}...",
        "Give me a sec, {name}...",
    ]

    def __init__(
        self,
        *,
        wake_detector: Optional[WakeWordDetector] = None,
        vad: Optional[VAD] = None,
        transcriber: Optional[Transcriber] = None,
        groq_client: Optional[GroqWhisperClient] = None,
        deepgram_client: Optional[DeepgramFluxClient] = None,
        llm: Optional[BaseLLMClient] = None,
        tts: Optional[TTSSynthesizer] = None,
    ):
        """Initialize voice session.

        Components can be passed in (shared) or will be created.

        Args:
            wake_detector: WakeWordDetector instance
            vad: VAD instance
            transcriber: Transcriber instance (Whisper, fallback)
            groq_client: GroqWhisperClient instance (primary STT)
            deepgram_client: DeepgramFluxClient instance (alternative STT)
            llm: LLM client instance (Claude or Groq)
            tts: TTSSynthesizer instance
        """
        self.settings = get_settings()

        # Components (will be initialized if not provided)
        self.wake_detector = wake_detector
        self.vad = vad
        self.transcriber = transcriber
        self.groq_client = groq_client
        self.deepgram_client = deepgram_client
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

        # Current speaker tracking
        self._current_speaker_id: Optional[int] = None
        self._current_speaker_name: Optional[str] = None
        self._current_speaker_activity: Optional[str] = None

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
        """Initialize ML components if not provided."""
        if self.wake_detector is None:
            logger.info("Loading wake word detector...")
            self.wake_detector = WakeWordDetector(
                model_name=self.settings.wake_word_model,
                threshold=self.settings.wake_word_confidence,
                sample_rate=16000,  # We resample to 16kHz
            )

        if self.vad is None:
            logger.info("Loading VAD...")
            self.vad = VAD(
                threshold=self.settings.vad_sensitivity,
                sample_rate=16000,
                speech_pad_ms=self.settings.speech_pad_ms,
                min_silence_duration_ms=self.settings.min_silence_ms,
            )

        # Initialize STT based on provider setting
        stt_provider = self.settings.stt_provider.lower()

        if stt_provider == "groq" and self.groq_client is None:
            if not self.settings.groq_api_key:
                logger.warning("GROQ_API_KEY not set, falling back to Deepgram")
                stt_provider = "deepgram"
            else:
                logger.info("Initializing Groq Whisper STT...")
                self.groq_client = GroqWhisperClient(
                    api_key=self.settings.groq_api_key,
                    model="whisper-large-v3-turbo",
                    language="en",
                )

        if stt_provider == "deepgram" and self.deepgram_client is None:
            if not self.settings.deepgram_api_key:
                logger.warning("DEEPGRAM_API_KEY not set, falling back to Whisper")
                stt_provider = "whisper"
            else:
                logger.info("Initializing Deepgram Flux STT...")
                self.deepgram_client = DeepgramFluxClient(
                    api_key=self.settings.deepgram_api_key,
                    sample_rate=16000,
                    eot_threshold=0.7,  # End-of-turn confidence (0.5-0.9)
                    eot_timeout_ms=2000,  # Max silence before forcing EOT
                )

        if stt_provider == "whisper" and self.transcriber is None:
            logger.info("Loading Whisper...")
            self.transcriber = Transcriber(
                model_size=self.settings.whisper_model,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )

        if self.llm is None:
            logger.info("Setting up LLM client...")
            self.llm = create_llm_client(
                self.settings,
                on_image_generated=self._post_image_to_discord,
                on_user_tagged=self._post_tag_to_discord,
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

    async def _post_image_to_discord(self, image_bytes: bytes, prompt: str) -> str:
        """Post a generated image to Discord.

        Args:
            image_bytes: The generated image as PNG bytes
            prompt: The prompt used to generate the image

        Returns:
            Result message to send back to the LLM
        """
        logger.info(f"[IMAGE] _post_image_to_discord called with {len(image_bytes)} bytes")

        if not self.voice_client or not self.voice_client.guild:
            logger.warning("[IMAGE] Cannot post image - no voice client or guild")
            return "Could not post image - not connected to a server"

        channel_name = self.settings.image_channel_name
        logger.info(f"[IMAGE] Looking for channel: #{channel_name}")

        # List available channels for debugging
        available_channels = [ch.name for ch in self.voice_client.guild.text_channels]
        logger.info(f"[IMAGE] Available text channels: {available_channels}")

        channel = discord.utils.get(
            self.voice_client.guild.text_channels,
            name=channel_name,
        )

        if not channel:
            logger.warning(f"[IMAGE] Could not find channel #{channel_name}")
            return f"Could not find channel #{channel_name} to post the image. Available channels: {available_channels}"

        try:
            requester = self._current_speaker_name or "Someone"
            logger.info(f"[IMAGE] Sending to #{channel_name}...")
            await channel.send(
                content=f"*{requester}'s requested image:*",
                file=discord.File(io.BytesIO(image_bytes), filename="generated.png"),
            )
            logger.info(f"[IMAGE] Posted generated image to #{channel_name}")
            return f"Done! I've posted the image to #{channel_name}."
        except Exception as e:
            logger.error(f"[IMAGE] Failed to post image: {e}", exc_info=True)
            return f"Failed to post image: {str(e)}"

    async def _post_tag_to_discord(self, target_name: str, message: str) -> str:
        """Post a tagged message to Discord.

        Args:
            target_name: Name of user to tag (fuzzy matched)
            message: Message content

        Returns:
            Result message to send back to the LLM
        """
        logger.info(f"[TAG] _post_tag_to_discord called: target={target_name}, message={message[:50]}...")

        if not self.voice_client or not self.voice_client.guild:
            logger.warning("[TAG] Cannot tag user - no voice client or guild")
            return "Could not tag user - not connected to a server"

        result = self._find_user_by_name(target_name)
        if not result:
            known = list(self.USER_NAMES.values())
            logger.warning(f"[TAG] Could not find user '{target_name}'. Known: {known}")
            return f"Could not find user '{target_name}'. Known users: {', '.join(known)}"

        target_id, target_display_name = result
        logger.info(f"[TAG] Found user: {target_display_name} (ID: {target_id})")

        channel_name = self.settings.image_channel_name
        logger.info(f"[TAG] Looking for channel: #{channel_name}")

        channel = discord.utils.get(
            self.voice_client.guild.text_channels,
            name=channel_name,
        )

        if not channel:
            available = [ch.name for ch in self.voice_client.guild.text_channels]
            logger.warning(f"[TAG] Could not find channel #{channel_name}. Available: {available}")
            return f"Could not find channel #{channel_name}"

        requester = self._current_speaker_name or "Someone"
        # Use <@user_id> format for mention - works even if user is offline/uncached
        formatted = f"<@{target_id}> - {requester} says: {message}"

        try:
            await channel.send(formatted)
            logger.info(f"[TAG] Posted tag message to #{channel_name}")
            return f"Done! I've tagged {target_display_name} in {channel_name} chat."
        except Exception as e:
            logger.error(f"[TAG] Failed to post tag: {e}", exc_info=True)
            return f"Failed to send message: {str(e)}"

    def _get_member_activity(self, member: discord.Member) -> Optional[str]:
        """Get the member's current activity (game, etc.)."""
        if not member or not member.activities:
            return None
        for activity in member.activities:
            if isinstance(activity, discord.Game):
                return f"Playing {activity.name}"
            elif isinstance(activity, discord.Activity):
                # activity.type is an enum (playing, streaming, listening, watching, etc.)
                return f"{activity.type.name.title()} {activity.name}"
        return None

    def _find_user_by_name(self, name: str) -> Optional[tuple[int, str]]:
        """Find Discord user by fuzzy name match.

        Args:
            name: Name to search for (case-insensitive, partial match)

        Returns:
            Tuple of (user_id, display_name) if found, None otherwise
        """
        if not self.voice_client or not self.voice_client.guild:
            return None

        guild = self.voice_client.guild
        name_lower = name.lower().strip()

        # Exact match in USER_NAMES (works even if user is offline)
        for user_id, friendly_name in self.USER_NAMES.items():
            if friendly_name.lower() == name_lower:
                return (user_id, friendly_name)

        # Partial match (e.g., "sa" matches "Sam")
        for user_id, friendly_name in self.USER_NAMES.items():
            if name_lower in friendly_name.lower():
                return (user_id, friendly_name)

        # Fallback: search cached guild members by display name
        for member in guild.members:
            if name_lower in member.display_name.lower():
                return (member.id, member.display_name)

        return None

    def _on_speech_end(self, audio: np.ndarray, user_id: Optional[int] = None):
        """Callback when speech ends (from sink thread)."""
        audio_duration = len(audio) / 16000 if audio is not None else 0
        logger.info(f"[CALLBACK] _on_speech_end called with {audio_duration:.2f}s audio from user {user_id}, state={self._state}")
        if self._state == "listening" and self._loop:
            self._pending_audio = audio
            # Store speaker info
            self._current_speaker_id = user_id
            self._current_speaker_activity = None
            if user_id and user_id in self.USER_NAMES:
                self._current_speaker_name = self.USER_NAMES[user_id]
                # Still get activity even for mapped users
                if self.voice_client:
                    try:
                        member = self.voice_client.guild.get_member(user_id)
                        self._current_speaker_activity = self._get_member_activity(member)
                    except Exception:
                        pass
            elif user_id and self.voice_client:
                # Fallback to Discord display name for unmapped users
                try:
                    member = self.voice_client.guild.get_member(user_id)
                    self._current_speaker_name = member.display_name if member else None
                    self._current_speaker_activity = self._get_member_activity(member)
                except Exception:
                    self._current_speaker_name = None
            else:
                self._current_speaker_name = None
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

        # Create the streaming sink
        self.sink = StreamingSink(
            wake_detector=self.wake_detector,
            vad=self.vad,
            on_wake_word=self._on_wake_word,
            on_speech_end=self._on_speech_end,
            max_speech_duration=self.settings.max_question_duration,
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

        if self.voice_client and self.voice_client.recording:
            self.voice_client.stop_recording()

        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.stop()

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
                listen_start = time.perf_counter()
                self.state = "listening"
                self._speech_started = False
                self.sink.start_listening()
                logger.info("[STATE] Now LISTENING for user speech (10s timeout)")

                # Play acknowledgment in background
                ack_task = asyncio.create_task(self._play_ack())

                # Wait for speech to end (with timeout)
                # Poll for both VAD end AND push-to-talk audio stop
                try:
                    logger.info("[STATE] Waiting for speech_end_event...")
                    deadline = time.perf_counter() + 10.0
                    speech_end_reason = None  # Track why speech ended
                    while time.perf_counter() < deadline:
                        # Check if VAD detected speech end
                        try:
                            await asyncio.wait_for(
                                self._speech_end_event.wait(),
                                timeout=0.2  # Poll every 200ms
                            )
                            # Event was set - speech ended via VAD
                            buffer_duration = len(self.sink.get_audio_buffer()) / 16000
                            speech_end_reason = f"VAD detected silence (buffer={buffer_duration:.2f}s)"
                            logger.info(f"[SPEECH_END] {speech_end_reason}")
                            break
                        except asyncio.TimeoutError:
                            pass

                        # Check if audio stopped (push-to-talk released)
                        if self.sink.check_audio_timeout():
                            self._pending_audio = self.sink.get_audio_buffer()
                            buffer_duration = len(self._pending_audio) / 16000 if self._pending_audio is not None else 0
                            speech_end_reason = f"PTT released / no audio packets (buffer={buffer_duration:.2f}s)"
                            logger.info(f"[SPEECH_END] {speech_end_reason}")
                            # Capture speaker info (same logic as _on_speech_end)
                            user_id = self.sink._last_user_id
                            self._current_speaker_id = user_id
                            self._current_speaker_activity = None
                            if user_id and user_id in self.USER_NAMES:
                                self._current_speaker_name = self.USER_NAMES[user_id]
                                # Still get activity even for mapped users
                                if self.voice_client:
                                    try:
                                        member = self.voice_client.guild.get_member(user_id)
                                        self._current_speaker_activity = self._get_member_activity(member)
                                    except Exception:
                                        pass
                            elif user_id and self.voice_client:
                                try:
                                    member = self.voice_client.guild.get_member(user_id)
                                    self._current_speaker_name = member.display_name if member else None
                                    self._current_speaker_activity = self._get_member_activity(member)
                                except Exception:
                                    self._current_speaker_name = None
                            else:
                                self._current_speaker_name = None
                            logger.info(f"[STATE] Speaker: {self._current_speaker_name} (ID: {user_id}), Activity: {self._current_speaker_activity}")
                            break
                    else:
                        # Loop completed without break = timeout
                        listen_time = time.perf_counter() - listen_start
                        logger.warning(f"[STATE] TIMEOUT! No speech detected in {listen_time:.2f}s - returning to idle")
                        ack_task.cancel()
                        continue

                    listen_time = time.perf_counter() - listen_start
                    logger.info(f"[TIMING] Speech captured in {listen_time:.2f}s")
                except asyncio.CancelledError:
                    raise

                # Cancel ack if still running
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

    async def _play_ack(self):
        """Play acknowledgment audio."""
        try:
            t0 = time.perf_counter()
            ack_phrase = random.choice(self.ACK_PHRASES)
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

    async def _play_audio(self, audio: np.ndarray):
        """Play audio through Discord voice.

        Args:
            audio: Audio as float32 numpy array (24kHz mono)
        """
        if not self.voice_client or len(audio) == 0:
            logger.debug("[TIMING] _play_audio: no audio or no voice client")
            return

        audio_duration = len(audio) / 24000  # 24kHz
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
        """
        logger.debug("[PLAYBACK] Consumer started")
        next_to_play = 1
        pending: dict[int, np.ndarray] = {}

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
        """Synthesize text and add to audio queue with ordering info."""
        try:
            t0 = time.perf_counter()
            audio = await self.tts.synthesize(text)
            elapsed = time.perf_counter() - t0
            logger.debug(f"[TTS] Sentence {sentence_num} synthesized in {elapsed:.2f}s")

            # Queue with sentence number for ordered playback
            await self._audio_queue.put((sentence_num, audio))
        except Exception as e:
            logger.error(f"[TTS] Error synthesizing sentence {sentence_num}: {e}")
            # Queue empty audio to not block ordering
            await self._audio_queue.put((sentence_num, np.array([], dtype=np.float32)))

    async def _process_speech(self, audio: np.ndarray):
        """Process captured speech.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
        """
        self.state = "processing"
        total_start = time.perf_counter()

        # Log audio stats
        audio_duration = len(audio) / 16000  # 16kHz sample rate
        logger.info(f"[TIMING] Audio captured: {audio_duration:.2f}s ({len(audio)} samples)")

        # Transcribe - use Deepgram Flux if available, otherwise Whisper
        t0 = time.perf_counter()
        logger.info("[TIMING] Starting transcription...")

        if self.groq_client is not None:
            logger.info("[STT] Using Groq Whisper")
            text = await self.groq_client.transcribe_audio(audio)
        elif self.deepgram_client is not None:
            logger.info("[STT] Using Deepgram Flux")
            text = await self.deepgram_client.transcribe_audio(audio)
        elif self.transcriber is not None:
            logger.info("[STT] Using Whisper")
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                self.transcriber.transcribe,
                audio,
            )
        else:
            logger.error("No STT provider available!")
            return

        transcribe_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Transcription: {transcribe_time:.2f}s")

        if not text.strip():
            logger.info("No speech detected")
            return

        # Include speaker name if known
        speaker = self._current_speaker_name
        activity = self._current_speaker_activity
        if speaker:
            logger.info(f"{speaker} said: {text}" + (f" (Activity: {activity})" if activity else ""))
            # Prepend speaker info and activity so Claude knows who's talking and what they're doing
            if activity:
                prompt_text = f"[{speaker} speaking, currently {activity}]: {text}"
            else:
                prompt_text = f"[{speaker} speaking]: {text}"
        else:
            logger.info(f"User said: {text}")
            prompt_text = text

        # Play thinking phrase (personalized if we know who's speaking)
        t0 = time.perf_counter()
        if speaker:
            phrase = random.choice(self.THINKING_PHRASES_PERSONAL).format(name=speaker)
        else:
            phrase = random.choice(self.THINKING_PHRASES)
        logger.info(f"[TIMING] Synthesizing thinking phrase: '{phrase}'")
        thinking_audio = await self.tts.synthesize(phrase)
        thinking_synth_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Thinking phrase TTS: {thinking_synth_time:.2f}s")

        t0 = time.perf_counter()
        await self._play_audio(thinking_audio)
        thinking_play_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Thinking phrase playback: {thinking_play_time:.2f}s")

        # Get LLM response with PARALLEL sentence-level streaming TTS
        # TTS runs in parallel: synthesize next sentence while current plays
        t0 = time.perf_counter()
        logger.info("[TIMING] Starting LLM request...")

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

        # Wait for all TTS tasks to complete (audio is now in queue)
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
            self._tts_tasks.clear()

        # Signal end of stream and wait for playback to finish
        await self._audio_queue.put(None)
        if self._playback_task:
            await self._playback_task
            self._playback_task = None

        llm_total_time = time.perf_counter() - t0
        logger.info(f"[TIMING] LLM total: {llm_total_time:.2f}s ({len(total_response)} chars, {sentences_queued} sentences streamed)")
        logger.info(f"LLM response: {total_response}")

        # Total summary
        total_time = time.perf_counter() - total_start
        logger.info(
            f"[TIMING] === TOTAL: {total_time:.2f}s === "
            f"(transcribe={transcribe_time:.1f}s, "
            f"think_tts={thinking_synth_time:.1f}s, "
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
