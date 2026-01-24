"""Voice session manager for Discord voice assistant.

Manages the voice pipeline state machine and coordinates all components
for a single Discord voice channel.
"""

import asyncio
import logging
import random
import time
from typing import Optional

import discord
import numpy as np

from ..config import get_settings
from ..detection import VAD, WakeWordDetector
from ..stt import Transcriber, DeepgramFluxClient, GroqWhisperClient
from ..llm import ClaudeClient
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

    # Thinking phrases (randomly selected)
    THINKING_PHRASES = [
        "One second...",
        "One moment...",
        "Let me check...",
        "Hmm, let me see...",
        "Just a moment...",
    ]

    def __init__(
        self,
        *,
        wake_detector: Optional[WakeWordDetector] = None,
        vad: Optional[VAD] = None,
        transcriber: Optional[Transcriber] = None,
        groq_client: Optional[GroqWhisperClient] = None,
        deepgram_client: Optional[DeepgramFluxClient] = None,
        claude: Optional[ClaudeClient] = None,
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
            claude: ClaudeClient instance
            tts: TTSSynthesizer instance
        """
        self.settings = get_settings()

        # Components (will be initialized if not provided)
        self.wake_detector = wake_detector
        self.vad = vad
        self.transcriber = transcriber
        self.groq_client = groq_client
        self.deepgram_client = deepgram_client
        self.claude = claude
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

        if self.claude is None:
            logger.info("Setting up Claude client...")
            self.claude = ClaudeClient()

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

    def _on_speech_end(self, audio: np.ndarray):
        """Callback when speech ends (from sink thread)."""
        audio_duration = len(audio) / 16000 if audio is not None else 0
        logger.info(f"[CALLBACK] _on_speech_end called with {audio_duration:.2f}s audio, state={self._state}")
        if self._state == "listening" and self._loop:
            self._pending_audio = audio
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

                # Wait for wake word
                logger.debug("Waiting for wake word...")
                idle_start = time.perf_counter()
                await self._wake_event.wait()
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
                    while time.perf_counter() < deadline:
                        # Check if VAD detected speech end
                        try:
                            await asyncio.wait_for(
                                self._speech_end_event.wait(),
                                timeout=0.2  # Poll every 200ms
                            )
                            # Event was set - speech ended via VAD
                            break
                        except asyncio.TimeoutError:
                            pass

                        # Check if audio stopped (push-to-talk released)
                        if self.sink.check_audio_timeout():
                            logger.info("[STATE] Audio stopped (PTT released)")
                            self._pending_audio = self.sink.get_audio_buffer()
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
            ack_audio = await self.tts.synthesize("Yes?")
            synth_time = time.perf_counter() - t0
            logger.info(f"[TIMING] Ack TTS: {synth_time:.2f}s")

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

        # Clean up source and give Discord a moment to clear speaking state
        source.cleanup()
        await asyncio.sleep(0.05)  # 50ms gap between audio

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

        logger.info(f"User said: {text}")

        # Play thinking phrase
        t0 = time.perf_counter()
        phrase = random.choice(self.THINKING_PHRASES)
        logger.info(f"[TIMING] Synthesizing thinking phrase: '{phrase}'")
        thinking_audio = await self.tts.synthesize(phrase)
        thinking_synth_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Thinking phrase TTS: {thinking_synth_time:.2f}s")

        t0 = time.perf_counter()
        await self._play_audio(thinking_audio)
        thinking_play_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Thinking phrase playback: {thinking_play_time:.2f}s")

        # Get Claude response with sentence-level streaming TTS
        # This allows us to speak early sentences while waiting for web search results
        t0 = time.perf_counter()
        logger.info("[TIMING] Starting Claude request...")

        buffer = ""
        first_chunk_time = None
        sentences_played = 0
        total_response = ""

        async for chunk in self.claude.chat_stream(text):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
                logger.info(f"[TIMING] Claude first chunk: {first_chunk_time:.2f}s")

            buffer += chunk
            total_response += chunk

            # Claude outputs one sentence per line - split on newlines for streaming
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                if line:
                    sentences_played += 1
                    logger.info(f"[STREAMING] Playing line {sentences_played}: {line[:50]}...")

                    # Synthesize and play this line
                    line_audio = await self.tts.synthesize(line)
                    if len(line_audio) > 0:
                        await self._play_audio(line_audio)

            # Also play if buffer ends with punctuation (sentence complete, no newline yet)
            # This handles "I'll look that up." before a long tool call
            stripped = buffer.strip()
            if stripped and stripped[-1] in '.!?':
                sentences_played += 1
                logger.info(f"[STREAMING] Playing sentence {sentences_played}: {stripped[:50]}...")

                sentence_audio = await self.tts.synthesize(stripped)
                if len(sentence_audio) > 0:
                    await self._play_audio(sentence_audio)
                buffer = ""  # Clear the buffer since we played it

        # Play any remaining text
        remaining = buffer.strip()
        if remaining:
            logger.info(f"[STREAMING] Playing remaining: {remaining[:50]}...")
            remaining_audio = await self.tts.synthesize(remaining)
            if len(remaining_audio) > 0:
                await self._play_audio(remaining_audio)

        claude_total_time = time.perf_counter() - t0
        logger.info(f"[TIMING] Claude total: {claude_total_time:.2f}s ({len(total_response)} chars, {sentences_played} sentences streamed)")
        logger.info(f"Claude: {total_response}")

        # Total summary
        total_time = time.perf_counter() - total_start
        logger.info(
            f"[TIMING] === TOTAL: {total_time:.2f}s === "
            f"(transcribe={transcribe_time:.1f}s, "
            f"think_tts={thinking_synth_time:.1f}s, "
            f"claude+streaming_tts={claude_total_time:.1f}s)"
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
