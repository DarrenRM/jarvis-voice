"""Deepgram Flux STT client for real-time transcription with turn detection.

Flux is a conversational speech recognition model that automatically detects
when the user has finished speaking (end-of-turn), eliminating the need for
separate VAD.
"""

import asyncio
import logging
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class DeepgramFluxClient:
    """Streaming STT client using Deepgram Flux.

    Features:
    - Real-time streaming transcription
    - Automatic end-of-turn detection (~260ms latency)
    - No separate VAD needed
    """

    def __init__(
        self,
        api_key: str,
        *,
        eot_threshold: float = 0.7,
        eot_timeout_ms: int = 3000,
        sample_rate: int = 16000,
    ):
        """Initialize Deepgram Flux client.

        Args:
            api_key: Deepgram API key
            eot_threshold: End-of-turn confidence threshold (0.5-0.9)
            eot_timeout_ms: Max silence before forcing end-of-turn (500-10000ms)
            sample_rate: Audio sample rate (must be 16000 for Flux)
        """
        self.api_key = api_key
        self.eot_threshold = eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.sample_rate = sample_rate

        self._client = None
        self._connection = None
        self._transcript = ""
        self._end_of_turn_event = asyncio.Event()
        self._is_connected = False
        self._on_interim_transcript: Optional[Callable[[str], None]] = None

    async def connect(self) -> None:
        """Establish connection to Deepgram Flux."""
        try:
            from deepgram import DeepgramClient, DeepgramClientOptions

            # Create client with API key
            config = DeepgramClientOptions(api_key=self.api_key)
            self._client = DeepgramClient(config=config)

            logger.info("Deepgram Flux client initialized")
            self._is_connected = True

        except ImportError:
            logger.error("deepgram-sdk not installed. Install with: pip install deepgram-sdk")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Deepgram client: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection to Deepgram."""
        self._is_connected = False
        self._client = None
        logger.info("Deepgram client disconnected")

    async def transcribe_audio(
        self,
        audio: np.ndarray,
        on_interim: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Transcribe audio buffer using Flux streaming.

        This sends the audio to Flux and waits for end-of-turn detection.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
            on_interim: Optional callback for interim transcripts

        Returns:
            Final transcript text
        """
        if not self._is_connected or self._client is None:
            await self.connect()

        self._on_interim_transcript = on_interim
        self._transcript = ""
        self._end_of_turn_event.clear()

        try:
            from deepgram import LiveTranscriptionEvents, LiveOptions

            # Convert float32 to int16 bytes for Deepgram
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # Use live transcription with Flux model
            options = LiveOptions(
                model="flux-general-en",
                encoding="linear16",
                sample_rate=self.sample_rate,
                channels=1,
                # Flux-specific options would go here but SDK may not expose them yet
            )

            # Create live connection
            connection = self._client.listen.live.v("1")

            final_transcript = ""

            def on_message(self_conn, result, **kwargs):
                nonlocal final_transcript
                try:
                    transcript = result.channel.alternatives[0].transcript
                    if transcript:
                        if result.is_final:
                            final_transcript += transcript + " "
                            logger.debug(f"[FLUX] Final: {transcript}")
                        else:
                            logger.debug(f"[FLUX] Interim: {transcript}")
                            if self._on_interim_transcript:
                                self._on_interim_transcript(transcript)
                except Exception as e:
                    logger.error(f"Error processing Flux message: {e}")

            def on_error(self_conn, error, **kwargs):
                logger.error(f"Deepgram error: {error}")

            def on_close(self_conn, close, **kwargs):
                logger.debug("Deepgram connection closed")
                self._end_of_turn_event.set()

            # Register handlers
            connection.on(LiveTranscriptionEvents.Transcript, on_message)
            connection.on(LiveTranscriptionEvents.Error, on_error)
            connection.on(LiveTranscriptionEvents.Close, on_close)

            # Start connection
            if not await connection.start(options):
                logger.error("Failed to start Deepgram connection")
                return ""

            # Send audio in chunks (80ms recommended for Flux)
            chunk_samples = int(self.sample_rate * 0.08)  # 80ms chunks
            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i:i + chunk_samples]
                await connection.send(chunk.tobytes())
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.01)

            # Signal end of audio
            await connection.finish()

            # Wait for processing to complete
            await asyncio.sleep(0.5)

            return final_transcript.strip()

        except Exception as e:
            logger.error(f"Flux transcription error: {e}", exc_info=True)
            return ""

    async def start_streaming_session(
        self,
        on_transcript: Callable[[str, bool], None],
        on_end_of_turn: Callable[[str], None],
    ) -> "FluxStreamingSession":
        """Start a streaming transcription session.

        This is for real-time audio streaming where audio chunks are fed
        continuously and we get callbacks on transcripts and end-of-turn.

        Args:
            on_transcript: Callback(text, is_final) for transcript updates
            on_end_of_turn: Callback(final_text) when speaker finished

        Returns:
            FluxStreamingSession for sending audio
        """
        if not self._is_connected or self._client is None:
            await self.connect()

        return FluxStreamingSession(
            client=self._client,
            sample_rate=self.sample_rate,
            eot_threshold=self.eot_threshold,
            eot_timeout_ms=self.eot_timeout_ms,
            on_transcript=on_transcript,
            on_end_of_turn=on_end_of_turn,
        )


class FluxStreamingSession:
    """Active streaming session for continuous audio input."""

    def __init__(
        self,
        client,
        sample_rate: int,
        eot_threshold: float,
        eot_timeout_ms: int,
        on_transcript: Callable[[str, bool], None],
        on_end_of_turn: Callable[[str], None],
    ):
        self._client = client
        self._sample_rate = sample_rate
        self._eot_threshold = eot_threshold
        self._eot_timeout_ms = eot_timeout_ms
        self._on_transcript = on_transcript
        self._on_end_of_turn = on_end_of_turn
        self._connection = None
        self._is_active = False
        self._accumulated_transcript = ""

    async def start(self) -> bool:
        """Start the streaming session."""
        try:
            from deepgram import LiveTranscriptionEvents, LiveOptions

            options = LiveOptions(
                model="flux-general-en",
                encoding="linear16",
                sample_rate=self._sample_rate,
                channels=1,
                interim_results=True,
                endpointing=self._eot_timeout_ms,
            )

            self._connection = self._client.listen.live.v("1")

            def on_message(self_conn, result, **kwargs):
                try:
                    transcript = result.channel.alternatives[0].transcript
                    if transcript:
                        is_final = result.is_final
                        self._on_transcript(transcript, is_final)
                        if is_final:
                            self._accumulated_transcript += transcript + " "

                        # Check for speech_final (end of turn indicator)
                        if hasattr(result, 'speech_final') and result.speech_final:
                            self._on_end_of_turn(self._accumulated_transcript.strip())
                            self._accumulated_transcript = ""

                except Exception as e:
                    logger.error(f"Error in Flux message handler: {e}")

            def on_error(self_conn, error, **kwargs):
                logger.error(f"Flux streaming error: {error}")

            def on_close(self_conn, close, **kwargs):
                logger.debug("Flux streaming session closed")
                self._is_active = False
                # Deliver any remaining transcript
                if self._accumulated_transcript:
                    self._on_end_of_turn(self._accumulated_transcript.strip())

            self._connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self._connection.on(LiveTranscriptionEvents.Error, on_error)
            self._connection.on(LiveTranscriptionEvents.Close, on_close)

            if await self._connection.start(options):
                self._is_active = True
                logger.info("Flux streaming session started")
                return True
            else:
                logger.error("Failed to start Flux streaming session")
                return False

        except Exception as e:
            logger.error(f"Error starting Flux session: {e}", exc_info=True)
            return False

    async def send_audio(self, audio: np.ndarray) -> None:
        """Send audio chunk to Flux.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
        """
        if not self._is_active or self._connection is None:
            return

        # Convert float32 to int16 bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        await self._connection.send(audio_int16.tobytes())

    async def stop(self) -> None:
        """Stop the streaming session."""
        if self._connection:
            try:
                await self._connection.finish()
            except Exception as e:
                logger.warning(f"Error finishing Flux connection: {e}")
        self._is_active = False
        logger.info("Flux streaming session stopped")

    @property
    def is_active(self) -> bool:
        """Whether the session is active."""
        return self._is_active
