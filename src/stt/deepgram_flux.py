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
            sample_rate: Audio sample rate (16000 for Flux)
        """
        self.api_key = api_key
        self.eot_threshold = eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.sample_rate = sample_rate

        self._client = None
        self._is_initialized = False

    async def _ensure_client(self) -> None:
        """Initialize the async client if not already done."""
        if self._client is not None:
            return

        try:
            from deepgram import AsyncDeepgramClient

            self._client = AsyncDeepgramClient(api_key=self.api_key)
            self._is_initialized = True
            logger.info("Deepgram Flux client initialized")

        except ImportError:
            logger.error("deepgram-sdk not installed. Install with: pip install deepgram-sdk")
            raise

    async def transcribe_audio(
        self,
        audio: np.ndarray,
        on_interim: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Transcribe audio buffer using Flux streaming.

        This sends the audio to Flux via WebSocket and collects the transcript.

        Args:
            audio: Audio as float32 numpy array (16kHz mono)
            on_interim: Optional callback for interim transcripts

        Returns:
            Final transcript text
        """
        await self._ensure_client()

        try:
            from deepgram.core.events import EventType
            from deepgram.listen import ListenV2CloseStream, ListenV2TurnInfo

            # Convert float32 to int16 bytes for Deepgram
            audio_int16 = (audio * 32767).astype(np.int16)

            final_transcript = ""
            transcript_complete = asyncio.Event()

            async with self._client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate=str(self.sample_rate),
            ) as connection:

                def on_message(message) -> None:
                    nonlocal final_transcript
                    try:
                        # Flux v2 returns TurnInfo messages
                        msg_type = getattr(message, "type", "Unknown")
                        logger.debug(f"[FLUX] Received message type: {msg_type}")

                        if msg_type == "TurnInfo":
                            # This is a Flux TurnInfo message
                            event = getattr(message, "event", None)
                            transcript = getattr(message, "transcript", "")

                            logger.debug(f"[FLUX] TurnInfo event={event}, transcript={transcript[:50] if transcript else 'empty'}...")

                            if transcript:
                                if event == "EndOfTurn":
                                    final_transcript = transcript
                                    logger.info(f"[FLUX] EndOfTurn: {transcript}")
                                    transcript_complete.set()
                                elif event == "Update":
                                    # Interim result
                                    if on_interim:
                                        on_interim(transcript)
                                elif event == "StartOfTurn":
                                    logger.debug("[FLUX] StartOfTurn detected")

                    except Exception as e:
                        logger.error(f"Error processing Flux message: {e}", exc_info=True)

                def on_close(_) -> None:
                    logger.debug("[FLUX] Connection closed")
                    transcript_complete.set()

                def on_error(error) -> None:
                    logger.error(f"[FLUX] Error: {error}")
                    transcript_complete.set()

                # Register handlers
                connection.on(EventType.OPEN, lambda _: logger.debug("[FLUX] Connected"))
                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.CLOSE, on_close)
                connection.on(EventType.ERROR, on_error)

                # Start listening
                await connection.start_listening()

                # Send audio in chunks (80ms = 1280 samples at 16kHz)
                chunk_samples = int(self.sample_rate * 0.08)  # 80ms chunks
                for i in range(0, len(audio_int16), chunk_samples):
                    chunk = audio_int16[i:i + chunk_samples]
                    # send_media takes raw bytes directly
                    await connection.send_media(chunk.tobytes())
                    # Small delay to simulate real-time (optional)
                    await asyncio.sleep(0.01)

                # Signal end of audio
                await connection.send_close_stream(
                    ListenV2CloseStream(type="CloseStream")
                )

                # Wait for processing to complete (with timeout)
                try:
                    await asyncio.wait_for(transcript_complete.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("[FLUX] Timeout waiting for transcript completion")

            return final_transcript.strip()

        except Exception as e:
            logger.error(f"Flux transcription error: {e}", exc_info=True)
            return ""


class FluxStreamingSession:
    """Active streaming session for continuous audio input.

    Use this for true real-time streaming where audio is fed continuously
    and transcripts are returned as they're ready.
    """

    def __init__(
        self,
        client,
        sample_rate: int = 16000,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
        on_end_of_turn: Optional[Callable[[str], None]] = None,
    ):
        """Initialize streaming session.

        Args:
            client: AsyncDeepgramClient instance
            sample_rate: Audio sample rate
            on_transcript: Callback(text, is_final) for transcript updates
            on_end_of_turn: Callback(final_text) when speaker finished
        """
        self._client = client
        self._sample_rate = sample_rate
        self._on_transcript = on_transcript
        self._on_end_of_turn = on_end_of_turn
        self._connection = None
        self._is_active = False
        self._accumulated_transcript = ""
        self._context_manager = None

    async def start(self) -> bool:
        """Start the streaming session."""
        try:
            from deepgram.core.events import EventType

            # Create connection context
            self._context_manager = self._client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate=str(self._sample_rate),
            )
            self._connection = await self._context_manager.__aenter__()

            def on_message(message) -> None:
                try:
                    msg_type = getattr(message, "type", "Unknown")

                    if msg_type == "TurnInfo":
                        event = getattr(message, "event", None)
                        transcript = getattr(message, "transcript", "")

                        if transcript:
                            is_final = event == "EndOfTurn"

                            if self._on_transcript:
                                self._on_transcript(transcript, is_final)

                            if is_final:
                                if self._on_end_of_turn:
                                    self._on_end_of_turn(transcript)

                except Exception as e:
                    logger.error(f"Error in Flux message handler: {e}")

            def on_close(_) -> None:
                logger.debug("Flux streaming session closed")
                self._is_active = False

            def on_error(error) -> None:
                logger.error(f"Flux streaming error: {error}")
                self._is_active = False

            self._connection.on(EventType.MESSAGE, on_message)
            self._connection.on(EventType.CLOSE, on_close)
            self._connection.on(EventType.ERROR, on_error)

            await self._connection.start_listening()
            self._is_active = True
            logger.info("Flux streaming session started")
            return True

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

        try:
            # Convert float32 to int16 bytes
            audio_int16 = (audio * 32767).astype(np.int16)
            await self._connection.send_media(audio_int16.tobytes())
        except Exception as e:
            logger.error(f"Error sending audio to Flux: {e}")

    async def stop(self) -> str:
        """Stop the streaming session.

        Returns:
            Any remaining accumulated transcript
        """
        remaining = self._accumulated_transcript.strip()

        if self._connection:
            try:
                from deepgram.listen import ListenV2CloseStream
                await self._connection.send_close_stream(
                    ListenV2CloseStream(type="CloseStream")
                )
            except Exception as e:
                logger.warning(f"Error closing Flux connection: {e}")

        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error exiting Flux context: {e}")

        self._is_active = False
        self._connection = None
        self._context_manager = None
        logger.info("Flux streaming session stopped")

        return remaining

    @property
    def is_active(self) -> bool:
        """Whether the session is active."""
        return self._is_active
