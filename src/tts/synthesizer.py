"""Text-to-speech synthesis using Inworld TTS API."""

import asyncio
import base64
import logging
import os
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class TTSSynthesizer:
    """Text-to-speech synthesizer using Inworld TTS API.

    Target output: 24kHz mono float32 audio
    """

    # Available Inworld voices (from API: /tts/v1/voices?filter=language=en)
    INWORLD_VOICES = [
        # Male voices
        "alex", "blake", "carter", "clive", "craig", "dennis", "dominus",
        "edward", "hades", "mark", "ronald", "shaun", "theodore", "timothy",
        # Female voices
        "ashley", "deborah", "elizabeth", "hana", "julia", "luna",
        "olivia", "pixie", "priya", "sarah", "wendy",
    ]

    def __init__(
        self,
        model: str = "inworld-tts-1-max",
        device: str = "cuda:0",  # Not used for Inworld but kept for API compatibility
        voice: str = "alex",
        api_key: Optional[str] = None,
    ):
        """Initialize TTS synthesizer.

        Args:
            model: Inworld model (inworld-tts-1-max or inworld-tts-1)
            device: Compute device (not used for Inworld, kept for compatibility)
            voice: Voice name (alex, ashley, craig, deborah, etc.)
            api_key: Inworld API key (or set INWORLD_API_KEY env var)
        """
        self.model_name = model
        self.device = device
        self.voice = voice.lower()
        self.sample_rate = 24000  # Inworld output rate

        # Get API key
        self.api_key = api_key or os.environ.get("INWORLD_API_KEY")

        self._tts_type = None
        self._setup_tts()

    def _setup_tts(self) -> None:
        """Set up TTS engine."""
        if not self.api_key:
            logger.error("INWORLD_API_KEY not set - TTS will not work")
            self._tts_type = None
            return

        # Validate voice
        if self.voice not in self.INWORLD_VOICES:
            logger.warning(f"Voice '{self.voice}' not in {self.INWORLD_VOICES}, using 'alex'")
            self.voice = "alex"

        self._tts_type = "inworld"
        logger.info(f"Using Inworld TTS ({self.model_name}) with voice '{self.voice}'")

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio as float32 numpy array (24kHz mono)
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        if self._tts_type != "inworld":
            logger.error("TTS not available - check INWORLD_API_KEY")
            return np.array([], dtype=np.float32)

        return await self._synthesize_inworld(text)

    async def _synthesize_inworld(self, text: str) -> np.ndarray:
        """Synthesize using Inworld TTS streaming API for lower latency."""
        import aiohttp
        import json

        start_time = time.perf_counter()
        first_chunk_time = None

        url = "https://api.inworld.ai/tts/v1/voice:stream"

        # Prepare auth header (Basic auth - key is already base64 encoded)
        auth_header = f"Basic {self.api_key}"

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": self.voice.capitalize(),  # API expects capitalized names
            "modelId": self.model_name,
            "audioConfig": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "speakingRate": 1.0,
            },
        }

        # Log what we're sending (for debugging emotion tags)
        logger.info(f"[TTS] Sending to Inworld: model={self.model_name}, text={text[:100]}...")

        audio_chunks = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Inworld API error {response.status}: {error_text}")
                        return np.array([], dtype=np.float32)

                    # Process newline-delimited JSON stream
                    raw_response = await response.text()
                    logger.debug(f"Inworld raw response: {raw_response[:500]}")

                    for line in raw_response.split('\n'):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            chunk_data = json.loads(line)
                            logger.debug(f"Inworld chunk keys: {chunk_data.keys()}")
                            # Audio can be at top level or nested under "result"
                            result = chunk_data.get("result", chunk_data)
                            audio_base64 = result.get("audioContent", "") or result.get("audio", "")
                            if audio_base64:
                                if first_chunk_time is None:
                                    first_chunk_time = time.perf_counter()
                                    ttfb = (first_chunk_time - start_time) * 1000
                                    logger.info(f"Inworld TTS first chunk in {ttfb:.0f}ms")

                                audio_bytes = base64.b64decode(audio_base64)

                                # Detect format and decode appropriately
                                if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                                    # MP3 audio - decode with pydub
                                    from pydub import AudioSegment
                                    import io
                                    audio_seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                                    audio_seg = audio_seg.set_frame_rate(self.sample_rate).set_channels(1)
                                    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.int16)
                                    audio_chunks.append(samples.astype(np.float32) / 32768.0)
                                elif audio_bytes[:4] == b'RIFF':
                                    # WAV audio - skip header
                                    audio_bytes = audio_bytes[44:]
                                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                                    audio_chunks.append(audio_int16.astype(np.float32) / 32768.0)
                                else:
                                    # Assume raw PCM
                                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                                    audio_chunks.append(audio_int16.astype(np.float32) / 32768.0)
                        except json.JSONDecodeError:
                            continue

            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                elapsed = time.perf_counter() - start_time
                logger.info(f"Inworld TTS generated {len(text)} chars in {elapsed:.2f}s ({len(text)/elapsed:.1f} chars/s)")
                return audio

            return np.array([], dtype=np.float32)

        except Exception as e:
            logger.error(f"Inworld TTS error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return np.array([], dtype=np.float32)

    def synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis.

        Args:
            text: Text to synthesize

        Returns:
            Audio as float32 numpy array
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.synthesize(text))
        finally:
            loop.close()

    def is_loaded(self) -> bool:
        """Check if TTS is available."""
        return self._tts_type is not None

    @property
    def using_inworld(self) -> bool:
        """Check if using Inworld TTS."""
        return self._tts_type == "inworld"
