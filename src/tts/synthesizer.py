"""Text-to-speech synthesis using Inworld TTS API."""

import asyncio
import base64
import contextlib
import logging
import os
import re
import time
from collections import OrderedDict
from typing import AsyncIterator, Optional
import numpy as np

from ..audio import SAMPLE_RATE_TTS
from .text_filter import normalize_numbers_for_tts, strip_urls_for_tts

logger = logging.getLogger(__name__)

# Repeat guard: Inworld occasionally speaks a sentence twice in one clip. A clip
# far longer than its text should take is synthesized again (shorter one kept).
# Calibrated on e2e captures: normal speech ~16 chars/s, short phrases carry fixed
# overhead, non-verbal tags add time; the observed repeat was ~2.2x expected.
_CHARS_PER_SECOND = 16.0
_REPEAT_FACTOR = 1.7
_REPEAT_SLACK_S = 0.9
_NONVERBAL_TAG_S = 0.8
_NONVERBAL_TAGS = re.compile(r"\[(sigh|laugh|laughing|breathe|cough|clear_throat|yawn)\]", re.IGNORECASE)
_ANY_TAG = re.compile(r"\[[a-z_ ]+\]", re.IGNORECASE)
# [laughing] is a delivery style (a laughing tone). TTS-2 Flash has no delivery styles
# and plays a full laugh for it instead, which on nearly every reply is too much.
# Explicit [laugh] non-verbals are kept.
_LAUGHING_STYLE = re.compile(r"\[laughing\]\s*", re.IGNORECASE)

# Short fixed phrases (acks, affirmations) are cached so they play instantly
_PHRASE_CACHE_SIZE = 64


def _max_plausible_seconds(text: str, speed: float) -> float:
    spoken = _ANY_TAG.sub("", text)
    expected = len(spoken.strip()) / _CHARS_PER_SECOND / max(speed, 0.5)
    expected += _NONVERBAL_TAG_S * len(_NONVERBAL_TAGS.findall(text))
    return _REPEAT_FACTOR * expected + _REPEAT_SLACK_S


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
        temperature: float = 1.1,
        speed: float = 1.0,
    ):
        """Initialize TTS synthesizer.

        Args:
            model: Inworld model (inworld-tts-1-max or inworld-tts-1)
            device: Compute device (not used for Inworld, kept for compatibility)
            voice: Voice name (alex, ashley, craig, deborah, etc.)
            api_key: Inworld API key (or set INWORLD_API_KEY env var)
            temperature: Expressiveness (0.0-2.0, default 1.1). Higher = more expressive.
            speed: Talking speed multiplier (0.5-1.5, default 1.0).
        """
        self.model_name = model
        self.device = device
        self.voice = voice.lower()
        self.sample_rate = SAMPLE_RATE_TTS  # Inworld output rate
        self.temperature = max(0.0, min(2.0, temperature))
        self.speed = max(0.5, min(1.5, speed))

        # Get API key
        self.api_key = api_key or os.environ.get("INWORLD_API_KEY")

        self._tts_type = None
        self._setup_tts()

        # One HTTP session reused across requests (saves a TLS handshake per clip)
        self._session = None
        self._session_loop = None
        self._phrase_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

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
        logger.info(f"Using Inworld TTS ({self.model_name}) with voice '{self.voice}', temp={self.temperature}, speed={self.speed}")

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

        # Never read links aloud; normalize numbers to words for better pronunciation
        text = normalize_numbers_for_tts(strip_urls_for_tts(text))
        if "flash" in self.model_name:
            text = _LAUGHING_STYLE.sub("", text)
            if not _ANY_TAG.sub("", text).strip():
                return np.array([], dtype=np.float32)

        audio = await self._synthesize_inworld(text)
        limit = _max_plausible_seconds(text, self.speed)
        duration = len(audio) / self.sample_rate
        if duration > limit:
            logger.warning(
                f"[TTS] Clip is {duration:.1f}s for {len(text)} chars (limit {limit:.1f}s), "
                f"possible repeat - synthesizing again: {text[:60]!r}"
            )
            retry = await self._synthesize_inworld(text)
            if 0 < len(retry) < len(audio):
                audio = retry
        return audio

    async def synthesize_cached(self, text: str) -> np.ndarray:
        """synthesize() for short fixed phrases (acks, affirmations); repeats are instant."""
        cached = self._phrase_cache.get(text)
        if cached is not None:
            self._phrase_cache.move_to_end(text)
            return cached
        audio = await self.synthesize(text)
        if len(audio) > 0:
            self._phrase_cache[text] = audio
            if len(self._phrase_cache) > _PHRASE_CACHE_SIZE:
                self._phrase_cache.popitem(last=False)
        return audio

    @contextlib.asynccontextmanager
    async def _shared_session(self):
        """Yield the reusable aiohttp session (recreated if closed or on a new loop)."""
        import aiohttp

        loop = asyncio.get_running_loop()
        if self._session is None or self._session.closed or self._session_loop is not loop:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=15))
            self._session_loop = loop
        yield self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _synthesize_inworld(self, text: str) -> np.ndarray:
        """Synthesize using Inworld TTS streaming API for lower latency."""
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
            "temperature": self.temperature,
            "audioConfig": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "speakingRate": self.speed,
            },
        }

        # Log what we're sending (for debugging emotion tags)
        logger.info(f"[TTS] Sending to Inworld: model={self.model_name}, text={text[:100]}...")

        # Collect all raw audio bytes first, then decode once to avoid MP3 frame boundary artifacts
        raw_audio_chunks: list[bytes] = []
        detected_format: Optional[str] = None

        try:
            async with self._shared_session() as session:
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
                                raw_audio_chunks.append(audio_bytes)

                                # Detect format from first chunk
                                if detected_format is None:
                                    if audio_bytes[:3] == b'ID3' or (len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and audio_bytes[1] >= 0xE0):
                                        detected_format = "mp3"
                                    elif audio_bytes[:4] == b'RIFF':
                                        detected_format = "wav"
                                    else:
                                        detected_format = "pcm"
                        except json.JSONDecodeError:
                            continue

            if not raw_audio_chunks:
                return np.array([], dtype=np.float32)

            # Decode all audio at once to avoid frame boundary artifacts
            combined_bytes = b''.join(raw_audio_chunks)

            if detected_format == "mp3":
                from pydub import AudioSegment
                import io
                audio_seg = AudioSegment.from_mp3(io.BytesIO(combined_bytes))
                audio_seg = audio_seg.set_frame_rate(self.sample_rate).set_channels(1)
                samples = np.array(audio_seg.get_array_of_samples(), dtype=np.int16)
                audio = samples.astype(np.float32) / 32768.0
            elif detected_format == "wav":
                # Skip WAV header (44 bytes)
                audio_int16 = np.frombuffer(combined_bytes[44:], dtype=np.int16)
                audio = audio_int16.astype(np.float32) / 32768.0
            else:
                # Raw PCM
                audio_int16 = np.frombuffer(combined_bytes, dtype=np.int16)
                audio = audio_int16.astype(np.float32) / 32768.0

            elapsed = time.perf_counter() - start_time
            logger.info(f"Inworld TTS generated {len(text)} chars in {elapsed:.2f}s ({len(text)/elapsed:.1f} chars/s)",
                        extra={"tts_seconds": elapsed})  # read by the live monitor
            return audio

        except Exception as e:
            logger.error(f"Inworld TTS error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return np.array([], dtype=np.float32)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Stream audio chunks as they arrive from Inworld.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as float32 numpy arrays (24kHz mono)
        """
        if not text.strip():
            return

        if self._tts_type != "inworld":
            logger.error("TTS not available - check INWORLD_API_KEY")
            return

        # Never read links aloud; normalize numbers to words for better pronunciation
        text = normalize_numbers_for_tts(strip_urls_for_tts(text))

        import json

        start_time = time.perf_counter()
        first_chunk_time = None
        total_chars = len(text)
        chunks_yielded = 0

        url = "https://api.inworld.ai/tts/v1/voice:stream"

        auth_header = f"Basic {self.api_key}"
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": self.voice.capitalize(),
            "modelId": self.model_name,
            "temperature": self.temperature,
            "audioConfig": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "speakingRate": self.speed,
            },
        }

        logger.info(f"[TTS-STREAM] Starting: {text[:80]}...")

        try:
            async with self._shared_session() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Inworld API error {response.status}: {error_text}")
                        return

                    # Stream line by line as chunks arrive
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8', errors='ignore')

                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                chunk_data = json.loads(line)
                                result = chunk_data.get("result", chunk_data)
                                audio_base64 = result.get("audioContent", "") or result.get("audio", "")

                                if audio_base64:
                                    if first_chunk_time is None:
                                        first_chunk_time = time.perf_counter()
                                        ttfb = (first_chunk_time - start_time) * 1000
                                        logger.info(f"[TTS-STREAM] First chunk in {ttfb:.0f}ms")

                                    audio_bytes = base64.b64decode(audio_base64)
                                    audio_chunk = self._decode_audio_bytes(audio_bytes)

                                    if len(audio_chunk) > 0:
                                        chunks_yielded += 1
                                        yield audio_chunk

                            except json.JSONDecodeError:
                                continue

                    # Process any remaining data in buffer
                    if buffer.strip():
                        try:
                            chunk_data = json.loads(buffer.strip())
                            result = chunk_data.get("result", chunk_data)
                            audio_base64 = result.get("audioContent", "") or result.get("audio", "")
                            if audio_base64:
                                audio_bytes = base64.b64decode(audio_base64)
                                audio_chunk = self._decode_audio_bytes(audio_bytes)
                                if len(audio_chunk) > 0:
                                    chunks_yielded += 1
                                    yield audio_chunk
                        except json.JSONDecodeError:
                            pass

            elapsed = time.perf_counter() - start_time
            logger.info(f"[TTS-STREAM] Done: {total_chars} chars, {chunks_yielded} chunks in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Inworld TTS stream error: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _decode_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to float32 numpy array."""
        # MP3: ID3 tag or frame sync (0xFF followed by 0xE0+ for MPEG audio)
        is_mp3 = (
            audio_bytes[:3] == b'ID3' or
            (len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and audio_bytes[1] >= 0xE0)
        )
        if is_mp3:
            # MP3 audio - decode with pydub
            from pydub import AudioSegment
            import io
            audio_seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_frame_rate(self.sample_rate).set_channels(1)
            samples = np.array(audio_seg.get_array_of_samples(), dtype=np.int16)
            return samples.astype(np.float32) / 32768.0
        elif audio_bytes[:4] == b'RIFF':
            # WAV audio - skip header
            audio_bytes = audio_bytes[44:]
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int16.astype(np.float32) / 32768.0
        else:
            # Assume raw PCM
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int16.astype(np.float32) / 32768.0

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
