"""Streaming TTS buffer for sentence-by-sentence synthesis."""

import asyncio
import logging
import re
from typing import AsyncIterator, Optional
import numpy as np

from .synthesizer import TTSSynthesizer

logger = logging.getLogger(__name__)


class StreamingTTSBuffer:
    """Buffers streaming text and synthesizes sentence by sentence.

    Features:
    - Sentence boundary detection
    - Concurrent synthesis while text is still arriving
    - Audio chunk streaming for low-latency playback
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]+\s*')

    def __init__(self, synthesizer: TTSSynthesizer):
        """Initialize streaming buffer.

        Args:
            synthesizer: TTS synthesizer instance
        """
        self.synthesizer = synthesizer
        self._text_buffer = ""
        self._pending_sentences: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
        self._running = False
        self._synthesis_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the synthesis pipeline."""
        self._running = True
        self._text_buffer = ""
        self._synthesis_task = asyncio.create_task(self._synthesis_loop())

    async def stop(self) -> None:
        """Stop synthesis and clear buffers."""
        self._running = False
        if self._synthesis_task:
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass

        # Clear queues
        while not self._pending_sentences.empty():
            try:
                self._pending_sentences.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def add_text(self, text: str) -> None:
        """Add text chunk to buffer.

        Args:
            text: Text chunk from Claude stream
        """
        self._text_buffer += text

        # Extract complete sentences
        while True:
            match = self.SENTENCE_ENDINGS.search(self._text_buffer)
            if not match:
                break

            # Extract sentence
            end_pos = match.end()
            sentence = self._text_buffer[:end_pos].strip()
            self._text_buffer = self._text_buffer[end_pos:]

            if sentence:
                self._pending_sentences.put_nowait(sentence)

    async def flush(self) -> None:
        """Flush remaining text in buffer."""
        if self._text_buffer.strip():
            self._pending_sentences.put_nowait(self._text_buffer.strip())
            self._text_buffer = ""

        # Signal end of text
        await self._pending_sentences.put(None)

    async def _synthesis_loop(self) -> None:
        """Background loop for sentence synthesis."""
        while self._running:
            try:
                sentence = await asyncio.wait_for(
                    self._pending_sentences.get(),
                    timeout=0.1,
                )

                if sentence is None:
                    # End signal
                    await self._audio_queue.put(None)
                    break

                # Synthesize sentence
                audio = await self.synthesizer.synthesize(sentence)
                if len(audio) > 0:
                    await self._audio_queue.put(audio)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synthesis error: {e}")

    async def get_audio_chunks(self) -> AsyncIterator[np.ndarray]:
        """Get synthesized audio chunks.

        Yields:
            Audio chunks as numpy arrays
        """
        while self._running:
            try:
                audio = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.1,
                )

                if audio is None:
                    break

                yield audio

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


async def stream_tts_from_llm(
    llm_stream: AsyncIterator[str],
    synthesizer: TTSSynthesizer,
) -> AsyncIterator[np.ndarray]:
    """Stream TTS from LLM output.

    Args:
        llm_stream: Async iterator of text chunks from LLM
        synthesizer: TTS synthesizer

    Yields:
        Audio chunks as numpy arrays
    """
    buffer = StreamingTTSBuffer(synthesizer)
    await buffer.start()

    async def feed_text():
        async for text_chunk in llm_stream:
            buffer.add_text(text_chunk)
        await buffer.flush()

    # Start feeding text in background
    feed_task = asyncio.create_task(feed_text())

    try:
        async for audio_chunk in buffer.get_audio_chunks():
            yield audio_chunk
    finally:
        feed_task.cancel()
        await buffer.stop()
