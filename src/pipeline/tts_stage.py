"""Streaming TTS stage with parallel synthesis and ordered playback.

Extracts patterns from voice_session.py:1066-1145:
- Sentence boundary detection (newlines + punctuation)
- Parallel TTS task spawning
- Ordered queue for out-of-order handling
- Error recovery (queue empty audio on failure)
"""

import asyncio
import logging
from typing import AsyncIterator, Callable, Optional, Protocol

import numpy as np

from ..audio import OrderedPlaybackQueue
from ..tts.text_filter import is_reasoning_text
from .base import PipelineContext, StreamingPipelineStage

logger = logging.getLogger(__name__)


# Inter-sentence pause duration (from voice_session.py:850)
INTER_SENTENCE_PAUSE_MS = 150


class TTSSynthesizerProtocol(Protocol):
    """Protocol for TTS synthesizers.

    Defines the interface the TTSStage needs.
    """

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio as float32 numpy array
        """
        ...


class TTSStage(StreamingPipelineStage):
    """Streaming TTS with parallel synthesis and ordered playback.

    Receives streaming text chunks, detects sentence boundaries,
    spawns parallel TTS tasks, and queues audio for ordered playback.

    Key patterns preserved from voice_session.py:
    - Sentence boundary detection (newlines + punctuation)
    - Parallel TTS task spawning per sentence
    - OrderedPlaybackQueue for out-of-order handling
    - Error recovery: queue empty audio on TTS failure

    Example:
        stage = TTSStage(synthesizer)

        async def handle_audio(audio: np.ndarray):
            # Play audio chunk
            player.play(audio)

        await stage.process_stream(response_stream, context, handle_audio)
    """

    def __init__(
        self,
        synthesizer: TTSSynthesizerProtocol,
        inter_sentence_pause_ms: int = INTER_SENTENCE_PAUSE_MS,
    ):
        """Initialize TTS stage.

        Args:
            synthesizer: TTS synthesizer for audio generation
            inter_sentence_pause_ms: Pause between sentences in ms
        """
        self._synthesizer = synthesizer
        self._inter_sentence_pause_ms = inter_sentence_pause_ms
        self._tasks: list[asyncio.Task] = []
        self._cancelled = False

    @property
    def name(self) -> str:
        """Stage name for logging and debugging."""
        return "tts"

    async def cancel(self) -> None:
        """Cancel any ongoing TTS processing."""
        self._cancelled = True
        for task in self._tasks:
            if not task.done():
                task.cancel()
        logger.debug(f"Cancelled {len(self._tasks)} TTS tasks")

    async def process_stream(
        self,
        response_stream: AsyncIterator[str],
        context: PipelineContext,
        playback_callback: Callable[[np.ndarray], asyncio.coroutine],
    ) -> None:
        """Process streaming response through parallel TTS.

        Args:
            response_stream: Async iterator yielding text chunks
            context: Pipeline context
            playback_callback: Async function to call with each audio chunk

        Note:
            Sentences are synthesized in parallel but played in order.
            If a sentence fails TTS, empty audio is queued to not block ordering.
        """
        self._cancelled = False
        self._tasks = []
        queue = OrderedPlaybackQueue()
        buffer = ""
        sentences_queued = 1  # 1-based to match OrderedPlaybackQueue

        # Start playback consumer
        playback_task = asyncio.create_task(
            self._playback_consumer(queue, playback_callback)
        )

        try:
            # Process incoming text chunks
            async for chunk in response_stream:
                if self._cancelled:
                    break

                buffer += chunk

                # Primary: Split on newlines (from voice_session.py:1096)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._spawn_synthesis_task(line, sentences_queued, queue)
                        sentences_queued += 1

                # Secondary: Split on sentence-ending punctuation
                # (from voice_session.py:1118-1122)
                buffer = buffer.strip()
                if buffer and buffer[-1] in ".!?":
                    self._spawn_synthesis_task(buffer, sentences_queued, queue)
                    sentences_queued += 1
                    buffer = ""

            # Process remaining buffer
            if buffer.strip() and not self._cancelled:
                self._spawn_synthesis_task(buffer.strip(), sentences_queued, queue)
                sentences_queued += 1

        finally:
            # Wait for all TTS tasks to complete
            # (from voice_session.py:1134-1145)
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            # Signal end of queue
            await queue.signal_end()

            # Wait for playback to finish
            await playback_task

        logger.debug(f"TTS stage completed: {sentences_queued} sentences processed")

    def _spawn_synthesis_task(
        self,
        text: str,
        sentence_num: int,
        queue: OrderedPlaybackQueue,
    ) -> None:
        """Spawn a parallel TTS task for a sentence.

        Args:
            text: Sentence text to synthesize
            sentence_num: Sentence number for ordering
            queue: Playback queue for audio output
        """
        # Filter out LLM reasoning/thinking text before TTS
        if is_reasoning_text(text):
            logger.debug(f"[TTS] Filtered reasoning text: {text[:80]}...")
            # Queue empty audio to maintain ordering
            asyncio.create_task(
                queue.put(sentence_num, np.array([], dtype=np.float32))
            )
            return

        task = asyncio.create_task(
            self._synthesize_and_queue(text, sentence_num, queue)
        )
        self._tasks.append(task)
        logger.debug(f"Spawned TTS task {sentence_num}: {text[:50]}...")

    async def _synthesize_and_queue(
        self,
        text: str,
        sentence_num: int,
        queue: OrderedPlaybackQueue,
    ) -> None:
        """Synthesize text and queue audio for playback.

        On TTS failure, queues empty audio to not block ordering.
        (Pattern from voice_session.py:950-953)

        Args:
            text: Text to synthesize
            sentence_num: Sentence number for ordering
            queue: Playback queue for audio output
        """
        try:
            audio = await self._synthesizer.synthesize(text)
            await queue.put(sentence_num, audio)
        except asyncio.CancelledError:
            logger.debug(f"TTS task {sentence_num} cancelled")
            # Queue empty audio to not block ordering
            await queue.put(sentence_num, np.array([], dtype=np.float32))
            raise
        except Exception as e:
            logger.error(f"TTS error for sentence {sentence_num}: {e}")
            # Queue empty audio to not block ordering
            await queue.put(sentence_num, np.array([], dtype=np.float32))

    async def _playback_consumer(
        self,
        queue: OrderedPlaybackQueue,
        callback: Callable[[np.ndarray], asyncio.coroutine],
    ) -> None:
        """Consume audio from queue and play in order.

        Args:
            queue: Ordered playback queue
            callback: Async function to call with each audio chunk
        """
        first_audio = True

        async for audio in queue.iter_in_order():
            if self._cancelled:
                break

            # Skip empty audio (from TTS errors)
            if len(audio) == 0:
                continue

            await callback(audio)

            # Inter-sentence pause (from voice_session.py:850)
            if not first_audio and self._inter_sentence_pause_ms > 0:
                await asyncio.sleep(self._inter_sentence_pause_ms / 1000)

            first_audio = False
