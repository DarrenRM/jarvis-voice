"""Audio processing pipeline: STT -> LLM -> TTS.

This module provides a reusable pipeline for processing speech:
1. Transcribe audio to text (STT)
2. Generate response with LLM (streaming)
3. Synthesize response to audio (TTS)
"""

import logging
import re
from typing import AsyncIterator, Iterator, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class STTClient(Protocol):
    """Protocol for Speech-to-Text clients."""

    async def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        ...


class LLMClient(Protocol):
    """Protocol for LLM clients with streaming."""

    def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Stream LLM response chunks."""
        ...


class TTSClient(Protocol):
    """Protocol for Text-to-Speech clients."""

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio."""
        ...


class AudioPipeline:
    """Orchestrates speech processing: STT -> LLM -> TTS.

    This class provides the core pipeline without Discord-specific
    coordination logic (thinking phrases, silence detection, etc.).

    Example:
        pipeline = AudioPipeline(stt=groq_client, llm=llm_client, tts=tts)

        async for audio_chunk in pipeline.process(captured_audio):
            play_audio(audio_chunk)
    """

    # Sentence-ending punctuation
    SENTENCE_END = re.compile(r'([.!?])\s*')

    def __init__(
        self,
        stt: STTClient,
        llm: LLMClient,
        tts: TTSClient,
    ):
        """Initialize the audio pipeline.

        Args:
            stt: Speech-to-text client.
            llm: LLM client with streaming support.
            tts: Text-to-speech synthesizer.
        """
        self._stt = stt
        self._llm = llm
        self._tts = tts

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio as float32 numpy array.

        Returns:
            Transcribed text, or empty string if no speech.
        """
        return await self._stt.transcribe_audio(audio)

    async def generate_response(self, prompt: str) -> AsyncIterator[str]:
        """Generate LLM response with streaming.

        Args:
            prompt: User prompt/question.

        Yields:
            Response text chunks.
        """
        async for chunk in self._llm.chat_stream(prompt):
            yield chunk

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.

        Returns:
            Audio as float32 numpy array.
        """
        return await self._tts.synthesize(text)

    def split_into_sentences(self, text: str) -> Iterator[str]:
        """Split text into sentences for streaming TTS.

        Args:
            text: Text to split.

        Yields:
            Individual sentences.
        """
        if not text:
            return

        # Split on sentence-ending punctuation
        parts = self.SENTENCE_END.split(text)

        current = ""
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Punctuation characters go with the previous part
            if part in '.!?':
                if current:
                    yield current + part
                    current = ""
            else:
                if current:
                    yield current
                current = part

        # Yield any remaining text
        if current:
            yield current

    async def process(
        self,
        audio: np.ndarray,
        prompt_transform: callable = None,
    ) -> AsyncIterator[np.ndarray]:
        """Full pipeline: transcribe -> LLM -> stream TTS.

        Args:
            audio: Input audio as float32 numpy array.
            prompt_transform: Optional function to transform transcribed text
                              before sending to LLM (e.g., add speaker context).

        Yields:
            TTS audio chunks as float32 numpy arrays.
        """
        # Step 1: Transcribe
        text = await self.transcribe(audio)
        if not text or not text.strip():
            logger.debug("No speech detected in audio")
            return

        logger.info(f"[PIPELINE] Transcribed: {text[:100]}...")

        # Step 2: Transform prompt if needed
        prompt = prompt_transform(text) if prompt_transform else text

        # Step 3: Stream LLM response and synthesize each sentence
        buffer = ""

        async for chunk in self.generate_response(prompt):
            buffer += chunk

            # Process complete sentences (ending with punctuation or newline)
            while True:
                # Check for newline-delimited output
                if '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        logger.debug(f"[PIPELINE] TTS for line: {line[:50]}...")
                        audio_chunk = await self.synthesize(line)
                        if len(audio_chunk) > 0:
                            yield audio_chunk
                    continue

                # Check for sentence-ending punctuation
                match = self.SENTENCE_END.search(buffer)
                if match:
                    # Include the punctuation
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]

                    if sentence:
                        logger.debug(f"[PIPELINE] TTS for sentence: {sentence[:50]}...")
                        audio_chunk = await self.synthesize(sentence)
                        if len(audio_chunk) > 0:
                            yield audio_chunk
                    continue

                # No complete sentence yet, wait for more chunks
                break

        # Synthesize any remaining text
        remaining = buffer.strip()
        if remaining:
            logger.debug(f"[PIPELINE] TTS for remaining: {remaining[:50]}...")
            audio_chunk = await self.synthesize(remaining)
            if len(audio_chunk) > 0:
                yield audio_chunk

    async def process_text(self, text: str) -> AsyncIterator[np.ndarray]:
        """Process text directly through LLM -> TTS (skip STT).

        Useful when text comes from a different source (e.g., typed input).

        Args:
            text: Input text to process.

        Yields:
            TTS audio chunks as float32 numpy arrays.
        """
        if not text or not text.strip():
            return

        buffer = ""

        async for chunk in self.generate_response(text):
            buffer += chunk

            # Process complete sentences
            while '\n' in buffer or self.SENTENCE_END.search(buffer):
                if '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        audio_chunk = await self.synthesize(line)
                        if len(audio_chunk) > 0:
                            yield audio_chunk
                elif match := self.SENTENCE_END.search(buffer):
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]
                    if sentence:
                        audio_chunk = await self.synthesize(sentence)
                        if len(audio_chunk) > 0:
                            yield audio_chunk
                else:
                    break

        # Remaining text
        remaining = buffer.strip()
        if remaining:
            audio_chunk = await self.synthesize(remaining)
            if len(audio_chunk) > 0:
                yield audio_chunk
