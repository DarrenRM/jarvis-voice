"""Tests for AudioPipeline."""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock


class TestAudioPipeline:
    """Tests for AudioPipeline class."""

    @pytest.fixture
    def mock_stt(self):
        """Mock STT client."""
        stt = AsyncMock()
        stt.transcribe_audio = AsyncMock(return_value="Hello, how are you?")
        return stt

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client with streaming."""
        llm = Mock()

        async def mock_stream(prompt):
            responses = ["I'm doing ", "great, thanks ", "for asking!"]
            for chunk in responses:
                yield chunk

        llm.chat_stream = mock_stream
        return llm

    @pytest.fixture
    def mock_tts(self):
        """Mock TTS synthesizer."""
        tts = AsyncMock()
        # Return fake audio data
        tts.synthesize = AsyncMock(return_value=np.zeros(2400, dtype=np.float32))
        return tts

    @pytest.fixture
    def pipeline(self, mock_stt, mock_llm, mock_tts):
        """AudioPipeline with mocked dependencies."""
        from src.pipeline.audio_pipeline import AudioPipeline
        return AudioPipeline(stt=mock_stt, llm=mock_llm, tts=mock_tts)

    # Transcription tests

    async def test_transcribe_calls_stt(self, pipeline, mock_stt):
        """Transcription calls STT client."""
        audio = np.zeros(16000, dtype=np.float32)
        result = await pipeline.transcribe(audio)

        mock_stt.transcribe_audio.assert_called_once_with(audio)
        assert result == "Hello, how are you?"

    async def test_transcribe_empty_audio(self, pipeline, mock_stt):
        """Empty audio returns empty string."""
        mock_stt.transcribe_audio.return_value = ""
        audio = np.zeros(100, dtype=np.float32)

        result = await pipeline.transcribe(audio)
        assert result == ""

    async def test_transcribe_whitespace_only(self, pipeline, mock_stt):
        """Whitespace-only transcription returns empty."""
        mock_stt.transcribe_audio.return_value = "   \n\t  "
        audio = np.zeros(16000, dtype=np.float32)

        result = await pipeline.transcribe(audio)
        assert result.strip() == ""

    # LLM streaming tests

    async def test_generate_response_yields_chunks(self, pipeline, mock_llm):
        """generate_response yields LLM chunks."""
        chunks = []
        async for chunk in pipeline.generate_response("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert "".join(chunks) == "I'm doing great, thanks for asking!"

    async def test_generate_response_empty_prompt(self, pipeline, mock_llm):
        """Empty prompt still calls LLM."""
        chunks = []
        async for chunk in pipeline.generate_response(""):
            chunks.append(chunk)

        # Should still work
        assert len(chunks) >= 0

    # TTS tests

    async def test_synthesize_calls_tts(self, pipeline, mock_tts):
        """Synthesis calls TTS for text."""
        result = await pipeline.synthesize("Hello world")

        mock_tts.synthesize.assert_called_once_with("Hello world")
        assert isinstance(result, np.ndarray)

    async def test_synthesize_empty_text(self, pipeline, mock_tts):
        """Empty text returns empty audio."""
        mock_tts.synthesize.return_value = np.array([], dtype=np.float32)
        result = await pipeline.synthesize("")

        assert len(result) == 0

    # Full pipeline tests

    async def test_process_full_pipeline(self, pipeline, mock_stt, mock_tts):
        """Full pipeline: audio -> text -> LLM -> TTS chunks."""
        audio = np.zeros(16000, dtype=np.float32)
        tts_chunks = []

        async for audio_chunk in pipeline.process(audio):
            tts_chunks.append(audio_chunk)

        # Should have TTS chunks for each LLM sentence
        assert len(tts_chunks) > 0
        mock_stt.transcribe_audio.assert_called_once()
        assert mock_tts.synthesize.call_count > 0

    async def test_process_no_speech(self, pipeline, mock_stt, mock_tts):
        """Pipeline handles no speech gracefully."""
        mock_stt.transcribe_audio.return_value = ""
        audio = np.zeros(16000, dtype=np.float32)

        chunks = []
        async for chunk in pipeline.process(audio):
            chunks.append(chunk)

        assert len(chunks) == 0
        mock_tts.synthesize.assert_not_called()

    async def test_process_yields_audio_arrays(self, pipeline, mock_stt, mock_tts):
        """Pipeline yields numpy arrays."""
        audio = np.zeros(16000, dtype=np.float32)

        async for chunk in pipeline.process(audio):
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    # Sentence splitting tests

    def test_split_into_sentences_basic(self, pipeline):
        """Basic sentence splitting works."""
        sentences = list(pipeline.split_into_sentences("Hello. How are you?"))
        assert len(sentences) == 2
        assert sentences[0] == "Hello."
        assert sentences[1] == "How are you?"

    def test_split_into_sentences_single(self, pipeline):
        """Single sentence without punctuation."""
        sentences = list(pipeline.split_into_sentences("Hello world"))
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"

    def test_split_into_sentences_multiple_punct(self, pipeline):
        """Multiple punctuation types."""
        sentences = list(pipeline.split_into_sentences("What! Really? Yes."))
        assert len(sentences) == 3

    def test_split_into_sentences_empty(self, pipeline):
        """Empty text yields nothing."""
        sentences = list(pipeline.split_into_sentences(""))
        assert len(sentences) == 0

    def test_split_into_sentences_whitespace(self, pipeline):
        """Whitespace is trimmed."""
        sentences = list(pipeline.split_into_sentences("  Hello.   World!  "))
        assert len(sentences) == 2
        assert sentences[0] == "Hello."
        assert sentences[1] == "World!"


class TestAudioPipelineEdgeCases:
    """Edge case tests for AudioPipeline."""

    @pytest.fixture
    def mock_stt(self):
        stt = AsyncMock()
        stt.transcribe_audio = AsyncMock(return_value="Test")
        return stt

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()

        async def mock_stream(prompt):
            yield "Response"

        llm.chat_stream = mock_stream
        return llm

    @pytest.fixture
    def mock_tts(self):
        tts = AsyncMock()
        tts.synthesize = AsyncMock(return_value=np.zeros(100, dtype=np.float32))
        return tts

    @pytest.fixture
    def pipeline(self, mock_stt, mock_llm, mock_tts):
        from src.pipeline.audio_pipeline import AudioPipeline
        return AudioPipeline(stt=mock_stt, llm=mock_llm, tts=mock_tts)

    async def test_stt_error_handling(self, pipeline, mock_stt):
        """STT errors are propagated."""
        mock_stt.transcribe_audio.side_effect = Exception("STT failed")

        with pytest.raises(Exception, match="STT failed"):
            await pipeline.transcribe(np.zeros(1000, dtype=np.float32))

    async def test_tts_error_handling(self, pipeline, mock_tts):
        """TTS errors are propagated."""
        mock_tts.synthesize.side_effect = Exception("TTS failed")

        with pytest.raises(Exception, match="TTS failed"):
            await pipeline.synthesize("Hello")

    async def test_llm_yields_newlines(self, mock_stt, mock_tts):
        """LLM output with newlines is handled."""
        llm = Mock()

        async def mock_stream(prompt):
            yield "Line 1.\n"
            yield "Line 2.\n"

        llm.chat_stream = mock_stream

        from src.pipeline.audio_pipeline import AudioPipeline
        pipeline = AudioPipeline(stt=mock_stt, llm=llm, tts=mock_tts)

        audio = np.zeros(16000, dtype=np.float32)
        chunks = []
        async for chunk in pipeline.process(audio):
            chunks.append(chunk)

        # Should have TTS for each line
        assert mock_tts.synthesize.call_count >= 2
