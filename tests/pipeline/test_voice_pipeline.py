"""Tests for VoicePipeline.

Tests the full voice processing pipeline with thinking phrase support.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.pipeline.base import PipelineContext
from src.pipeline.transcription_stage import TranscriptionResult
from src.pipeline.voice_pipeline import OpeningPhrase, VoicePipeline


def make_audio(samples: int = 1000) -> np.ndarray:
    """Create test audio data."""
    return np.zeros(samples, dtype=np.float32)


@pytest.fixture
def mock_transcription():
    """Create a mock TranscriptionStage."""
    stage = MagicMock()
    stage.process = AsyncMock(
        return_value=TranscriptionResult(text="hello world", raw_text="hey jarvis hello world")
    )
    stage.cancel = AsyncMock()
    return stage


@pytest.fixture
def mock_response():
    """Create a mock ResponseStage."""
    stage = MagicMock()

    async def mock_stream(text, context):
        yield "Response "
        yield "text"

    stage.process_stream = mock_stream
    stage.reset_conversation = MagicMock()
    return stage


@pytest.fixture
def mock_tts():
    """Create a mock TTSStage."""
    stage = MagicMock()

    async def mock_process_stream(response_stream, context, callback):
        async for chunk in response_stream:
            await callback(make_audio())

    stage.process_stream = mock_process_stream
    stage.cancel = AsyncMock()
    return stage


@pytest.fixture
def pipeline(mock_transcription, mock_response, mock_tts):
    """Create a VoicePipeline with mocks."""
    return VoicePipeline(mock_transcription, mock_response, mock_tts)


@pytest.fixture
def context():
    """Create a pipeline context."""
    return PipelineContext(speaker_name="Alice")


@pytest.fixture
def play_audio():
    """Create a mock play_audio callback."""
    played = []

    async def callback(audio):
        played.append(audio)

    callback.played = played
    return callback


@pytest.fixture
def wait_for_silence():
    """Create a mock wait_for_silence callback."""
    return AsyncMock(return_value=True)


class TestVoicePipeline:
    """Test VoicePipeline basic functionality."""

    @pytest.mark.asyncio
    async def test_normal_flow(
        self, pipeline, context, play_audio, wait_for_silence
    ):
        """Normal flow processes audio through all stages."""
        result = await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
        )

        assert result is None  # Normal completion
        assert len(play_audio.played) > 0  # Audio was played

    @pytest.mark.asyncio
    async def test_calls_transcription(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        """Transcription stage is called with audio."""
        audio = make_audio()

        await pipeline.process(
            audio=audio,
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
        )

        mock_transcription.process.assert_called_once_with(audio, context)

    @pytest.mark.asyncio
    async def test_waits_for_silence(
        self, pipeline, context, play_audio, wait_for_silence
    ):
        """Wait for silence is called before first audio."""
        await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
        )

        wait_for_silence.assert_called()


class TestTranscriptionResults:
    """Test handling of different transcription results."""

    @pytest.mark.asyncio
    async def test_empty_transcription(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        """Empty transcription returns 'empty'."""
        mock_transcription.process.return_value = TranscriptionResult.empty()

        result = await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
        )

        assert result == "empty"
        assert len(play_audio.played) == 0  # No audio played

    @pytest.mark.asyncio
    async def test_wake_word_only(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        """Wake word only returns 'wake_word_only'."""
        mock_transcription.process.return_value = TranscriptionResult.wake_only("hey jarvis")

        result = await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
        )

        assert result == "wake_word_only"
        assert len(play_audio.played) == 0  # No audio played


class TestThinkingPhrase:
    """Test thinking phrase handling."""

    @pytest.mark.asyncio
    async def test_thinking_phrase_plays_first(
        self, pipeline, context, wait_for_silence
    ):
        """Thinking phrase plays before response."""
        played_order = []

        async def track_play(audio):
            if len(audio) == 100:
                played_order.append("thinking")
            else:
                played_order.append("response")

        thinking_audio = make_audio(100)

        await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=track_play,
            wait_for_silence=wait_for_silence,
            thinking_phrase="Let me think...",
            thinking_audio=thinking_audio,
        )

        # Thinking phrase should play before response
        assert "thinking" in played_order
        assert played_order.index("thinking") < played_order.index("response")

    @pytest.mark.asyncio
    async def test_response_waits_for_thinking(
        self, pipeline, mock_response, mock_tts, context, wait_for_silence
    ):
        """Response playback waits for thinking phrase to complete."""
        thinking_done = False
        response_started = False

        async def slow_play(audio):
            nonlocal thinking_done, response_started
            if len(audio) == 100:  # Thinking phrase
                await asyncio.sleep(0.1)
                thinking_done = True
            else:
                response_started = True
                # Response should only start after thinking
                assert thinking_done, "Response started before thinking finished"

        thinking_audio = make_audio(100)

        await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=slow_play,
            wait_for_silence=wait_for_silence,
            thinking_phrase="Thinking...",
            thinking_audio=thinking_audio,
        )

    @pytest.mark.asyncio
    async def test_no_thinking_phrase(
        self, pipeline, context, play_audio, wait_for_silence
    ):
        """Pipeline works without thinking phrase."""
        result = await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
            thinking_phrase=None,
            thinking_audio=None,
        )

        assert result is None  # Normal completion

    @pytest.mark.asyncio
    async def test_empty_thinking_audio(
        self, pipeline, context, play_audio, wait_for_silence
    ):
        """Empty thinking audio is skipped."""
        result = await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
            thinking_phrase="Thinking...",
            thinking_audio=np.array([], dtype=np.float32),
        )

        assert result is None  # Normal completion


class TestCancellation:
    """Test cancellation behavior."""

    @pytest.mark.asyncio
    async def test_cancel_stops_pipeline(
        self, pipeline, mock_transcription, mock_tts, context, play_audio, wait_for_silence
    ):
        """Cancel stops pipeline and returns 'cancelled'."""
        # Make transcription slow so we can cancel during it
        async def slow_transcribe(audio, ctx):
            await asyncio.sleep(1)
            return TranscriptionResult(text="hello", raw_text="hello")

        mock_transcription.process = slow_transcribe

        async def run_and_cancel():
            task = asyncio.create_task(
                pipeline.process(
                    audio=make_audio(),
                    context=context,
                    play_audio=play_audio,
                    wait_for_silence=wait_for_silence,
                )
            )
            await asyncio.sleep(0.1)
            await pipeline.cancel()
            task.cancel()
            try:
                return await task
            except asyncio.CancelledError:
                return "cancelled"

        result = await run_and_cancel()
        # Task was cancelled
        assert result == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_propagates_to_stages(
        self, pipeline, mock_transcription, mock_tts, context, play_audio, wait_for_silence
    ):
        """Cancel propagates to all stages."""
        await pipeline.cancel()

        mock_transcription.cancel.assert_called()
        mock_tts.cancel.assert_called()


class TestConversationReset:
    """Test conversation reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_conversation(
        self, pipeline, mock_response
    ):
        """Reset conversation calls response stage reset."""
        pipeline.reset_conversation()

        mock_response.reset_conversation.assert_called_once()


class TestContextPropagation:
    """Test that context is properly propagated."""

    @pytest.mark.asyncio
    async def test_context_to_transcription(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        """Context is passed to transcription stage."""
        await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=play_audio,
            wait_for_silence=wait_for_silence,
        )

        # Check context was passed
        call_args = mock_transcription.process.call_args
        assert call_args[0][1] == context


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_transcription_error(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        """Transcription error propagates."""
        mock_transcription.process.side_effect = RuntimeError("STT error")

        with pytest.raises(RuntimeError, match="STT error"):
            await pipeline.process(
                audio=make_audio(),
                context=context,
                play_audio=play_audio,
                wait_for_silence=wait_for_silence,
            )

    @pytest.mark.asyncio
    async def test_thinking_phrase_error_continues(
        self, pipeline, context, wait_for_silence
    ):
        """Error in thinking phrase doesn't stop pipeline."""
        first_call = [True]

        async def error_on_thinking(audio):
            if first_call[0]:
                first_call[0] = False
                raise RuntimeError("Play error")
            # Subsequent calls succeed

        thinking_audio = make_audio(100)

        # Should complete despite thinking phrase error
        result = await pipeline.process(
            audio=make_audio(),
            context=context,
            play_audio=error_on_thinking,
            wait_for_silence=wait_for_silence,
            thinking_phrase="Thinking...",
            thinking_audio=thinking_audio,
        )

        # Pipeline continues after thinking phrase error
        # (result depends on whether response playback also fails)


class TestOpeningPhrase:
    """Opening phrase (affirmation) starts at speech end, before transcription finishes."""

    @staticmethod
    def make_opening(play_audio, audio_len=100, should_play=None, gate=None):
        async def synthesize():
            if gate is not None:
                await gate.wait()
            return make_audio(audio_len)

        return OpeningPhrase(synthesize, play_audio, should_play)

    @pytest.mark.asyncio
    async def test_plays_while_transcription_runs(
        self, pipeline, mock_transcription, context, wait_for_silence
    ):
        events = []
        transcribe_release = asyncio.Event()

        async def slow_transcribe(audio, ctx):
            events.append("stt_start")
            await transcribe_release.wait()
            events.append("stt_done")
            return TranscriptionResult(text="hello world", raw_text="hey jarvis hello world")

        mock_transcription.process = slow_transcribe

        async def play(audio):
            events.append("opening" if len(audio) == 100 else "response")
            if len(audio) == 100:
                transcribe_release.set()

        opening = self.make_opening(play)
        await pipeline.process(
            audio=make_audio(), context=context, play_audio=play,
            wait_for_silence=wait_for_silence, opening=opening,
        )
        assert events.index("opening") < events.index("stt_done")
        assert events.index("opening") < events.index("response")

    @pytest.mark.asyncio
    async def test_dynamic_phrase_follows_opening(self, pipeline, context, wait_for_silence):
        order = []

        async def play(audio):
            order.append({100: "opening", 50: "dynamic"}.get(len(audio), "response"))

        generator = MagicMock()
        generator.generate = AsyncMock(return_value="Checking the weather")
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=make_audio(50))

        opening = self.make_opening(play)
        await pipeline.process(
            audio=make_audio(), context=context, play_audio=play,
            wait_for_silence=wait_for_silence, thinking_generator=generator,
            tts_synthesizer=tts, opening=opening,
        )
        assert order[:2] == ["opening", "dynamic"]
        assert "response" in order[2:]

    @pytest.mark.asyncio
    async def test_skipped_opening_skips_dynamic_phrase(self, pipeline, context, wait_for_silence):
        order = []

        async def play(audio):
            order.append({100: "opening", 50: "dynamic"}.get(len(audio), "response"))

        generator = MagicMock()
        generator.generate = AsyncMock(return_value="Checking the weather")
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=make_audio(50))

        opening = self.make_opening(play, should_play=AsyncMock(return_value=False))
        await pipeline.process(
            audio=make_audio(), context=context, play_audio=play,
            wait_for_silence=wait_for_silence, thinking_generator=generator,
            tts_synthesizer=tts, opening=opening,
        )
        assert opening.skipped
        assert "opening" not in order and "dynamic" not in order
        assert "response" in order

    @pytest.mark.asyncio
    async def test_empty_transcription_cancels_unstarted_opening(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        mock_transcription.process.return_value = TranscriptionResult.empty()
        gate = asyncio.Event()
        opening = self.make_opening(play_audio, gate=gate)

        result = await pipeline.process(
            audio=make_audio(), context=context, play_audio=play_audio,
            wait_for_silence=wait_for_silence, opening=opening,
        )
        gate.set()
        await asyncio.sleep(0)
        assert result == "empty"
        assert play_audio.played == []

    @pytest.mark.asyncio
    async def test_static_response_waits_for_opening(self, pipeline, context, wait_for_silence):
        opening_finished = False

        async def play(audio):
            nonlocal opening_finished
            if len(audio) == 100:
                await asyncio.sleep(0.05)
                opening_finished = True
            else:
                assert opening_finished, "Response started before opening finished"

        opening = self.make_opening(play)
        await pipeline.process(
            audio=make_audio(), context=context, play_audio=play,
            wait_for_silence=wait_for_silence, thinking_phrase="One moment...", opening=opening,
        )
        assert opening_finished


class TestAccept:
    """The follow-up gate between transcription and the reply."""

    @pytest.mark.asyncio
    async def test_rejected_speech_gets_no_reply(
        self, pipeline, context, play_audio, wait_for_silence
    ):
        seen = []

        async def accept(transcription):
            seen.append(transcription.text)
            return False

        result = await pipeline.process(
            audio=make_audio(), context=context, play_audio=play_audio,
            wait_for_silence=wait_for_silence, accept=accept,
        )
        assert result == "not_addressed"
        assert seen == ["hello world"]
        assert play_audio.played == []

    @pytest.mark.asyncio
    async def test_accepted_speech_is_answered(
        self, pipeline, context, play_audio, wait_for_silence
    ):
        async def accept(transcription):
            return True

        result = await pipeline.process(
            audio=make_audio(), context=context, play_audio=play_audio,
            wait_for_silence=wait_for_silence, accept=accept,
        )
        assert result is None
        assert len(play_audio.played) == 2

    @pytest.mark.asyncio
    async def test_not_asked_for_empty_speech(
        self, pipeline, mock_transcription, context, play_audio, wait_for_silence
    ):
        mock_transcription.process.return_value = TranscriptionResult.empty()
        accept = AsyncMock(return_value=True)
        result = await pipeline.process(
            audio=make_audio(), context=context, play_audio=play_audio,
            wait_for_silence=wait_for_silence, accept=accept,
        )
        assert result == "empty"
        accept.assert_not_called()
