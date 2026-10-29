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

    @pytest.mark.asyncio
    async def test_carry_on_cancel_never_plays_the_cancelled_answer(
        self, mock_transcription, context
    ):
        """VoiceSession._carried_on cancels the turn while the model is still streaming
        and the first sentence waits for the room to go quiet. With the real TTSStage,
        that answer must not play: the speaker is heard out and the turn starts over."""
        from src.pipeline.tts_stage import TTSStage

        played = []

        async def play_audio(audio):
            played.append(audio)

        async def slow_room(*args, **kwargs):
            await asyncio.sleep(0.2)  # room-silence debounce before the first sentence
            return True

        response = MagicMock()

        async def still_streaming(text, ctx):
            yield "The answer is forty two."
            await asyncio.sleep(10)
            yield " It was in the book."

        response.process_stream = still_streaming
        synth = MagicMock()
        synth.synthesize = AsyncMock(return_value=np.ones(2400, dtype=np.float32))
        pipeline = VoicePipeline(
            mock_transcription, response, TTSStage(synth, inter_sentence_pause_ms=0)
        )

        process = asyncio.create_task(pipeline.process(
            audio=make_audio(), context=context,
            play_audio=play_audio, wait_for_silence=slow_room,
        ))
        await asyncio.sleep(0.1)  # answer synthesized, waiting for the room
        assert synth.synthesize.await_count == 1 and not pipeline.answer_started
        process.cancel()
        await asyncio.wait({process}, timeout=1.0)
        await pipeline.cancel()
        await asyncio.sleep(0.3)  # past the room wait

        assert process.done() and process.result() == "cancelled"
        assert played == []

    @pytest.mark.asyncio
    async def test_carry_on_cancel_after_the_stream_ended_plays_nothing(
        self, mock_transcription, context
    ):
        """Same carry-on cancel, but the model has finished streaming: the last sentence
        is still synthesizing and the first waits for the room to go quiet. Cancelling
        the pipeline opens the thinking gate; the first sentence must not play then, and
        answer_started must stay False for the turn that starts over."""
        from src.pipeline.tts_stage import TTSStage

        played = []

        async def play_audio(audio):
            played.append(audio)

        async def slow_room(*args, **kwargs):
            await asyncio.sleep(0.2)  # the speaker is still talking
            return True

        response = MagicMock()

        async def short_answer(text, ctx):
            yield "First sentence.\n"
            yield "Second sentence."  # the stream ends here

        response.process_stream = short_answer

        async def synthesize(text):
            await asyncio.sleep(0.01 if text.startswith("First") else 10)
            return np.ones(2400, dtype=np.float32)

        synth = MagicMock()
        synth.synthesize = synthesize
        pipeline = VoicePipeline(
            mock_transcription, response, TTSStage(synth, inter_sentence_pause_ms=0)
        )

        process = asyncio.create_task(pipeline.process(
            audio=make_audio(), context=context,
            play_audio=play_audio, wait_for_silence=slow_room,
        ))
        await asyncio.sleep(0.1)  # stream done, sentence 1 at the gate, sentence 2 slow
        assert not pipeline.answer_started
        process.cancel()  # as VoiceSession._carried_on does
        await asyncio.wait({process}, timeout=1.0)
        await pipeline.cancel()
        await asyncio.sleep(0.4)  # well past the room wait

        assert process.done() and process.result() == "cancelled"
        assert played == []
        assert not pipeline.answer_started
        assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []

    @pytest.mark.asyncio
    async def test_cancel_while_the_answer_waits_behind_the_thinking_phrase(
        self, mock_transcription, context, wait_for_silence
    ):
        """pipeline.cancel() stops the thinking phrase, which opens the gate the answer
        waits at. The answer must not slip through that gate."""
        from src.pipeline.tts_stage import TTSStage

        played = []

        async def play_audio(audio):
            played.append(len(audio))
            if len(audio) == 100:  # the thinking phrase
                await asyncio.sleep(10)

        response = MagicMock()

        async def answer(text, ctx):
            yield "The answer is forty two."

        response.process_stream = answer
        synth = MagicMock()
        synth.synthesize = AsyncMock(return_value=np.ones(2400, dtype=np.float32))
        pipeline = VoicePipeline(
            mock_transcription, response, TTSStage(synth, inter_sentence_pause_ms=0)
        )

        process = asyncio.create_task(pipeline.process(
            audio=make_audio(), context=context, play_audio=play_audio,
            wait_for_silence=wait_for_silence, thinking_audio=make_audio(100),
        ))
        await asyncio.sleep(0.1)  # thinking phrase playing, the answer at the gate
        await pipeline.cancel()
        await asyncio.wait({process}, timeout=1.0)

        assert process.done()
        assert played == [100]
        assert not pipeline.answer_started


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
        played = []

        async def error_on_thinking(audio):
            if len(audio) == 100:
                raise RuntimeError("Play error")
            played.append(audio)

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

        # The answer still plays: both response chunks from mock_tts
        assert result is None
        assert len(played) == 2


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
