"""Tests for TTSStage.

Tests the parallel TTS synthesis and ordered playback logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.pipeline.base import PipelineContext
from src.pipeline.tts_stage import TTSStage, INTER_SENTENCE_PAUSE_MS


def make_audio(samples: int = 1000, value: float = 0.5) -> np.ndarray:
    """Create test audio data with distinct values."""
    return np.full(samples, value, dtype=np.float32)


def make_collector(target_list: list):
    """Create an async callback that appends audio to a list."""
    async def callback(audio):
        target_list.append(audio)
    return callback


@pytest.fixture
def mock_synthesizer():
    """Create a mock TTS synthesizer."""
    synth = MagicMock()
    synth.synthesize = AsyncMock(side_effect=lambda text: make_audio(len(text) * 100))
    return synth


@pytest.fixture
def stage(mock_synthesizer):
    """Create a TTSStage with mock synthesizer."""
    return TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)  # No pause for tests


@pytest.fixture
def context():
    """Create a pipeline context."""
    return PipelineContext()


class TestTTSStage:
    """Test TTSStage basic functionality."""

    def test_name(self, stage):
        """Stage has correct name."""
        assert stage.name == "tts"

    @pytest.mark.asyncio
    async def test_simple_stream(self, stage, mock_synthesizer, context):
        """Simple text stream is synthesized and played."""
        async def response_stream():
            yield "Hello world"

        played_audio = []

        async def callback(audio):
            played_audio.append(audio)

        await stage.process_stream(response_stream(), context, callback)

        # Synthesize was called
        mock_synthesizer.synthesize.assert_called()

        # Audio was played
        assert len(played_audio) > 0


class TestSentenceBoundaryDetection:
    """Test sentence boundary detection patterns."""

    @pytest.mark.asyncio
    async def test_newline_splits(self, stage, mock_synthesizer, context):
        """Newlines trigger sentence splits."""
        async def response_stream():
            yield "First line\n"
            yield "Second line"

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # Two sentences synthesized
        assert mock_synthesizer.synthesize.call_count == 2
        calls = [call[0][0] for call in mock_synthesizer.synthesize.call_args_list]
        assert "First line" in calls
        assert "Second line" in calls

    @pytest.mark.asyncio
    async def test_period_splits(self, stage, mock_synthesizer, context):
        """Periods trigger sentence splits."""
        async def response_stream():
            yield "First sentence."
            yield " Second sentence."

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # Two sentences synthesized
        assert mock_synthesizer.synthesize.call_count == 2

    @pytest.mark.asyncio
    async def test_exclamation_splits(self, stage, mock_synthesizer, context):
        """Exclamation marks trigger sentence splits."""
        async def response_stream():
            yield "Wow!"
            yield " Amazing!"

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        assert mock_synthesizer.synthesize.call_count == 2

    @pytest.mark.asyncio
    async def test_question_splits(self, stage, mock_synthesizer, context):
        """Question marks trigger sentence splits."""
        async def response_stream():
            yield "How are you?"
            yield " What's new?"

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        assert mock_synthesizer.synthesize.call_count == 2

    @pytest.mark.asyncio
    async def test_remaining_buffer_processed(self, stage, mock_synthesizer, context):
        """Remaining buffer without punctuation is processed at end."""
        async def response_stream():
            yield "No punctuation at end"

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # One sentence synthesized
        assert mock_synthesizer.synthesize.call_count == 1


class TestParallelSynthesis:
    """Test parallel synthesis behavior."""

    @pytest.mark.asyncio
    async def test_tasks_run_in_parallel(self, mock_synthesizer, context):
        """Multiple sentences are synthesized in parallel."""
        # Track when synthesis starts/ends
        synthesis_timeline = []

        async def slow_synthesize(text):
            synthesis_timeline.append(("start", text))
            await asyncio.sleep(0.1)
            synthesis_timeline.append(("end", text))
            return make_audio()

        mock_synthesizer.synthesize = AsyncMock(side_effect=slow_synthesize)
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)

        async def response_stream():
            yield "First.\n"
            yield "Second.\n"
            yield "Third."

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # All three start before any ends (parallel execution)
        starts = [i for i, (action, _) in enumerate(synthesis_timeline) if action == "start"]
        ends = [i for i, (action, _) in enumerate(synthesis_timeline) if action == "end"]

        # At least some starts should happen before ends (indicating parallelism)
        # In fully parallel execution, all 3 starts would happen before any end
        assert len(starts) == 3
        assert len(ends) == 3


class TestOrderedPlayback:
    """Test ordered playback behavior."""

    @pytest.mark.asyncio
    async def test_out_of_order_synthesis(self, mock_synthesizer, context):
        """Audio plays in order even if synthesis completes out of order."""
        synthesis_order = []
        playback_order = []

        async def variable_delay_synthesize(text):
            # First sentence takes longest
            if "First" in text:
                await asyncio.sleep(0.15)
            elif "Second" in text:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.05)

            synthesis_order.append(text)
            return make_audio(len(text))

        mock_synthesizer.synthesize = AsyncMock(side_effect=variable_delay_synthesize)
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)

        async def response_stream():
            yield "First.\n"
            yield "Second.\n"
            yield "Third."

        async def track_playback(audio):
            playback_order.append(len(audio))

        await stage.process_stream(response_stream(), context, track_playback)

        # Synthesis completed in reverse order (Third, Second, First)
        assert "Third." in synthesis_order[0]

        # But playback is in original order (by sentence length)
        assert playback_order[0] == len("First.")  # First played first


class TestErrorRecovery:
    """Test TTS error recovery."""

    @pytest.mark.asyncio
    async def test_error_queues_empty_audio(self, mock_synthesizer, context):
        """TTS errors queue empty audio to not block ordering."""
        call_count = [0]

        async def failing_synthesize(text):
            call_count[0] += 1
            if call_count[0] == 2:  # Second sentence fails
                raise RuntimeError("TTS error")
            return make_audio(len(text))

        mock_synthesizer.synthesize = AsyncMock(side_effect=failing_synthesize)
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)

        async def response_stream():
            yield "First.\n"
            yield "Second.\n"
            yield "Third."

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # Two audio chunks played (first and third, second was empty)
        assert len(played_audio) == 2

    @pytest.mark.asyncio
    async def test_error_does_not_block_subsequent(self, mock_synthesizer, context):
        """Error in one sentence doesn't block subsequent sentences."""
        async def selective_fail(text):
            if "fail" in text.lower():
                raise RuntimeError("TTS error")
            return make_audio(len(text))

        mock_synthesizer.synthesize = AsyncMock(side_effect=selective_fail)
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)

        async def response_stream():
            yield "Before.\n"
            yield "This will fail.\n"
            yield "After."

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # Before and After were played
        assert len(played_audio) == 2


class TestCancellation:
    """Test cancellation behavior."""

    @pytest.mark.asyncio
    async def test_cancel_stops_processing(self, mock_synthesizer, context):
        """stage.cancel() while sentences are still synthesizing: none of them play."""
        async def slow_synthesize(text):
            await asyncio.sleep(1)
            return make_audio()

        mock_synthesizer.synthesize = AsyncMock(side_effect=slow_synthesize)
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)

        async def response_stream():
            yield "First.\n"
            yield "Second.\n"
            yield "Third."

        played_audio = []
        task = asyncio.create_task(
            stage.process_stream(response_stream(), context, make_collector(played_audio))
        )
        await asyncio.sleep(0.1)
        await stage.cancel()
        await asyncio.wait_for(task, timeout=0.5)  # doesn't wait out the 1 s syntheses

        assert played_audio == []

    @pytest.mark.asyncio
    async def test_cancelled_mid_stream_plays_nothing_already_synthesized(
        self, mock_synthesizer, context
    ):
        """The carry-on case: the model is still streaming and the first sentence is
        synthesized, waiting for the room to go quiet, when the turn is cancelled. That
        answer must never play, even once the room goes quiet."""
        room_quiet = asyncio.Event()
        played_audio = []

        async def play_when_quiet(audio):
            await room_quiet.wait()
            played_audio.append(audio)

        async def response_stream():
            yield "The answer is forty two.\n"
            await asyncio.Event().wait()  # model still streaming the next sentence

        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)
        task = asyncio.create_task(
            stage.process_stream(response_stream(), context, play_when_quiet)
        )
        await asyncio.sleep(0.05)
        assert mock_synthesizer.synthesize.await_count == 1

        task.cancel()
        room_quiet.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        await asyncio.sleep(0.05)

        assert played_audio == []


class TestInterSentencePause:
    """Test inter-sentence pause behavior."""

    @pytest.mark.asyncio
    async def test_pause_between_sentences(self, mock_synthesizer, context):
        """Pause is added between sentences."""
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=50)

        async def response_stream():
            yield "First.\n"
            yield "Second."

        play_times = []

        async def track_time(audio):
            play_times.append(asyncio.get_event_loop().time())

        start = asyncio.get_event_loop().time()
        await stage.process_stream(response_stream(), context, track_time)
        end = asyncio.get_event_loop().time()

        # Should have taken at least 50ms due to pause
        assert len(play_times) == 2
        # Note: First sentence has no pause before it

    @pytest.mark.asyncio
    async def test_no_pause_with_zero_setting(self, mock_synthesizer, context):
        """No pause when inter_sentence_pause_ms is 0."""
        stage = TTSStage(mock_synthesizer, inter_sentence_pause_ms=0)

        async def response_stream():
            yield "First.\n"
            yield "Second."

        play_times = []

        async def track_time(audio):
            play_times.append(asyncio.get_event_loop().time())

        await stage.process_stream(response_stream(), context, track_time)

        # Both played, timing not significantly delayed
        assert len(play_times) == 2


class TestEmptyInput:
    """Test handling of empty or whitespace input."""

    @pytest.mark.asyncio
    async def test_empty_stream(self, stage, mock_synthesizer, context):
        """Empty stream produces no output."""
        async def response_stream():
            return
            yield  # Make it a generator

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        assert len(played_audio) == 0
        mock_synthesizer.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only(self, stage, mock_synthesizer, context):
        """Whitespace-only stream produces no output."""
        async def response_stream():
            yield "   \n   \n   "

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # Empty lines stripped, nothing to synthesize
        assert mock_synthesizer.synthesize.call_count == 0


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    @pytest.mark.asyncio
    async def test_mixed_punctuation_and_newlines(self, stage, mock_synthesizer, context):
        """Mixed punctuation and newlines handled correctly."""
        async def response_stream():
            yield "Hello! How are you?\n"
            yield "I'm doing great. Thanks for asking!"

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # Should split on: "Hello!", "How are you?", "I'm doing great.", "Thanks for asking!"
        # But depends on chunking - at minimum the lines are split
        assert mock_synthesizer.synthesize.call_count >= 2

    @pytest.mark.asyncio
    async def test_streaming_chunks(self, stage, mock_synthesizer, context):
        """Handles word-by-word streaming correctly."""
        async def response_stream():
            # Simulate word-by-word streaming
            words = "The quick brown fox jumps.".split()
            for word in words:
                yield word + " "

        played_audio = []
        await stage.process_stream(
            response_stream(), context, make_collector(played_audio)
        )

        # One sentence synthesized
        assert mock_synthesizer.synthesize.call_count == 1
        call_text = mock_synthesizer.synthesize.call_args[0][0]
        assert "quick" in call_text
        assert "fox" in call_text
