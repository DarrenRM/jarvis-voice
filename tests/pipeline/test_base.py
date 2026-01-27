"""Tests for pipeline base classes."""

import pytest

from src.pipeline.base import PipelineContext, PipelineStage, StreamingPipelineStage


class TestPipelineContext:
    """Test PipelineContext dataclass."""

    def test_default_values(self):
        """Context has sensible defaults."""
        ctx = PipelineContext()

        assert ctx.speaker_id is None
        assert ctx.speaker_name is None
        assert ctx.speaker_activity is None
        assert ctx.metadata == {}

    def test_with_values(self):
        """Context stores provided values."""
        ctx = PipelineContext(
            speaker_id=12345,
            speaker_name="Alice",
            speaker_activity="playing Minecraft",
            metadata={"key": "value"},
        )

        assert ctx.speaker_id == 12345
        assert ctx.speaker_name == "Alice"
        assert ctx.speaker_activity == "playing Minecraft"
        assert ctx.metadata == {"key": "value"}

    def test_with_metadata_returns_new_context(self):
        """with_metadata returns new context, doesn't mutate original."""
        original = PipelineContext(speaker_name="Alice", metadata={"a": 1})
        new_ctx = original.with_metadata(b=2)

        # Original unchanged
        assert original.metadata == {"a": 1}

        # New context has merged metadata
        assert new_ctx.metadata == {"a": 1, "b": 2}
        assert new_ctx.speaker_name == "Alice"

    def test_with_metadata_overwrites_keys(self):
        """with_metadata can overwrite existing keys."""
        original = PipelineContext(metadata={"key": "old"})
        new_ctx = original.with_metadata(key="new")

        assert original.metadata == {"key": "old"}
        assert new_ctx.metadata == {"key": "new"}


class TestPipelineStage:
    """Test PipelineStage abstract base class."""

    def test_cannot_instantiate_directly(self):
        """PipelineStage is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            PipelineStage()

    def test_subclass_must_implement_name(self):
        """Subclass must implement name property."""

        class IncompleteStage(PipelineStage):
            async def process(self, input, context):
                return input

        with pytest.raises(TypeError):
            IncompleteStage()

    def test_subclass_must_implement_process(self):
        """Subclass must implement process method."""

        class IncompleteStage(PipelineStage):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteStage()

    @pytest.mark.asyncio
    async def test_valid_subclass(self):
        """Valid subclass can be instantiated and used."""

        class ValidStage(PipelineStage):
            @property
            def name(self):
                return "valid"

            async def process(self, input, context):
                return f"processed: {input}"

        stage = ValidStage()
        assert stage.name == "valid"

        ctx = PipelineContext()
        result = await stage.process("test", ctx)
        assert result == "processed: test"

    @pytest.mark.asyncio
    async def test_default_cancel_is_noop(self):
        """Default cancel implementation is a no-op."""

        class ValidStage(PipelineStage):
            @property
            def name(self):
                return "valid"

            async def process(self, input, context):
                return input

        stage = ValidStage()
        # Should not raise
        await stage.cancel()

    @pytest.mark.asyncio
    async def test_cancel_can_be_overridden(self):
        """Subclass can override cancel."""
        cancelled = []

        class CancellableStage(PipelineStage):
            @property
            def name(self):
                return "cancellable"

            async def process(self, input, context):
                return input

            async def cancel(self):
                cancelled.append(True)

        stage = CancellableStage()
        await stage.cancel()
        assert cancelled == [True]


class TestStreamingPipelineStage:
    """Test StreamingPipelineStage abstract base class."""

    def test_cannot_instantiate_directly(self):
        """StreamingPipelineStage is abstract."""
        with pytest.raises(TypeError):
            StreamingPipelineStage()

    @pytest.mark.asyncio
    async def test_process_raises_not_implemented(self):
        """process() raises NotImplementedError for streaming stages."""

        class ValidStreamingStage(StreamingPipelineStage):
            @property
            def name(self):
                return "streaming"

            async def process_stream(self, input, context):
                yield input

        stage = ValidStreamingStage()
        ctx = PipelineContext()

        with pytest.raises(NotImplementedError) as exc_info:
            await stage.process("test", ctx)

        assert "streaming" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_process_stream(self):
        """Valid streaming stage yields output."""

        class ValidStreamingStage(StreamingPipelineStage):
            @property
            def name(self):
                return "streaming"

            async def process_stream(self, input, context):
                for i in range(3):
                    yield f"chunk_{i}"

        stage = ValidStreamingStage()
        ctx = PipelineContext()

        results = []
        async for chunk in stage.process_stream("test", ctx):
            results.append(chunk)

        assert results == ["chunk_0", "chunk_1", "chunk_2"]

    @pytest.mark.asyncio
    async def test_streaming_stage_with_context(self):
        """Streaming stage can use context."""

        class ContextAwareStage(StreamingPipelineStage):
            @property
            def name(self):
                return "context_aware"

            async def process_stream(self, input, context):
                yield f"Hello, {context.speaker_name or 'friend'}!"

        stage = ContextAwareStage()
        ctx = PipelineContext(speaker_name="Alice")

        results = []
        async for chunk in stage.process_stream("hi", ctx):
            results.append(chunk)

        assert results == ["Hello, Alice!"]
