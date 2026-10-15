"""Pipeline stage base classes and context.

This module provides the foundation for the voice processing pipeline:
- PipelineContext: Context passed through all stages
- PipelineStage: Abstract base class for pipeline stages

The pipeline follows this flow:
    Audio -> TranscriptionStage -> ResponseStage -> TTSStage -> Playback
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineContext:
    """Context passed through pipeline stages.

    Contains speaker information and metadata that stages can use
    to customize their behavior (e.g., personalized responses).

    Attributes:
        speaker_id: Discord user ID of the speaker (if known)
        speaker_name: Display name of the speaker (if known)
        speaker_activity: Current Discord activity (e.g., "playing Minecraft")
        channel_activities: Activities of the others in the voice channel
        metadata: Additional context for extensibility
    """

    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    speaker_activity: Optional[str] = None
    channel_activities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **kwargs) -> "PipelineContext":
        """Return a new context with additional metadata.

        Args:
            **kwargs: Metadata key-value pairs to add

        Returns:
            New PipelineContext with merged metadata
        """
        new_metadata = {**self.metadata, **kwargs}
        return PipelineContext(
            speaker_id=self.speaker_id,
            speaker_name=self.speaker_name,
            speaker_activity=self.speaker_activity,
            metadata=new_metadata,
        )


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.

    Each stage processes input and produces output, with optional
    cancellation support. Stages should be stateless between calls
    (any state should be managed externally or reset on each call).

    Example implementation:
        class MyStage(PipelineStage):
            @property
            def name(self) -> str:
                return "my_stage"

            async def process(self, input: str, context: PipelineContext) -> str:
                return f"Processed: {input}"
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging and debugging.

        Returns:
            Short identifier for this stage (e.g., "transcription", "response")
        """
        pass

    @abstractmethod
    async def process(self, input: Any, context: PipelineContext) -> Any:
        """Process input and produce output.

        Args:
            input: Input data (type depends on stage)
            context: Pipeline context with speaker info

        Returns:
            Output data (type depends on stage)

        Raises:
            asyncio.CancelledError: If cancelled during processing
            Exception: On processing errors
        """
        pass

    async def cancel(self) -> None:
        """Cancel any ongoing processing.

        Default implementation is a no-op. Override in stages
        that have long-running operations that can be cancelled.

        This should:
        - Stop any pending work
        - Clean up resources
        - Allow process() to exit cleanly (may raise CancelledError)
        """
        pass


class StreamingPipelineStage(PipelineStage):
    """Base class for pipeline stages that produce streaming output.

    Used for stages like ResponseStage and TTSStage that yield
    chunks rather than returning a complete result.

    Subclasses should implement process_stream() instead of process().
    """

    async def process(self, input: Any, context: PipelineContext) -> Any:
        """Not used for streaming stages.

        Raises:
            NotImplementedError: Streaming stages use process_stream()
        """
        raise NotImplementedError(
            f"{self.name} is a streaming stage. Use process_stream() instead."
        )

    @abstractmethod
    async def process_stream(self, input: Any, context: PipelineContext):
        """Process input and yield output chunks.

        Args:
            input: Input data (type depends on stage)
            context: Pipeline context with speaker info

        Yields:
            Output chunks (type depends on stage)

        Raises:
            asyncio.CancelledError: If cancelled during processing
            Exception: On processing errors
        """
        pass
