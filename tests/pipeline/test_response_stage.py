"""Tests for ResponseStage.

Tests the wrapper around ChatOrchestrator that adds speaker context.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.base import PipelineContext
from src.pipeline.response_stage import ResponseStage


@pytest.fixture
def mock_orchestrator():
    """Create a mock ChatOrchestrator."""
    orchestrator = MagicMock()

    async def mock_chat_stream(message, messages, on_message_update=None):
        yield "Hello "
        yield "world!"

    orchestrator.chat_stream = mock_chat_stream
    return orchestrator


@pytest.fixture
def stage(mock_orchestrator):
    """Create a ResponseStage with mock orchestrator."""
    return ResponseStage(mock_orchestrator, system_prompt="You are a helpful assistant.")


@pytest.fixture
def context():
    """Create a pipeline context."""
    return PipelineContext(speaker_name="Alice")


class TestResponseStage:
    """Test ResponseStage functionality."""

    def test_name(self, stage):
        """Stage has correct name."""
        assert stage.name == "response"

    @pytest.mark.asyncio
    async def test_streams_response(self, stage, context):
        """Response chunks are streamed."""
        chunks = []
        async for chunk in stage.process_stream("hello", context):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]

    @pytest.mark.asyncio
    async def test_initializes_messages_with_system_prompt(self, stage, context):
        """First call initializes messages with system prompt."""
        async for _ in stage.process_stream("hello", context):
            pass

        messages = stage.get_messages()
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_adds_user_message(self, stage, context):
        """User message is added to history."""
        async for _ in stage.process_stream("hello", context):
            pass

        messages = stage.get_messages()
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "hello" in user_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_adds_assistant_response(self, stage, context):
        """Assistant response is added to history."""
        async for _ in stage.process_stream("hello", context):
            pass

        messages = stage.get_messages()
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Hello world!"


class TestSpeakerContext:
    """Test speaker context handling."""

    @pytest.mark.asyncio
    async def test_speaker_name_prefix(self, stage, mock_orchestrator):
        """Speaker name is added as prefix to user message."""
        captured_messages = []

        async def capture_chat_stream(message, messages, on_message_update=None):
            captured_messages.extend(messages)
            yield "OK"

        mock_orchestrator.chat_stream = capture_chat_stream

        context = PipelineContext(speaker_name="Bob")
        async for _ in stage.process_stream("hello", context):
            pass

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "[Bob]: hello" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_no_speaker_name(self, stage, mock_orchestrator):
        """No prefix when speaker name is absent."""
        captured_messages = []

        async def capture_chat_stream(message, messages, on_message_update=None):
            captured_messages.extend(messages)
            yield "OK"

        mock_orchestrator.chat_stream = capture_chat_stream

        context = PipelineContext()  # No speaker name
        async for _ in stage.process_stream("hello", context):
            pass

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert user_msg["content"] == "hello"

    @pytest.mark.asyncio
    async def test_speaker_activity_in_system_prompt(self, stage, mock_orchestrator):
        """Speaker activity is added to system prompt."""
        captured_messages = []

        async def capture_chat_stream(message, messages, on_message_update=None):
            captured_messages.extend(messages)
            yield "OK"

        mock_orchestrator.chat_stream = capture_chat_stream

        context = PipelineContext(speaker_activity="playing Minecraft")
        async for _ in stage.process_stream("hello", context):
            pass

        system_msg = next(m for m in captured_messages if m["role"] == "system")
        assert "playing Minecraft" in system_msg["content"]


class TestConversationHistory:
    """Test conversation history management."""

    @pytest.mark.asyncio
    async def test_history_persists(self, stage, context):
        """Conversation history persists across calls."""
        # First message
        async for _ in stage.process_stream("hello", context):
            pass

        # Second message
        async for _ in stage.process_stream("how are you", context):
            pass

        messages = stage.get_messages()
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 2

    @pytest.mark.asyncio
    async def test_reset_conversation(self, stage, context):
        """Reset clears conversation history."""
        # First message
        async for _ in stage.process_stream("hello", context):
            pass

        stage.reset_conversation()

        assert stage.get_messages() == []

    @pytest.mark.asyncio
    async def test_system_prompt_reinitializes_after_reset(self, stage, context):
        """System prompt is re-added after reset."""
        # First conversation
        async for _ in stage.process_stream("hello", context):
            pass

        stage.reset_conversation()

        # New conversation
        async for _ in stage.process_stream("hello again", context):
            pass

        messages = stage.get_messages()
        assert messages[0]["role"] == "system"

    def test_get_messages_returns_copy(self, stage):
        """get_messages returns a copy, not the original list."""
        messages = stage.get_messages()
        messages.append({"role": "test", "content": "test"})

        assert {"role": "test", "content": "test"} not in stage.get_messages()


class TestMessageCallback:
    """Test message update callback."""

    @pytest.mark.asyncio
    async def test_callback_passed_to_orchestrator(self, stage, mock_orchestrator, context):
        """Message callback is passed to orchestrator."""
        captured_callback = []

        async def capture_chat_stream(message, messages, on_message_update=None):
            captured_callback.append(on_message_update)
            yield "OK"

        mock_orchestrator.chat_stream = capture_chat_stream

        callback = MagicMock()
        stage.set_message_callback(callback)

        async for _ in stage.process_stream("hello", context):
            pass

        assert captured_callback[0] is callback


class TestEmptyResponse:
    """Test handling of empty responses."""

    @pytest.mark.asyncio
    async def test_empty_response_not_added_to_history(self, mock_orchestrator):
        """Empty response is not added to message history."""
        async def empty_stream(message, messages, on_message_update=None):
            return
            yield  # Make it a generator

        mock_orchestrator.chat_stream = empty_stream
        stage = ResponseStage(mock_orchestrator, system_prompt="Test")
        context = PipelineContext()

        async for _ in stage.process_stream("hello", context):
            pass

        messages = stage.get_messages()
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 0
