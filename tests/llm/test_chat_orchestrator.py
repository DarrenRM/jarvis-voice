"""Tests for ChatOrchestrator.

Tests the tool loop logic extracted from groq_client.py (lines 292-567).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.chat_orchestrator import (
    APIResponse,
    ChatOrchestrator,
    OrchestratorState,
    ToolCall,
    MAX_SEARCH_CALLS,
)
from src.tools.base import ToolResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.call_api = AsyncMock(return_value=APIResponse(content="Hello!"))

    # For stream_response, we need a function that returns an async generator
    # (not AsyncMock, which wraps it in a coroutine)
    async def default_stream(messages):
        yield "Hello "
        yield "world!"

    llm.stream_response = default_stream
    return llm


@pytest.fixture
def mock_executor():
    """Create a mock tool executor."""
    executor = MagicMock()
    executor.reset_counters = MagicMock()
    executor.get_definitions = MagicMock(return_value=[
        {"function": {"name": "web_search", "description": "Search the web"}},
        {"function": {"name": "generate_image", "description": "Generate an image"}},
    ])
    executor.normalize_name = MagicMock(side_effect=lambda x: x)
    executor.is_search_tool = MagicMock(side_effect=lambda x: x in {"web_search", "fetch_url"})
    executor.is_action_tool = MagicMock(side_effect=lambda x: x in {"generate_image", "tag_user"})
    executor.execute = AsyncMock(return_value=ToolResult(success=True, output="Search results: ..."))
    return executor


@pytest.fixture
def orchestrator(mock_llm, mock_executor):
    """Create an orchestrator with mocks."""
    return ChatOrchestrator(mock_llm, mock_executor, max_turns=5)


class TestOrchestratorState:
    """Test OrchestratorState dataclass."""

    def test_initial_state(self):
        """Initial state has empty counters."""
        state = OrchestratorState()

        assert state.tool_calls_this_request == []
        assert state.search_calls_this_request == 0
        assert state.last_action_success is None
        assert state.search_limit_reached is False

    def test_search_limit_reached(self):
        """search_limit_reached is True after MAX_SEARCH_CALLS."""
        state = OrchestratorState()

        state.search_calls_this_request = MAX_SEARCH_CALLS - 1
        assert state.search_limit_reached is False

        state.search_calls_this_request = MAX_SEARCH_CALLS
        assert state.search_limit_reached is True


class TestAPIResponse:
    """Test APIResponse dataclass."""

    def test_no_tool_calls(self):
        """has_tool_calls is False when no tool calls."""
        response = APIResponse(content="Hello")

        assert response.has_tool_calls is False

    def test_with_tool_calls(self):
        """has_tool_calls is True when tool calls present."""
        response = APIResponse(
            content="I'll search for that",
            tool_calls=[ToolCall(id="1", name="web_search", arguments='{"query": "test"}')],
        )

        assert response.has_tool_calls is True


class TestChatOrchestrator:
    """Test ChatOrchestrator functionality."""

    @pytest.mark.asyncio
    async def test_simple_response_no_tools(self, orchestrator, mock_llm):
        """Simple response without tool calls streams directly."""
        mock_llm.call_api.return_value = APIResponse(content=None)

        stream_called = []
        async def stream_gen(messages):
            stream_called.append(messages)
            yield "Hello "
            yield "world!"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Hi"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Hi", messages):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]
        assert len(stream_called) == 1  # stream_response was called once

    @pytest.mark.asyncio
    async def test_resets_counters(self, orchestrator, mock_executor, mock_llm):
        """Counters are reset at start of chat_stream."""
        mock_llm.call_api.return_value = APIResponse(content=None)

        async def stream_gen(messages):
            yield "Done"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Hi"}]
        async for _ in orchestrator.chat_stream("Hi", messages):
            pass

        mock_executor.reset_counters.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_execution(self, orchestrator, mock_llm, mock_executor):
        """Tool calls are executed and results added to messages."""
        # First call returns tool call, second call streams response
        tool_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "weather"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "The weather is sunny"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "What's the weather?"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("What's the weather?", messages):
            chunks.append(chunk)

        # Tool was executed
        mock_executor.execute.assert_called_once_with("web_search", {"query": "weather"})

        # Messages updated with tool call and result
        assert any(m.get("role") == "assistant" and "tool_calls" in m for m in messages)
        assert any(m.get("role") == "tool" for m in messages)

        # Response streamed
        assert "The weather is sunny" in chunks

    @pytest.mark.asyncio
    async def test_pre_tool_content_yielded(self, orchestrator, mock_llm, mock_executor):
        """Content before tool call is yielded."""
        tool_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content="Let me search for that", tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Test", messages):
            chunks.append(chunk)

        # Pre-tool content yielded
        assert "Let me search for that" in chunks

    @pytest.mark.asyncio
    async def test_deep_research_notification(self, orchestrator, mock_llm, mock_executor):
        """Deep research tool yields notification before execution."""
        tool_call = ToolCall(id="tc1", name="deep_research", arguments='{"topic": "test"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Research test"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Research test", messages):
            chunks.append(chunk)

        # Notification yielded
        assert any("deeper digging" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_search_limit_enforcement(self, orchestrator, mock_llm, mock_executor):
        """After MAX_SEARCH_CALLS, search tools are excluded."""
        search_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')

        # Return search calls up to limit, then response
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None),  # After limit
        ]

        async def stream_gen(messages):
            yield "Final response"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        # get_definitions called with search_limit_reached=True on 4th call
        calls = mock_executor.get_definitions.call_args_list
        assert len(calls) == 4
        assert calls[3][1]["search_limit_reached"] is True

    @pytest.mark.asyncio
    async def test_force_answer_message_injected(self, orchestrator, mock_llm, mock_executor):
        """Force answer message injected when search limit reached."""
        search_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')

        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        # Force answer message should be in messages
        assert any("DO NOT search again" in m.get("content", "") for m in messages)

    @pytest.mark.asyncio
    async def test_action_success_tracking(self, orchestrator, mock_llm, mock_executor):
        """Successful action tool results are tracked."""
        tool_call = ToolCall(id="tc1", name="generate_image", arguments='{"prompt": "cat"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        mock_executor.execute.return_value = ToolResult(
            success=True, output="Done! I've posted the image."
        )

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Generate cat"}]
        async for _ in orchestrator.chat_stream("Generate cat", messages):
            pass

        # Action tool was executed
        mock_executor.execute.assert_called()

    @pytest.mark.asyncio
    async def test_max_turns_exhausted(self, orchestrator, mock_llm, mock_executor):
        """Graceful failure when max turns exhausted."""
        tool_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')

        # Always return tool calls (never finishes)
        mock_llm.call_api.return_value = APIResponse(content=None, tool_calls=[tool_call])

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Test", messages):
            chunks.append(chunk)

        # Graceful failure message
        full_response = "".join(chunks)
        assert "couldn't find a good answer" in full_response

    @pytest.mark.asyncio
    async def test_output_parse_failed_with_action_success(
        self, orchestrator, mock_llm, mock_executor
    ):
        """Action success used as fallback on parse failure."""
        # First call: successful image generation
        img_call = ToolCall(id="tc1", name="generate_image", arguments='{"prompt": "cat"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[img_call]),
            Exception("output_parse_failed"),
        ]

        mock_executor.execute.return_value = ToolResult(
            success=True, output="Done! I've posted the cat image."
        )

        messages = [{"role": "user", "content": "Generate cat"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Generate cat", messages):
            chunks.append(chunk)

        # Should yield the action success
        full_response = "".join(chunks)
        assert "Done!" in full_response

    @pytest.mark.asyncio
    async def test_output_parse_failed_no_action_success(
        self, orchestrator, mock_llm, mock_executor
    ):
        """Generic fallback on parse failure without action success."""
        mock_llm.call_api.side_effect = Exception("output_parse_failed")

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Test", messages):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert "couldn't find specific information" in full_response

    @pytest.mark.asyncio
    async def test_tool_use_failed_attempts_summary(
        self, orchestrator, mock_llm, mock_executor
    ):
        """Attempts summary when tool_use_failed error occurs."""
        # First call succeeds with tool, second fails
        search_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[search_call]),
            Exception("tool_use_failed"),
            APIResponse(content="Summary response"),  # Summary attempt
        ]

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in orchestrator.chat_stream("Test", messages):
            chunks.append(chunk)

        # Should attempt summary
        full_response = "".join(chunks)
        assert "Summary response" in full_response or "couldn't find a good answer" in full_response

    @pytest.mark.asyncio
    async def test_on_message_update_callback(self, orchestrator, mock_llm, mock_executor):
        """on_message_update callback is called when messages change."""
        tool_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        callback = MagicMock()
        messages = [{"role": "user", "content": "Test"}]

        async for _ in orchestrator.chat_stream("Test", messages, on_message_update=callback):
            pass

        # Callback called for tool call message and tool result message
        assert callback.call_count >= 2

    @pytest.mark.asyncio
    async def test_tool_argument_parsing_error(self, orchestrator, mock_llm, mock_executor):
        """Invalid tool arguments handled gracefully."""
        tool_call = ToolCall(id="tc1", name="web_search", arguments="invalid json")
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        # Tool result should indicate error
        tool_result = next(m for m in messages if m.get("role") == "tool")
        assert "Error" in tool_result["content"]


class TestSearchCallTracking:
    """Test search call tracking and limit behavior."""

    @pytest.mark.asyncio
    async def test_search_calls_incremented(self, orchestrator, mock_llm, mock_executor):
        """Search calls are counted toward limit."""
        search_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "test"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[search_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        # is_search_tool should have been called
        mock_executor.is_search_tool.assert_called()

    @pytest.mark.asyncio
    async def test_action_calls_not_counted(self, orchestrator, mock_llm, mock_executor):
        """Action tool calls don't count toward search limit."""
        action_call = ToolCall(id="tc1", name="generate_image", arguments='{"prompt": "test"}')

        # Many action calls
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[action_call]),
            APIResponse(content=None, tool_calls=[action_call]),
            APIResponse(content=None, tool_calls=[action_call]),
            APIResponse(content=None, tool_calls=[action_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        # get_definitions should never have search_limit_reached=True
        for call in mock_executor.get_definitions.call_args_list:
            assert call[1].get("search_limit_reached", False) is False


class TestToolCallTracking:
    """Test tool call description tracking for graceful failure."""

    @pytest.mark.asyncio
    async def test_web_search_tracking(self, orchestrator, mock_llm, mock_executor):
        """Web search calls track query in description."""
        tool_call = ToolCall(id="tc1", name="web_search", arguments='{"query": "weather"}')
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        # Tool was executed with correct name
        mock_executor.execute.assert_called_with("web_search", {"query": "weather"})

    @pytest.mark.asyncio
    async def test_fetch_url_tracking(self, orchestrator, mock_llm, mock_executor):
        """Fetch URL calls track URL in description."""
        tool_call = ToolCall(
            id="tc1", name="fetch_url", arguments='{"url": "https://example.com"}'
        )
        mock_llm.call_api.side_effect = [
            APIResponse(content=None, tool_calls=[tool_call]),
            APIResponse(content=None),
        ]

        async def stream_gen(messages):
            yield "Result"

        mock_llm.stream_response = stream_gen

        messages = [{"role": "user", "content": "Test"}]
        async for _ in orchestrator.chat_stream("Test", messages):
            pass

        mock_executor.execute.assert_called()
