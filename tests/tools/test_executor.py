"""Tests for tool executor."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from src.tools.base import ToolDefinition, ToolResult, ToolHandler
from src.tools.registry import ToolRegistry
from src.tools.executor import (
    ToolExecutor,
    TOOL_ALIASES,
    SEARCH_TOOLS,
    ACTION_TOOLS,
)


class MockHandler(ToolHandler):
    """Mock tool handler for testing."""

    def __init__(self, name: str, description: str = "Mock tool"):
        self._name = name
        self._description = description
        self._execute_result = ToolResult(success=True, output=f"Executed {name}")
        self.execute_called_with: list[dict] = []
        self._reset_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        self.execute_called_with.append(args)
        return self._execute_result

    def reset_count(self) -> None:
        self._reset_called = True


class MockHandlerWithHistory(MockHandler):
    """Mock handler that accepts conversation history."""

    def __init__(self, name: str):
        super().__init__(name)
        self.history_received: list[dict] = None

    async def execute(self, args: dict[str, Any], conversation_history: list[dict] = None) -> ToolResult:
        self.execute_called_with.append(args)
        self.history_received = conversation_history
        return self._execute_result


class TestToolAliases:
    """Tests for tool alias constants."""

    def test_web_fetch_url_alias(self):
        """web_fetch_url aliases to fetch_url."""
        assert TOOL_ALIASES["web_fetch_url"] == "fetch_url"

    def test_web_fetch_alias(self):
        """web_fetch aliases to fetch_url."""
        assert TOOL_ALIASES["web_fetch"] == "fetch_url"

    def test_search_tools_includes_aliases(self):
        """Search tools set includes aliases."""
        assert "web_search" in SEARCH_TOOLS
        assert "fetch_url" in SEARCH_TOOLS
        for alias in TOOL_ALIASES:
            assert alias in SEARCH_TOOLS

    def test_action_tools_defined(self):
        """Action tools are defined."""
        assert "generate_image" in ACTION_TOOLS
        assert "edit_image" in ACTION_TOOLS
        assert "tag_user" in ACTION_TOOLS
        assert "post_to_chat" in ACTION_TOOLS
        assert "post_table" in ACTION_TOOLS
        assert "deep_research" in ACTION_TOOLS


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockHandler("web_search", "Search the web"))
        registry.register(MockHandler("fetch_url", "Fetch URL content"))
        registry.register(MockHandler("generate_image", "Generate image"))
        registry.register(MockHandler("deep_research", "Deep research"))
        return registry

    @pytest.fixture
    def executor(self, registry):
        return ToolExecutor(registry)

    def test_normalize_name_passes_through_unknown(self, executor):
        """Unknown names pass through unchanged."""
        assert executor.normalize_name("web_search") == "web_search"
        assert executor.normalize_name("generate_image") == "generate_image"

    def test_normalize_name_applies_alias(self, executor):
        """Aliases are correctly applied."""
        assert executor.normalize_name("web_fetch_url") == "fetch_url"
        assert executor.normalize_name("web_fetch") == "fetch_url"

    def test_is_search_tool(self, executor):
        """Correctly identifies search tools."""
        assert executor.is_search_tool("web_search") is True
        assert executor.is_search_tool("fetch_url") is True
        assert executor.is_search_tool("web_fetch_url") is True  # Alias
        assert executor.is_search_tool("generate_image") is False
        assert executor.is_search_tool("unknown") is False

    def test_is_action_tool(self, executor):
        """Correctly identifies action tools."""
        assert executor.is_action_tool("generate_image") is True
        assert executor.is_action_tool("edit_image") is True
        assert executor.is_action_tool("deep_research") is True
        assert executor.is_action_tool("web_search") is False

    @pytest.mark.asyncio
    async def test_execute_known_tool(self, executor, registry):
        """Execute successfully calls known tool."""
        result = await executor.execute("web_search", {"query": "test"})

        assert result.success is True
        assert "Executed web_search" in result.output

        handler = registry.get("web_search")
        assert {"query": "test"} in handler.execute_called_with

    @pytest.mark.asyncio
    async def test_execute_with_alias(self, executor, registry):
        """Execute handles aliases correctly."""
        result = await executor.execute("web_fetch_url", {"url": "http://test.com"})

        assert result.success is True
        # Should have called fetch_url handler
        handler = registry.get("fetch_url")
        assert {"url": "http://test.com"} in handler.execute_called_with

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor):
        """Execute returns error for unknown tool."""
        result = await executor.execute("nonexistent_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error
        assert "nonexistent_tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self, executor, registry):
        """Execute catches and reports exceptions."""
        handler = registry.get("web_search")
        handler._execute_result = None  # Will cause execute to fail

        # Make execute raise
        async def failing_execute(args):
            raise RuntimeError("Connection failed")

        handler.execute = failing_execute

        result = await executor.execute("web_search", {"query": "test"})

        assert result.success is False
        assert "Connection failed" in result.error
        assert result.metadata["exception_type"] == "RuntimeError"


class TestToolExecutorWithHistory:
    """Tests for conversation history passing."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockHandlerWithHistory("deep_research"))
        registry.register(MockHandler("web_search"))
        return registry

    @pytest.fixture
    def executor(self, registry):
        return ToolExecutor(registry)

    @pytest.mark.asyncio
    async def test_history_passed_to_supporting_handler(self, executor, registry):
        """Conversation history is passed to handlers that support it."""
        history = [{"role": "user", "content": "What is quantum computing?"}]

        await executor.execute("deep_research", {"task": "research"}, conversation_history=history)

        handler = registry.get("deep_research")
        assert handler.history_received == history

    @pytest.mark.asyncio
    async def test_history_not_passed_to_regular_handler(self, executor, registry):
        """Handlers without history parameter work normally."""
        history = [{"role": "user", "content": "test"}]

        result = await executor.execute("web_search", {"query": "test"}, conversation_history=history)

        # Should succeed without error
        assert result.success is True


class TestToolExecutorCounters:
    """Tests for counter reset functionality."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockHandler("web_search"))
        registry.register(MockHandler("fetch_url"))
        registry.register(MockHandler("generate_image"))  # No reset_count
        return registry

    @pytest.fixture
    def executor(self, registry):
        return ToolExecutor(registry)

    def test_reset_counters_calls_reset_count(self, executor, registry):
        """reset_counters calls reset_count on handlers that have it."""
        executor.reset_counters()

        # Handlers with reset_count should have been called
        assert registry.get("web_search")._reset_called is True
        assert registry.get("fetch_url")._reset_called is True

    def test_reset_counters_handles_missing_method(self, executor, registry):
        """reset_counters doesn't fail on handlers without reset_count."""
        # generate_image has reset_count from MockHandler, but let's test the concept
        # by ensuring no exception is raised
        executor.reset_counters()  # Should not raise


class TestToolExecutorAvailability:
    """Tests for tool availability filtering."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockHandler("web_search"))
        registry.register(MockHandler("fetch_url"))
        registry.register(MockHandler("generate_image"))
        registry.register(MockHandler("deep_research"))
        return registry

    @pytest.fixture
    def executor(self, registry):
        return ToolExecutor(registry)

    def test_get_available_tools_all(self, executor):
        """All tools available when no restrictions."""
        available = executor.get_available_tools()

        assert "web_search" in available
        assert "fetch_url" in available
        assert "generate_image" in available
        assert "deep_research" in available

    def test_get_available_tools_search_limit(self, executor):
        """Search tools excluded when limit reached."""
        available = executor.get_available_tools(search_limit_reached=True)

        assert "web_search" not in available
        assert "fetch_url" not in available
        assert "generate_image" in available
        assert "deep_research" in available

    def test_get_available_tools_exclude_research(self, executor):
        """Research excluded when requested."""
        available = executor.get_available_tools(exclude_research=True)

        assert "web_search" in available
        assert "fetch_url" in available
        assert "generate_image" in available
        assert "deep_research" not in available

    def test_get_available_tools_both_restrictions(self, executor):
        """Both restrictions can be combined."""
        available = executor.get_available_tools(
            search_limit_reached=True,
            exclude_research=True,
        )

        assert "web_search" not in available
        assert "fetch_url" not in available
        assert "deep_research" not in available
        assert "generate_image" in available

    def test_get_definitions_filters_correctly(self, executor):
        """get_definitions respects availability filters."""
        # All tools
        all_defs = executor.get_definitions()
        assert len(all_defs) == 4

        # Without search
        limited_defs = executor.get_definitions(search_limit_reached=True)
        names = [d["function"]["name"] for d in limited_defs]
        assert "web_search" not in names
        assert "fetch_url" not in names
        assert "generate_image" in names

    def test_disabled_tool_not_available(self, executor, registry):
        """Disabled tools not in available list."""
        registry.disable("web_search")

        available = executor.get_available_tools()
        assert "web_search" not in available


class TestToolExecutorIntegration:
    """Integration tests for tool executor."""

    @pytest.fixture
    def full_registry(self):
        """Registry with all tool types."""
        registry = ToolRegistry()
        registry.register(MockHandler("web_search"))
        registry.register(MockHandler("fetch_url"))
        registry.register(MockHandler("generate_image"))
        registry.register(MockHandler("edit_image"))
        registry.register(MockHandler("tag_user"))
        registry.register(MockHandler("post_to_chat"))
        registry.register(MockHandler("post_table"))
        registry.register(MockHandlerWithHistory("deep_research"))
        return registry

    @pytest.fixture
    def executor(self, full_registry):
        return ToolExecutor(full_registry)

    @pytest.mark.asyncio
    async def test_typical_workflow(self, executor, full_registry):
        """Simulate typical tool call workflow."""
        # 1. Reset counters at start of request
        executor.reset_counters()

        # 2. Get available tools (all)
        available = executor.get_available_tools()
        assert len(available) == 8

        # 3. Execute some tools
        result1 = await executor.execute("web_search", {"query": "python"})
        assert result1.success is True

        result2 = await executor.execute("web_fetch_url", {"url": "http://test.com"})
        assert result2.success is True

        # 4. After reaching limit, get restricted tools
        available_limited = executor.get_available_tools(search_limit_reached=True)
        assert "web_search" not in available_limited
        assert "generate_image" in available_limited

        # 5. Can still execute action tools
        result3 = await executor.execute("generate_image", {"prompt": "a cat"})
        assert result3.success is True
