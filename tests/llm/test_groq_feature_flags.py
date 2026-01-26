"""Integration tests for GroqClient feature flags.

Tests that the feature flags correctly toggle between old and new code paths.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Any

from src.tools import ToolRegistry, ToolExecutor
from src.tools.base import ToolDefinition, ToolResult, ToolHandler


class MockSearchService:
    """Mock search service for testing."""

    search_and_fetch = AsyncMock(return_value="Search results")
    fetch_url = AsyncMock(return_value="Page content")


class MockToolHandler(ToolHandler):
    """Mock tool handler for testing."""

    def __init__(self, name: str, result: str = "Mock result"):
        self._name = name
        self._result = result
        self.execute_calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"Mock {self._name} tool",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        self.execute_calls.append(args)
        return ToolResult(success=True, output=self._result)


class TestFeatureFlagsDisabled:
    """Tests with feature flags disabled (default behavior)."""

    @pytest.fixture
    def mock_settings(self):
        """Create settings with feature flags disabled."""
        mock = MagicMock()
        mock.use_new_tool_registry = False
        mock.use_new_prompt_builder = False
        return mock

    def test_no_executor_created_when_flag_disabled(self, mock_settings):
        """ToolExecutor is NOT created when flag is disabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            registry.register(MockToolHandler("web_search"))

            client = GroqClient(
                api_key="test-key",
                search_service=MockSearchService(),
                tool_registry=registry,
            )

            assert client._tool_executor is None

    def test_no_prompt_builder_created_when_flag_disabled(self, mock_settings):
        """SystemPromptBuilder is NOT created when flag is disabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            client = GroqClient(
                api_key="test-key",
                search_service=MockSearchService(),
            )

            assert client._prompt_builder is None

    def test_old_system_prompt_used_when_flag_disabled(self, mock_settings):
        """Old _build_system_prompt is used when flag is disabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            client = GroqClient(api_key="test-key")
            prompt = client._build_system_prompt()

            # Old prompt starts with "You are Jarvis"
            assert "You are Jarvis" in prompt
            # Old prompt has inline tool descriptions
            assert "web_search" in prompt


class TestFeatureFlagsEnabled:
    """Tests with feature flags enabled."""

    @pytest.fixture
    def mock_settings(self):
        """Create settings with feature flags enabled."""
        mock = MagicMock()
        mock.use_new_tool_registry = True
        mock.use_new_prompt_builder = True
        return mock

    def test_executor_created_when_flag_enabled(self, mock_settings):
        """ToolExecutor IS created when flag is enabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            registry.register(MockToolHandler("web_search"))

            client = GroqClient(
                api_key="test-key",
                search_service=MockSearchService(),
                tool_registry=registry,
            )

            assert client._tool_executor is not None
            assert isinstance(client._tool_executor, ToolExecutor)

    def test_prompt_builder_created_when_flag_enabled(self, mock_settings):
        """SystemPromptBuilder IS created when flag is enabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            client = GroqClient(api_key="test-key")

            assert client._prompt_builder is not None

    def test_new_system_prompt_used_when_flag_enabled(self, mock_settings):
        """New SystemPromptBuilder is used when flag is enabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            client = GroqClient(api_key="test-key")
            prompt = client._build_system_prompt()

            # New prompt builder also contains Jarvis (same personality)
            assert "Jarvis" in prompt
            # New prompt should have tool descriptions
            assert "search" in prompt.lower()

    def test_no_executor_without_registry(self, mock_settings):
        """ToolExecutor NOT created if no registry provided even with flag on."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings):
            from src.llm.groq_client import GroqClient

            # No tool_registry parameter
            client = GroqClient(api_key="test-key")

            assert client._tool_executor is None


class TestToolExecutionRouting:
    """Tests for tool execution routing between old and new paths."""

    @pytest.fixture
    def mock_settings_enabled(self):
        """Create settings with tool registry flag enabled."""
        mock = MagicMock()
        mock.use_new_tool_registry = True
        mock.use_new_prompt_builder = False
        return mock

    @pytest.fixture
    def mock_settings_disabled(self):
        """Create settings with tool registry flag disabled."""
        mock = MagicMock()
        mock.use_new_tool_registry = False
        mock.use_new_prompt_builder = False
        return mock

    @pytest.mark.asyncio
    async def test_execute_tool_uses_executor_when_enabled(self, mock_settings_enabled):
        """_execute_tool uses ToolExecutor when flag is enabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_enabled):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            handler = MockToolHandler("web_search", "New executor result")
            registry.register(handler)

            client = GroqClient(
                api_key="test-key",
                tool_registry=registry,
            )

            # Create mock tool call
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "web_search"
            mock_tool_call.function.arguments = '{"query": "test"}'

            result = await client._execute_tool(mock_tool_call)

            # Should use new executor path
            assert result == "New executor result"
            assert len(handler.execute_calls) == 1
            assert handler.execute_calls[0] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_execute_tool_uses_old_path_when_disabled(self, mock_settings_disabled):
        """_execute_tool uses old inline code when flag is disabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_disabled):
            from src.llm.groq_client import GroqClient

            search_service = MockSearchService()
            search_service.search_and_fetch = AsyncMock(return_value="Old path result")

            client = GroqClient(
                api_key="test-key",
                search_service=search_service,
            )

            # Create mock tool call
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "web_search"
            mock_tool_call.function.arguments = '{"query": "test"}'

            result = await client._execute_tool(mock_tool_call)

            # Should use old path
            assert result == "Old path result"
            search_service.search_and_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_executor_handles_tool_alias(self, mock_settings_enabled):
        """ToolExecutor correctly handles tool aliases."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_enabled):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            handler = MockToolHandler("fetch_url", "Fetched content")
            registry.register(handler)

            client = GroqClient(
                api_key="test-key",
                tool_registry=registry,
            )

            # Use aliased name
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "web_fetch_url"  # Alias
            mock_tool_call.function.arguments = '{"url": "http://test.com"}'

            result = await client._execute_tool(mock_tool_call)

            # Should resolve alias and use fetch_url handler
            assert result == "Fetched content"
            assert len(handler.execute_calls) == 1

    @pytest.mark.asyncio
    async def test_executor_returns_error_for_unknown_tool(self, mock_settings_enabled):
        """ToolExecutor returns error for unknown tool."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_enabled):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            client = GroqClient(
                api_key="test-key",
                tool_registry=registry,
            )

            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "nonexistent_tool"
            mock_tool_call.function.arguments = "{}"

            result = await client._execute_tool(mock_tool_call)

            assert "Error:" in result
            assert "nonexistent_tool" in result or "Unknown" in result


class TestAvailableToolsRouting:
    """Tests for available tools selection between old and new paths."""

    @pytest.fixture
    def mock_settings_enabled(self):
        """Create settings with tool registry flag enabled."""
        mock = MagicMock()
        mock.use_new_tool_registry = True
        mock.use_new_prompt_builder = False
        return mock

    def test_get_definitions_called_when_enabled(self, mock_settings_enabled):
        """get_definitions is called on executor when flag is enabled."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_enabled):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            registry.register(MockToolHandler("web_search"))
            registry.register(MockToolHandler("generate_image"))

            client = GroqClient(
                api_key="test-key",
                search_service=MockSearchService(),
                tool_registry=registry,
            )

            # Get definitions via executor
            defs = client._tool_executor.get_definitions()

            assert len(defs) == 2
            names = [d["function"]["name"] for d in defs]
            assert "web_search" in names
            assert "generate_image" in names

    def test_definitions_filtered_when_search_limit_reached(self, mock_settings_enabled):
        """Definitions exclude search tools when limit reached."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_enabled):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            registry.register(MockToolHandler("web_search"))
            registry.register(MockToolHandler("fetch_url"))
            registry.register(MockToolHandler("generate_image"))

            client = GroqClient(
                api_key="test-key",
                tool_registry=registry,
            )

            # Get definitions with search limit
            defs = client._tool_executor.get_definitions(search_limit_reached=True)

            names = [d["function"]["name"] for d in defs]
            assert "web_search" not in names
            assert "fetch_url" not in names
            assert "generate_image" in names


class TestCounterReset:
    """Tests for counter reset behavior."""

    @pytest.fixture
    def mock_settings_enabled(self):
        """Create settings with tool registry flag enabled."""
        mock = MagicMock()
        mock.use_new_tool_registry = True
        mock.use_new_prompt_builder = False
        return mock

    def test_reset_counters_called_on_executor(self, mock_settings_enabled):
        """reset_counters is called on executor at request start."""
        with patch("src.llm.groq_client.get_settings", return_value=mock_settings_enabled):
            from src.llm.groq_client import GroqClient

            registry = ToolRegistry()
            mock_handler = MockToolHandler("web_search")
            mock_handler.reset_count = MagicMock()
            registry.register(mock_handler)

            client = GroqClient(
                api_key="test-key",
                tool_registry=registry,
            )

            # Reset counters
            client._tool_executor.reset_counters()

            mock_handler.reset_count.assert_called_once()
