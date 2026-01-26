"""Tests for ToolRegistry and base classes."""

import pytest
from typing import Any

from src.tools.base import ToolDefinition, ToolResult, ToolHandler
from src.tools.registry import ToolRegistry


class MockToolHandler(ToolHandler):
    """Mock tool handler for testing."""

    def __init__(self, name: str, description: str = "Mock tool"):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Test parameter"}
                },
                "required": ["query"],
            },
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output=f"Executed {self._name}")


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_create_definition(self):
        """Can create a tool definition."""
        defn = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert defn.name == "test_tool"
        assert defn.description == "A test tool"
        assert defn.parameters == {"type": "object", "properties": {}}


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_create_success_result(self):
        """Can create a successful result."""
        result = ToolResult(success=True, output="Done!")
        assert result.success is True
        assert result.output == "Done!"
        assert result.error is None
        assert result.metadata == {}

    def test_create_failure_result(self):
        """Can create a failure result with error."""
        result = ToolResult(success=False, output="", error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Can include metadata in result."""
        result = ToolResult(
            success=True,
            output="Done!",
            metadata={"query": "test", "results": 5},
        )
        assert result.metadata["query"] == "test"
        assert result.metadata["results"] == 5


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    @pytest.fixture
    def registry(self):
        """Fresh registry instance."""
        return ToolRegistry()

    @pytest.fixture
    def mock_handler(self):
        """Mock tool handler."""
        return MockToolHandler("test_tool", "A test tool")

    def test_register_handler(self, registry, mock_handler):
        """Can register a tool handler."""
        registry.register(mock_handler)
        assert "test_tool" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises_error(self, registry, mock_handler):
        """Registering duplicate name raises ValueError."""
        registry.register(mock_handler)
        with pytest.raises(ValueError) as exc_info:
            registry.register(mock_handler)
        assert "already registered" in str(exc_info.value)

    def test_get_registered_handler(self, registry, mock_handler):
        """Can get a registered handler."""
        registry.register(mock_handler)
        handler = registry.get("test_tool")
        assert handler is mock_handler

    def test_get_unregistered_returns_none(self, registry):
        """Getting unregistered tool returns None."""
        handler = registry.get("nonexistent")
        assert handler is None

    def test_unregister_handler(self, registry, mock_handler):
        """Can unregister a handler."""
        registry.register(mock_handler)
        registry.unregister("test_tool")
        assert "test_tool" not in registry
        assert registry.get("test_tool") is None

    def test_unregister_nonexistent_is_safe(self, registry):
        """Unregistering nonexistent tool doesn't raise."""
        registry.unregister("nonexistent")  # Should not raise


class TestToolRegistryDisable:
    """Tests for tool enable/disable functionality."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockToolHandler("tool1"))
        registry.register(MockToolHandler("tool2"))
        return registry

    def test_disable_tool(self, registry):
        """Disabling a tool makes get() return None."""
        registry.disable("tool1")
        assert registry.get("tool1") is None
        assert registry.get("tool2") is not None

    def test_enable_tool(self, registry):
        """Enabling a tool makes it available again."""
        registry.disable("tool1")
        registry.enable("tool1")
        assert registry.get("tool1") is not None

    def test_disabled_tool_still_registered(self, registry):
        """Disabled tool is still in registry."""
        registry.disable("tool1")
        assert "tool1" in registry
        assert len(registry) == 2

    def test_is_disabled(self, registry):
        """is_disabled returns correct status."""
        assert registry.is_disabled("tool1") is False
        registry.disable("tool1")
        assert registry.is_disabled("tool1") is True

    def test_enabled_tools_list(self, registry):
        """enabled_tools only returns enabled tools."""
        registry.disable("tool1")
        enabled = registry.enabled_tools
        assert "tool1" not in enabled
        assert "tool2" in enabled

    def test_all_tools_list(self, registry):
        """all_tools returns all tools including disabled."""
        registry.disable("tool1")
        all_tools = registry.all_tools
        assert "tool1" in all_tools
        assert "tool2" in all_tools


class TestToolRegistryDefinitions:
    """Tests for get_definitions() method."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockToolHandler("tool1", "First tool"))
        registry.register(MockToolHandler("tool2", "Second tool"))
        return registry

    def test_get_definitions_format(self, registry):
        """Definitions are in correct format."""
        definitions = registry.get_definitions()
        assert len(definitions) == 2

        defn = definitions[0]
        assert defn["type"] == "function"
        assert "function" in defn
        assert "name" in defn["function"]
        assert "description" in defn["function"]
        assert "parameters" in defn["function"]

    def test_get_definitions_excludes_disabled(self, registry):
        """Disabled tools not included in definitions."""
        registry.disable("tool1")
        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "tool2"

    def test_definitions_contain_all_fields(self, registry):
        """Definitions contain name, description, parameters."""
        definitions = registry.get_definitions()
        defn = definitions[0]

        assert defn["function"]["name"] in ["tool1", "tool2"]
        assert len(defn["function"]["description"]) > 0
        assert "type" in defn["function"]["parameters"]
        assert "properties" in defn["function"]["parameters"]


class TestToolHandlerExecution:
    """Tests for tool execution via registry."""

    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register(MockToolHandler("test_tool"))
        return registry

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry):
        """Can execute a registered tool."""
        handler = registry.get("test_tool")
        result = await handler.execute({"query": "test"})
        assert result.success is True
        assert "Executed test_tool" in result.output

    @pytest.mark.asyncio
    async def test_execute_disabled_tool_returns_none(self, registry):
        """Cannot get disabled tool for execution."""
        registry.disable("test_tool")
        handler = registry.get("test_tool")
        assert handler is None
