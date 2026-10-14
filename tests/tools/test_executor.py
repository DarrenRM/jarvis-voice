"""Tests for ToolExecutor."""

import pytest

from src.tools import ToolDefinition, ToolExecutor, ToolHandler, ToolRegistry, ToolResult


class StubHandler(ToolHandler):
    def __init__(self, name="stub", result=None, error=None):
        self._name = name
        self._result = result or ToolResult(success=True, output="ok")
        self._error = error
        self.calls = []

    @property
    def name(self):
        return self._name

    @property
    def definition(self):
        return ToolDefinition(name=self._name, description="stub", parameters={"type": "object"})

    async def execute(self, args):
        self.calls.append(args)
        if self._error:
            raise self._error
        return self._result


@pytest.fixture
def registry():
    return ToolRegistry()


class TestExecute:
    async def test_dispatches_to_handler(self, registry):
        handler = StubHandler()
        registry.register(handler)
        result = await ToolExecutor(registry).execute("stub", {"a": 1})
        assert result.success and result.output == "ok"
        assert handler.calls == [{"a": 1}]

    async def test_unknown_tool(self, registry):
        result = await ToolExecutor(registry).execute("missing", {})
        assert not result.success
        assert result.error == "Unknown tool: missing"

    async def test_handler_exception_becomes_error(self, registry):
        registry.register(StubHandler(error=ValueError("bad")))
        result = await ToolExecutor(registry).execute("stub", {})
        assert not result.success
        assert result.error == "Tool error: bad"
        assert result.metadata["exception_type"] == "ValueError"

    async def test_disabled_tool_is_unknown(self, registry):
        registry.register(StubHandler())
        registry.disable("stub")
        result = await ToolExecutor(registry).execute("stub", {})
        assert not result.success


class TestDefinitions:
    def test_returns_enabled_definitions(self, registry):
        registry.register(StubHandler("a"))
        registry.register(StubHandler("b"))
        registry.disable("b")
        defs = ToolExecutor(registry).get_definitions()
        assert [d["function"]["name"] for d in defs] == ["a"]
        assert defs[0]["type"] == "function"

    def test_empty_registry(self, registry):
        assert ToolExecutor(registry).get_definitions() == []
