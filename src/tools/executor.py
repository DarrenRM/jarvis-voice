"""Tool executor for dispatching tool calls with error handling."""

import logging
from typing import Any

from .base import ToolResult
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Dispatches tool calls with error handling.

    Features:
    - Automatic error handling with graceful ToolResult responses
    - Unknown tool handling
    """

    def __init__(self, registry: ToolRegistry):
        """Initialize tool executor.

        Args:
            registry: Tool registry containing registered handlers
        """
        self._registry = registry

    async def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with automatic error handling.

        Args:
            tool_name: Name of tool to execute
            args: Arguments to pass to the tool

        Returns:
            ToolResult with success/error information
        """
        logger.info(f"Executing tool: {tool_name} with args: {args}")

        handler = self._registry.get(tool_name)
        if not handler:
            logger.warning(f"Unknown tool: {tool_name}")
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
                metadata={"tool_name": tool_name},
            )

        try:
            return await handler.execute(args)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Tool error: {str(e)}",
                metadata={"tool_name": tool_name, "exception_type": type(e).__name__},
            )

    def get_definitions(self) -> list[dict]:
        """Get definitions of all enabled tools in LLM API format."""
        return self._registry.get_definitions()
