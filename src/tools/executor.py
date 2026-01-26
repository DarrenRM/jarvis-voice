"""Tool executor for dispatching tool calls with error handling and aliasing."""

import logging
from typing import Any, Optional

from .base import ToolResult, ToolHandler
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# Tool name aliases - some models may use different names for the same tool
TOOL_ALIASES: dict[str, str] = {
    "web_fetch_url": "fetch_url",
    "web_fetch": "fetch_url",
}

# Tools that track toward search limit
SEARCH_TOOLS = {"web_search", "fetch_url"} | set(TOOL_ALIASES.keys())

# Action tools that don't count toward search limit
ACTION_TOOLS = {
    "generate_image",
    "edit_image",
    "tag_user",
    "post_to_chat",
    "post_table",
    "deep_research",
}


class ToolExecutor:
    """Dispatches tool calls with aliasing, error handling, and counter tracking.

    Features:
    - Tool name aliasing for model compatibility
    - Automatic error handling with graceful ToolResult responses
    - Per-request counter reset for search limits
    - Unknown tool handling
    """

    def __init__(self, registry: ToolRegistry):
        """Initialize tool executor.

        Args:
            registry: Tool registry containing registered handlers
        """
        self._registry = registry

    def normalize_name(self, tool_name: str) -> str:
        """Normalize tool name by applying aliases.

        Args:
            tool_name: Original tool name (may be an alias)

        Returns:
            Canonical tool name
        """
        return TOOL_ALIASES.get(tool_name, tool_name)

    def is_search_tool(self, tool_name: str) -> bool:
        """Check if tool counts toward search limit.

        Args:
            tool_name: Tool name to check

        Returns:
            True if tool counts toward search limit
        """
        normalized = self.normalize_name(tool_name)
        return normalized in SEARCH_TOOLS or tool_name in SEARCH_TOOLS

    def is_action_tool(self, tool_name: str) -> bool:
        """Check if tool is an action tool (doesn't count toward search limit).

        Args:
            tool_name: Tool name to check

        Returns:
            True if tool is an action tool
        """
        return tool_name in ACTION_TOOLS

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        conversation_history: Optional[list[dict]] = None,
    ) -> ToolResult:
        """Execute a tool by name with automatic error handling.

        Args:
            tool_name: Name of tool to execute (may be an alias)
            args: Arguments to pass to the tool
            conversation_history: Optional conversation context for tools that need it

        Returns:
            ToolResult with success/error information
        """
        # Normalize tool name (handle aliases)
        normalized_name = self.normalize_name(tool_name)
        logger.info(f"Executing tool: {tool_name} -> {normalized_name} with args: {args}")

        # Get handler from registry
        handler = self._registry.get(normalized_name)
        if not handler:
            logger.warning(f"Unknown tool: {tool_name} (normalized: {normalized_name})")
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
                metadata={"original_name": tool_name, "normalized_name": normalized_name},
            )

        # Execute with error handling
        try:
            # Check if handler accepts conversation_history (deep_research does)
            execute_method = handler.execute
            if conversation_history and hasattr(execute_method, "__code__"):
                # Check if method accepts conversation_history parameter
                params = execute_method.__code__.co_varnames[:execute_method.__code__.co_argcount]
                if "conversation_history" in params:
                    return await execute_method(args, conversation_history=conversation_history)

            return await handler.execute(args)

        except Exception as e:
            logger.error(f"Tool execution error ({normalized_name}): {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Tool error: {str(e)}",
                metadata={"tool_name": normalized_name, "exception_type": type(e).__name__},
            )

    def reset_counters(self) -> None:
        """Reset all tool counters for new conversation turn.

        Call this at the start of each new user request to reset
        per-request limits like search count.
        """
        for tool_name in self._registry.all_tools:
            handler = self._registry.get(tool_name)
            if handler and hasattr(handler, "reset_count"):
                handler.reset_count()
                logger.debug(f"Reset counter for {tool_name}")

    def get_available_tools(
        self,
        search_limit_reached: bool = False,
        exclude_research: bool = False,
    ) -> list[str]:
        """Get list of currently available tools.

        Args:
            search_limit_reached: If True, exclude search tools
            exclude_research: If True, exclude deep_research

        Returns:
            List of available tool names
        """
        available = []
        for tool_name in self._registry.enabled_tools:
            # Skip search tools if limit reached
            if search_limit_reached and self.is_search_tool(tool_name):
                continue

            # Skip research if requested
            if exclude_research and tool_name == "deep_research":
                continue

            available.append(tool_name)

        return available

    def get_definitions(
        self,
        search_limit_reached: bool = False,
        exclude_research: bool = False,
    ) -> list[dict]:
        """Get tool definitions for LLM API, filtered by availability.

        Args:
            search_limit_reached: If True, exclude search tools
            exclude_research: If True, exclude deep_research

        Returns:
            List of tool definitions in LLM API format
        """
        available_names = set(self.get_available_tools(search_limit_reached, exclude_research))

        definitions = []
        for defn in self._registry.get_definitions():
            if defn["function"]["name"] in available_names:
                definitions.append(defn)

        return definitions
