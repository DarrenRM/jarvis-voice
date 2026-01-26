"""Tool registry for managing tool handlers.

Provides registration, lookup, and definition export for LLM tools.
"""

from typing import Optional
import logging

from .base import ToolHandler

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool handlers.

    Supports:
    - Registration by name
    - Lookup for execution
    - Definition export for LLM
    - Enable/disable individual tools

    Example:
        registry = ToolRegistry()
        registry.register(WebSearchHandler(search_service))
        registry.register(ImageGenHandler(image_client))

        # Get handler for execution
        handler = registry.get("web_search")
        if handler:
            result = await handler.execute(args)

        # Get definitions for LLM
        tools = registry.get_definitions()
        # Pass to LLM API
    """

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}
        self._disabled: set[str] = set()

    def register(self, handler: ToolHandler) -> None:
        """Register a tool handler.

        Args:
            handler: ToolHandler instance to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if handler.name in self._handlers:
            raise ValueError(f"Tool '{handler.name}' already registered")

        self._handlers[handler.name] = handler
        logger.debug(f"Registered tool: {handler.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool handler.

        Args:
            name: Name of tool to unregister
        """
        self._handlers.pop(name, None)
        self._disabled.discard(name)
        logger.debug(f"Unregistered tool: {name}")

    def disable(self, name: str) -> None:
        """Disable a tool (keeps registered but won't be returned by get()).

        Useful for temporarily removing tools without losing registration.

        Args:
            name: Name of tool to disable
        """
        self._disabled.add(name)
        logger.debug(f"Disabled tool: {name}")

    def enable(self, name: str) -> None:
        """Re-enable a disabled tool.

        Args:
            name: Name of tool to enable
        """
        self._disabled.discard(name)
        logger.debug(f"Enabled tool: {name}")

    def get(self, name: str) -> Optional[ToolHandler]:
        """Get handler by name.

        Returns None if not found or disabled.

        Args:
            name: Tool name to look up

        Returns:
            ToolHandler if found and enabled, None otherwise
        """
        if name in self._disabled:
            return None
        return self._handlers.get(name)

    def get_definitions(self) -> list[dict]:
        """Get all enabled tool definitions for LLM.

        Returns definitions in the format expected by LLM APIs
        (OpenAI-compatible tool format).

        Returns:
            List of tool definition dicts
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": handler.definition.name,
                    "description": handler.definition.description,
                    "parameters": handler.definition.parameters,
                },
            }
            for name, handler in self._handlers.items()
            if name not in self._disabled
        ]

    @property
    def enabled_tools(self) -> list[str]:
        """List of enabled tool names."""
        return [n for n in self._handlers if n not in self._disabled]

    @property
    def all_tools(self) -> list[str]:
        """List of all registered tool names (including disabled)."""
        return list(self._handlers.keys())

    def is_disabled(self, name: str) -> bool:
        """Check if a tool is disabled.

        Args:
            name: Tool name to check

        Returns:
            True if disabled, False otherwise
        """
        return name in self._disabled

    def __len__(self) -> int:
        """Number of registered tools (including disabled)."""
        return len(self._handlers)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered (regardless of disabled state)."""
        return name in self._handlers
