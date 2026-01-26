"""Base classes for tool system.

Defines the interfaces and data structures for LLM function calling tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolDefinition:
    """Tool schema for LLM function calling.

    Attributes:
        name: Unique tool name (used in function calls)
        description: Human-readable description for LLM context
        parameters: JSON Schema for tool parameters
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolResult:
    """Result from tool execution.

    Attributes:
        success: Whether the tool executed successfully
        output: The output string to return to the LLM
        error: Optional error message if success is False
        metadata: Additional data (e.g., for logging or tracking)
    """

    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ToolHandler(ABC):
    """Abstract base class for tool handlers.

    Each tool handler implements a specific LLM tool (e.g., web_search, generate_image).
    Handlers are registered with a ToolRegistry for dispatch.

    Example:
        class MyToolHandler(ToolHandler):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="my_tool",
                    description="Does something useful",
                    parameters={
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                )

            async def execute(self, args: dict[str, Any]) -> ToolResult:
                # Implementation
                return ToolResult(success=True, output="Done!")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must match definition.name and LLM tool call)."""
        pass

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Tool definition for LLM schema."""
        pass

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            args: Parsed arguments from LLM tool call

        Returns:
            ToolResult with success status and output
        """
        pass
