"""Discord tool handlers for user tagging, message posting, and table formatting."""

import logging
from typing import Any, Callable, Awaitable, Optional

from ..base import ToolDefinition, ToolResult, ToolHandler

logger = logging.getLogger(__name__)


# Type alias for async callbacks
AsyncCallback = Callable[..., Awaitable[str]]


class TagUserHandler(ToolHandler):
    """Handler for tagging/mentioning users in Discord."""

    def __init__(self, on_user_tagged: Optional[AsyncCallback] = None):
        """Initialize tag user handler.

        Args:
            on_user_tagged: Async callback(target_name, message) -> str result
        """
        self._on_user_tagged = on_user_tagged

    @property
    def name(self) -> str:
        return "tag_user"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="tag_user",
            description="Tag or mention a user in the Discord channel. Use this when someone asks you to notify, ping, or get someone's attention.",
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "The name or nickname of the user to tag",
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional message to include with the tag",
                    },
                },
                "required": ["target_name"],
            },
        )

    def set_callback(self, callback: AsyncCallback) -> None:
        """Set the callback for tagging users."""
        self._on_user_tagged = callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute user tag.

        Args:
            args: Must contain "target_name", optionally "message"

        Returns:
            ToolResult with tag result or error
        """
        if not self._on_user_tagged:
            return ToolResult(
                success=False,
                output="",
                error="User tagging not available - not connected to Discord",
            )

        target_name = args.get("target_name", "").strip()
        if not target_name:
            return ToolResult(
                success=False,
                output="",
                error="Target name cannot be empty",
            )

        message = args.get("message", "")
        logger.info(f"Tagging user: {target_name} with message: {message}")

        try:
            result = await self._on_user_tagged(target_name, message)
            return ToolResult(
                success=True,
                output=result,
                metadata={"target_name": target_name, "message": message},
            )
        except Exception as e:
            logger.error(f"Error tagging user: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to tag user: {str(e)}",
            )


class PostMessageHandler(ToolHandler):
    """Handler for posting messages to Discord text channel."""

    def __init__(self, on_message_posted: Optional[AsyncCallback] = None):
        """Initialize post message handler.

        Args:
            on_message_posted: Async callback(message) -> str result
        """
        self._on_message_posted = on_message_posted

    @property
    def name(self) -> str:
        return "post_to_chat"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="post_to_chat",
            description="Post a message to the Discord text channel. Use for sharing links, formatted text, or content that works better written than spoken.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to post to the chat",
                    },
                },
                "required": ["message"],
            },
        )

    def set_callback(self, callback: AsyncCallback) -> None:
        """Set the callback for posting messages."""
        self._on_message_posted = callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute message post.

        Args:
            args: Must contain "message"

        Returns:
            ToolResult with post result or error
        """
        if not self._on_message_posted:
            return ToolResult(
                success=False,
                output="",
                error="Message posting not available - not connected to Discord",
            )

        message = args.get("message", "").strip()
        if not message:
            return ToolResult(
                success=False,
                output="",
                error="Message cannot be empty",
            )

        logger.info(f"Posting message to chat: {message[:50]}...")

        try:
            result = await self._on_message_posted(message)
            return ToolResult(
                success=True,
                output=result,
                metadata={"message_length": len(message)},
            )
        except Exception as e:
            logger.error(f"Error posting message: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to post message: {str(e)}",
            )


class PostTableHandler(ToolHandler):
    """Handler for posting formatted tables to Discord."""

    def __init__(self, on_message_posted: Optional[AsyncCallback] = None):
        """Initialize post table handler.

        Args:
            on_message_posted: Async callback(formatted_table) -> str result
        """
        self._on_message_posted = on_message_posted

    @property
    def name(self) -> str:
        return "post_table"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="post_table",
            description="Post a formatted table to the Discord text channel. Use when presenting structured data, comparisons, or lists that are easier to read as a table.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Optional title shown above the table",
                    },
                    "headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column headers for the table",
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "description": "Table rows, each row is an array of cell values",
                    },
                },
                "required": ["headers", "rows"],
            },
        )

    def set_callback(self, callback: AsyncCallback) -> None:
        """Set the callback for posting messages."""
        self._on_message_posted = callback

    @staticmethod
    def format_table(
        headers: list[str],
        rows: list[list[str]],
        title: Optional[str] = None,
    ) -> str:
        """Format data as a Discord-friendly markdown table.

        Args:
            headers: Column headers
            rows: List of rows, each row is a list of cell values
            title: Optional title shown above the table

        Returns:
            Formatted table string wrapped in code block
        """
        if not headers or not rows:
            return "Empty table"

        # Calculate column widths (min 6, max of header/data)
        widths = []
        for i, h in enumerate(headers):
            col_max = max(
                len(str(h)),
                max((len(str(r[i])) for r in rows if i < len(r)), default=0),
            )
            widths.append(max(col_max + 2, 6))

        # Build table rows
        def row_str(cells: list) -> str:
            return (
                "| "
                + " | ".join(str(c).center(w - 2) for c, w in zip(cells, widths))
                + " |"
            )

        def separator() -> str:
            return "| " + " | ".join("-" * (w - 2) for w in widths) + " |"

        lines = [row_str(headers), separator()]
        lines.extend(row_str(row) for row in rows)

        table = "```\n" + "\n".join(lines) + "\n```"

        if title:
            return f"**{title}**\n{table}"
        return table

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute table post.

        Args:
            args: Must contain "headers" and "rows", optionally "title"

        Returns:
            ToolResult with post result or error
        """
        if not self._on_message_posted:
            return ToolResult(
                success=False,
                output="",
                error="Table posting not available - not connected to Discord",
            )

        headers = args.get("headers", [])
        rows = args.get("rows", [])
        title = args.get("title")

        if not headers:
            return ToolResult(
                success=False,
                output="",
                error="Table headers cannot be empty",
            )

        if not rows:
            return ToolResult(
                success=False,
                output="",
                error="Table rows cannot be empty",
            )

        # Format the table
        formatted = self.format_table(headers, rows, title)
        logger.info(f"Posting table with {len(rows)} rows")

        try:
            result = await self._on_message_posted(formatted)
            return ToolResult(
                success=True,
                output=result,
                metadata={
                    "title": title,
                    "num_columns": len(headers),
                    "num_rows": len(rows),
                },
            )
        except Exception as e:
            logger.error(f"Error posting table: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to post table: {str(e)}",
            )
