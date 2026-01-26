"""Search tool handlers for web search and URL fetching."""

import logging
from typing import Any, Optional

from ..base import ToolDefinition, ToolResult, ToolHandler

logger = logging.getLogger(__name__)


class WebSearchHandler(ToolHandler):
    """Handler for web search tool with search limit tracking.

    Limits search calls per request to prevent infinite search loops.
    Call reset_count() at the start of each new conversation turn.
    """

    def __init__(
        self,
        search_service: Any,  # SearchService, but avoid circular import
        max_searches: int = 3,
    ):
        """Initialize web search handler.

        Args:
            search_service: SearchService instance for executing searches
            max_searches: Maximum searches allowed per request (default 3)
        """
        self._search_service = search_service
        self._max_searches = max_searches
        self._search_count = 0

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information. Use for questions about current events, recent data, or anything that might have changed since training. Returns search results with snippets and extracted content from top result.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up",
                    }
                },
                "required": ["query"],
            },
        )

    @property
    def search_count(self) -> int:
        """Current number of searches this request."""
        return self._search_count

    @property
    def limit_reached(self) -> bool:
        """Whether search limit has been reached."""
        return self._search_count >= self._max_searches

    def reset_count(self) -> None:
        """Reset search count for new request.

        Call this at the start of each new conversation turn.
        """
        self._search_count = 0

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute web search.

        Args:
            args: Must contain "query" key with search query string

        Returns:
            ToolResult with search results or error if limit reached
        """
        # Check search limit
        if self.limit_reached:
            logger.warning(f"Search limit reached ({self._search_count}/{self._max_searches})")
            return ToolResult(
                success=False,
                output="",
                error=f"Search limit reached ({self._max_searches} searches). Use the information already gathered to answer the question.",
                metadata={"limit_reached": True, "search_count": self._search_count},
            )

        query = args.get("query", "").strip()
        if not query:
            return ToolResult(
                success=False,
                output="",
                error="Search query cannot be empty",
            )

        # Increment count BEFORE executing (in case of error)
        self._search_count += 1
        logger.info(f"Executing web search ({self._search_count}/{self._max_searches}): {query}")

        try:
            # Use search_and_fetch to get full content from top result via Firecrawl
            result = await self._search_service.search_and_fetch(query, fetch_top_n=1)

            return ToolResult(
                success=True,
                output=result,
                metadata={
                    "query": query,
                    "search_count": self._search_count,
                    "limit_reached": self.limit_reached,
                },
            )
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {str(e)}",
                metadata={"query": query, "search_count": self._search_count},
            )


class FetchUrlHandler(ToolHandler):
    """Handler for fetching URL content.

    Uses Firecrawl for smart content extraction.
    Shares search limit tracking with WebSearchHandler.
    """

    def __init__(
        self,
        search_service: Any,  # SearchService, but avoid circular import
        web_search_handler: Optional[WebSearchHandler] = None,
    ):
        """Initialize URL fetch handler.

        Args:
            search_service: SearchService instance for fetching URLs
            web_search_handler: Optional WebSearchHandler to share limit tracking
        """
        self._search_service = search_service
        self._web_search_handler = web_search_handler

    @property
    def name(self) -> str:
        return "fetch_url"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fetch_url",
            description="Fetch and extract content from a specific URL. Use when you need detailed information from a particular webpage. Returns extracted text content from the page.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query context for smarter extraction",
                    },
                },
                "required": ["url"],
            },
        )

    @property
    def limit_reached(self) -> bool:
        """Whether search limit has been reached (shared with web search)."""
        if self._web_search_handler:
            return self._web_search_handler.limit_reached
        return False

    def reset_count(self) -> None:
        """Reset count via web search handler if linked."""
        # Count is managed by WebSearchHandler, this is a no-op
        pass

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Fetch URL content.

        Args:
            args: Must contain "url" key, optional "query" for smart extraction

        Returns:
            ToolResult with page content or error
        """
        # Check search limit (shared with web search)
        if self.limit_reached:
            logger.warning("URL fetch blocked - search limit reached")
            return ToolResult(
                success=False,
                output="",
                error="Search limit reached. Use the information already gathered.",
                metadata={"limit_reached": True},
            )

        url = args.get("url", "").strip()
        if not url:
            return ToolResult(
                success=False,
                output="",
                error="URL cannot be empty",
            )

        query = args.get("query", "")

        # Increment web search count if linked (fetch counts toward search limit)
        if self._web_search_handler:
            self._web_search_handler._search_count += 1
            logger.info(f"Fetching URL (search count now {self._web_search_handler._search_count}): {url}")
        else:
            logger.info(f"Fetching URL: {url}")

        try:
            result = await self._search_service.fetch_url(url, query)

            # Check for fetch failure
            if result.startswith("Could not fetch"):
                return ToolResult(
                    success=False,
                    output="",
                    error=result,
                    metadata={"url": url},
                )

            return ToolResult(
                success=True,
                output=result,
                metadata={"url": url, "query": query},
            )
        except Exception as e:
            logger.error(f"URL fetch error: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to fetch URL: {str(e)}",
                metadata={"url": url},
            )
