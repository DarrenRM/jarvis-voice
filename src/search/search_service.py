"""Unified search service combining DuckDuckGo and Firecrawl."""

import logging
from typing import Optional

from .duckduckgo import DuckDuckGoSearch
from .firecrawl import FirecrawlClient

logger = logging.getLogger(__name__)


class SearchService:
    """Unified search service for LLM tool use.

    Combines DuckDuckGo for web search and Firecrawl for content extraction.
    """

    def __init__(
        self,
        firecrawl_url: str = "http://localhost:3002",
        firecrawl_api_key: Optional[str] = None,
        max_results: int = 3,
        max_content_length: int = 4000,
    ):
        """Initialize search service.

        Args:
            firecrawl_url: URL of self-hosted Firecrawl instance
            firecrawl_api_key: Optional API key for Firecrawl
            max_results: Maximum search results to return
            max_content_length: Maximum content length per URL (truncate if longer)
        """
        self.ddg = DuckDuckGoSearch(max_results=max_results)
        self.firecrawl = FirecrawlClient(
            base_url=firecrawl_url,
            api_key=firecrawl_api_key,
        )
        self.max_content_length = max_content_length

    async def search(self, query: str) -> str:
        """Search the web and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted search results as a string
        """
        results = await self.ddg.search(query)

        if not results:
            return "No search results found."

        # Format results for the model
        output_parts = []
        for i, result in enumerate(results, 1):
            output_parts.append(
                f"[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Snippet: {result['snippet']}"
            )

        return "\n\n".join(output_parts)

    async def fetch_url(self, url: str, query: str = "") -> str:
        """Fetch URL content via Firecrawl with smart extraction.

        Tries structured extraction first, falls back to markdown.

        Args:
            url: URL to fetch content from
            query: Optional query context for better extraction

        Returns:
            Extracted JSON or markdown content, or error message
        """
        if query:
            # Use smart extraction with query context
            return await self.firecrawl.smart_extract(
                url, query, max_markdown_length=self.max_content_length
            )

        # No query - just get markdown
        content = await self.firecrawl.scrape(url)

        if not content:
            return f"Could not fetch content from {url}. Firecrawl may not be running."

        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n\n[Content truncated...]"

        return content

    async def search_and_fetch(self, query: str, fetch_top_n: int = 1) -> str:
        """Search and fetch content from top results using smart extraction.

        Args:
            query: Search query string
            fetch_top_n: Number of top results to fetch full content for (0 to skip)

        Returns:
            Combined search results and fetched/extracted content
        """
        results = await self.ddg.search(query)

        if not results:
            return "No search results found."

        output_parts = []

        # Add search results
        for i, result in enumerate(results, 1):
            output_parts.append(
                f"[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Snippet: {result['snippet']}"
            )

        # Fetch content from top results using smart extraction
        if fetch_top_n > 0:
            output_parts.append("\n--- Extracted Content ---")
            for i, result in enumerate(results[:fetch_top_n], 1):
                content = await self.firecrawl.smart_extract(
                    result['url'], query, max_markdown_length=self.max_content_length
                )
                if content and not content.startswith("Could not fetch"):
                    output_parts.append(f"\n[{i}] From {result['url']}:\n{content}")

        return "\n\n".join(output_parts)
