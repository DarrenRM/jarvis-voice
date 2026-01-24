"""DuckDuckGo search using ddgs package."""

import asyncio
import logging

logger = logging.getLogger(__name__)


class DuckDuckGoSearch:
    """DuckDuckGo search client.

    Uses the ddgs package for free web search without API keys.
    """

    def __init__(self, max_results: int = 3):
        """Initialize DuckDuckGo search.

        Args:
            max_results: Maximum number of results to return
        """
        self.max_results = max_results

    async def search(self, query: str) -> list[dict]:
        """Search DuckDuckGo and return results.

        Args:
            query: Search query string

        Returns:
            List of search results with title, url, and snippet
        """
        from ddgs import DDGS

        loop = asyncio.get_running_loop()

        def do_search():
            try:
                results = DDGS().text(query, max_results=self.max_results)
                return results
            except Exception as e:
                logger.error(f"DuckDuckGo search error: {e}")
                return []

        try:
            results = await loop.run_in_executor(None, do_search)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"DuckDuckGo executor error: {e}")
            return []
