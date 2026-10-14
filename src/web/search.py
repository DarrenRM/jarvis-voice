"""Web search via ddgs. No API key, no hosted service.

ddgs' default "auto" backend is a metasearch: Wikipedia first, then a rotating
mix of Bing, Brave, DuckDuckGo, Google, Mojeek, Yahoo and Yandex.
"""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class WebSearcher:
    """Runs ddgs text searches off the event loop."""

    def __init__(self, max_results: int = 5, timeout: float = 10.0, region: str = "us-en"):
        self._max_results = max_results
        self._timeout = timeout
        self._region = region

    async def search(self, query: str) -> list[SearchHit]:
        """Return up to max_results hits. An empty list means nothing was found.

        Raises:
            TimeoutError: if the search takes too long
        """
        rows = await asyncio.wait_for(asyncio.to_thread(self._search_sync, query),
                                      timeout=self._timeout + 5)
        return [
            SearchHit(
                title=(row.get("title") or "").strip(),
                url=(row.get("href") or row.get("url") or "").strip(),
                snippet=(row.get("body") or "").strip(),
            )
            for row in rows
            if row.get("href") or row.get("url")
        ]

    def _search_sync(self, query: str) -> list[dict]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException

        try:
            return DDGS(timeout=self._timeout).text(
                query, region=self._region, max_results=self._max_results
            ) or []
        except DDGSException as e:
            # ddgs raises when no backend returns results
            logger.info(f"Web search returned nothing for {query!r}: {e}")
            return []
