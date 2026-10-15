"""Web search via ddgs. No API key, no hosted service.

ddgs' default "auto" backend is a metasearch: Wikipedia first, then a rotating
mix of Bing, Brave, DuckDuckGo, Google, Mojeek, Yahoo and Yandex.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# A voice answer can't wait long, but a throttled metasearch often needs
# 3-8s; 6s cut off searches that would have answered. ddgs gets the per-request
# timeout; the whole search is abandoned after the cap.
REQUEST_TIMEOUT_SECONDS = 8.0
SEARCH_CAP_SECONDS = 10.0
# Identical searches (follow-ups, retries) reuse results for a few minutes
CACHE_TTL_SECONDS = 300.0
CACHE_MAX_ENTRIES = 100


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class WebSearcher:
    """Web search: a hosted backend (TinyFish) when configured, else ddgs off the event loop.

    If the hosted backend fails, the same search falls back to ddgs.
    """

    def __init__(
        self,
        max_results: int = 5,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        region: str = "us-en",
        cap: float = SEARCH_CAP_SECONDS,
        cache_ttl: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        remote: Optional[Any] = None,
    ):
        """
        Args:
            remote: Hosted backend with `async search(query, max_results)` (TinyFishClient)
        """
        self._remote = remote
        self._max_results = max_results
        self._timeout = timeout
        self._region = region
        self._cap = cap
        self._cache_ttl = cache_ttl
        self._clock = clock
        # normalized query -> (time stored, hits)
        self._cache: dict[str, tuple[float, list["SearchHit"]]] = {}

    async def search(self, query: str) -> list[SearchHit]:
        """Return up to max_results hits. An empty list means nothing was found.

        Raises:
            TimeoutError: if the search takes too long
        """
        key = " ".join(query.lower().split())
        cached = self._cache.get(key)
        if cached and self._clock() - cached[0] < self._cache_ttl:
            logger.info(f"Web search cache hit for {query!r}")
            return list(cached[1])

        hits = await self._search_remote(query) if self._remote is not None else None
        if hits is None:
            # The worker thread can't be killed; on timeout it finishes in the background
            rows = await asyncio.wait_for(asyncio.to_thread(self._search_sync, query),
                                          timeout=self._cap)
            hits = [
                SearchHit(
                    title=(row.get("title") or "").strip(),
                    url=(row.get("href") or row.get("url") or "").strip(),
                    snippet=(row.get("body") or "").strip(),
                )
                for row in rows
                if row.get("href") or row.get("url")
            ]
        if hits:
            self._remember(key, hits)
        return hits

    async def _search_remote(self, query: str) -> Optional[list[SearchHit]]:
        """Hits from the hosted backend, or None to fall back to ddgs."""
        started = time.monotonic()
        try:
            hits = await self._remote.search(query, self._max_results)
        except Exception as e:
            logger.warning(f"Hosted search failed ({e}), falling back to ddgs")
            return None
        logger.info(f"[TIMING] web_search hosted: {time.monotonic() - started:.2f}s "
                    f"{len(hits)} hits")
        return hits

    def _remember(self, key: str, hits: list[SearchHit]) -> None:
        now = self._clock()
        self._cache = {k: v for k, v in self._cache.items() if now - v[0] < self._cache_ttl}
        while len(self._cache) >= CACHE_MAX_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (now, list(hits))

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
