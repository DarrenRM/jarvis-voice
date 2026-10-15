"""TinyFish hosted Search and Fetch.

Searches and page reads run on TinyFish's servers, so the search-engine
throttling and bot walls this machine's IP hits don't apply. Free tier:
12,000 searches and 1,000 fetched URLs per day. Docs: https://docs.tinyfish.ai
"""

import logging
from typing import Optional

import httpx

from .search import SearchHit

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.search.tinyfish.ai"
FETCH_URL = "https://api.fetch.tinyfish.ai"
# Measured: search 1-2.6s on a new query, fetch ~0.6s for a wiki page. JS-heavy pages
# (Reddit) can take 10-20s to render; too slow for a spoken answer, so give up early.
SEARCH_TIMEOUT = 6.0
FETCH_TIMEOUT = 6.0


class TinyFishError(Exception):
    """TinyFish could not answer; callers fall back to the local path."""


class TinyFishClient:
    """Thin async client for the two free TinyFish APIs."""

    def __init__(self, api_key: str, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._api_key = api_key
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"X-API-Key": self._api_key},
                trust_env=False,
                transport=self._transport,
            )
        return self._client

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        """Ranked results for a query. Raises TinyFishError on any failure."""
        try:
            response = await self._http().get(
                SEARCH_URL, params={"query": query}, timeout=SEARCH_TIMEOUT
            )
            response.raise_for_status()
            rows = response.json().get("results") or []
        except (httpx.HTTPError, ValueError) as e:
            raise TinyFishError(f"search failed: {_describe(e)}") from None
        return [
            SearchHit(
                title=(row.get("title") or "").strip(),
                url=(row.get("url") or "").strip(),
                snippet=(row.get("snippet") or "").strip(),
            )
            for row in rows[:max_results]
            if row.get("url")
        ]

    async def fetch(self, url: str) -> tuple[str, str, str]:
        """Return (final_url, title, markdown text) for one page.

        Raises:
            TinyFishError: request failed, or TinyFish couldn't read the page
        """
        try:
            response = await self._http().post(
                FETCH_URL, json={"urls": [url], "format": "markdown"}, timeout=FETCH_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as e:
            raise TinyFishError(f"fetch failed: {_describe(e)}") from None
        results = data.get("results") or []
        if not results:
            errors = data.get("errors") or [{}]
            raise TinyFishError(f"could not read page: {errors[0].get('error', 'unknown')}")
        page = results[0]
        final_url = page.get("final_url") or page.get("url") or url
        return final_url, (page.get("title") or "").strip(), (page.get("text") or "").strip()

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


def _describe(error: Exception) -> str:
    """Short reason that never includes request headers (the API key)."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__
