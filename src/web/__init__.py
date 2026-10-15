"""Web access: TinyFish hosted search/fetch when keyed, else ddgs metasearch and local
Trafilatura page reading (also the fallback when TinyFish fails)."""

from .reader import PageReader, WebPage, WebReadError
from .search import SearchHit, WebSearcher
from .tinyfish import TinyFishClient, TinyFishError

__all__ = [
    "PageReader", "WebPage", "WebReadError", "SearchHit", "WebSearcher",
    "TinyFishClient", "TinyFishError",
]
