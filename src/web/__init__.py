"""Local, credit-free web access: ddgs metasearch and Trafilatura page reading."""

from .reader import PageReader, WebPage, WebReadError
from .search import SearchHit, WebSearcher

__all__ = ["PageReader", "WebPage", "WebReadError", "SearchHit", "WebSearcher"]
