"""Tests for WebSearcher's time cap and result cache (ddgs is faked)."""

import threading

import pytest

from src.web.search import WebSearcher

ROWS = [{"title": "Chicago weather", "href": "https://example.com/w", "body": "68F sunny"}]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_searcher(rows=ROWS, **kwargs):
    clock = Clock()
    searcher = WebSearcher(clock=clock, **kwargs)
    calls = []

    def fake_search_sync(query):
        calls.append(query)
        return list(rows)

    searcher._search_sync = fake_search_sync
    return searcher, calls, clock


class TestCache:
    async def test_identical_query_reuses_results(self):
        searcher, calls, _ = make_searcher()
        first = await searcher.search("Weather in  Chicago")
        second = await searcher.search("weather in chicago")
        assert len(calls) == 1
        assert second == first

    async def test_cache_expires(self):
        searcher, calls, clock = make_searcher()
        await searcher.search("weather in chicago")
        clock.now += 301
        await searcher.search("weather in chicago")
        assert len(calls) == 2

    async def test_empty_results_not_cached(self):
        searcher, calls, _ = make_searcher(rows=[])
        await searcher.search("nothing here")
        await searcher.search("nothing here")
        assert len(calls) == 2


class TestCap:
    async def test_slow_search_times_out_at_cap(self):
        searcher = WebSearcher(cap=0.05)
        release = threading.Event()
        searcher._search_sync = lambda q: release.wait(2) and []
        try:
            with pytest.raises(TimeoutError):
                await searcher.search("slow")
        finally:
            release.set()
