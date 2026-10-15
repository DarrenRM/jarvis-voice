"""Tests for the TinyFish client and the hosted-first, local-fallback wiring. No real network."""

import json

import httpx
import pytest

from src.web import PageReader, TinyFishClient, TinyFishError, WebReadError, WebSearcher
from src.web import tinyfish as tf

from .test_reader import PUBLIC, html_response, resolver

KEY = "test-key-123"


def client(handler):
    return TinyFishClient(KEY, transport=httpx.MockTransport(handler))


def search_response(n=3):
    return httpx.Response(200, json={"query": "q", "results": [
        {"position": i, "site_name": "s", "title": f"T{i}", "snippet": f"S{i}",
         "url": f"https://example.com/{i}"} for i in range(n)]})


class FakeRemote:
    def __init__(self, hits=None, page=None, fail=False):
        self.hits, self.page, self.fail = hits, page, fail
        self.calls = []

    async def search(self, query, max_results):
        self.calls.append(query)
        if self.fail:
            raise TinyFishError("search failed: HTTP 503")
        return self.hits

    async def fetch(self, url):
        self.calls.append(url)
        if self.fail:
            raise TinyFishError("fetch failed: ReadTimeout")
        return self.page


class TestClient:
    async def test_search_sends_key_and_parses_hits(self):
        seen = {}

        def handler(request):
            seen["key"] = request.headers.get("x-api-key")
            seen["query"] = request.url.params.get("query")
            return search_response(7)

        hits = await client(handler).search("enshrouded glider", max_results=5)
        assert seen == {"key": KEY, "query": "enshrouded glider"}
        assert len(hits) == 5
        assert (hits[0].title, hits[0].url, hits[0].snippet) == ("T0", "https://example.com/0", "S0")

    async def test_http_error_raises_without_leaking_key(self):
        with pytest.raises(TinyFishError) as err:
            await client(lambda r: httpx.Response(401, text="bad key")).search("q")
        assert "401" in str(err.value) and KEY not in str(err.value)

    async def test_fetch_returns_page(self):
        def handler(request):
            assert json.loads(request.content) == {"urls": ["https://example.com/a"],
                                                   "format": "markdown"}
            return httpx.Response(200, json={"results": [
                {"url": "https://example.com/a", "final_url": "https://example.com/b",
                 "title": "Gliders", "text": "# Gliders\nShroud Wood x3"}], "errors": []})

        assert await client(handler).fetch("https://example.com/a") == (
            "https://example.com/b", "Gliders", "# Gliders\nShroud Wood x3")

    async def test_fetch_per_url_error_raises(self):
        resp = httpx.Response(200, json={"results": [], "errors": [
            {"url": "https://example.com/a", "error": "anti-bot block"}]})
        with pytest.raises(TinyFishError, match="anti-bot"):
            await client(lambda r: resp).fetch("https://example.com/a")


class TestSearcherWiring:
    @staticmethod
    def searcher(remote):
        s = WebSearcher(remote=remote)
        s.local_calls = []

        def local(query):
            s.local_calls.append(query)
            return [{"title": "ddg", "href": "https://ddg.example/x", "body": "local"}]

        s._search_sync = local
        return s

    async def test_hosted_results_used(self):
        from src.web import SearchHit
        remote = FakeRemote(hits=[SearchHit("T", "https://example.com/t", "S")])
        s = self.searcher(remote)
        hits = await s.search("q")
        assert [h.title for h in hits] == ["T"] and s.local_calls == []

    async def test_falls_back_to_ddgs_when_hosted_fails(self):
        s = self.searcher(FakeRemote(fail=True))
        hits = await s.search("q")
        assert [h.title for h in hits] == ["ddg"] and s.local_calls == ["q"]

    async def test_hosted_results_cached(self):
        from src.web import SearchHit
        remote = FakeRemote(hits=[SearchHit("T", "https://example.com/t", "S")])
        s = self.searcher(remote)
        await s.search("Same  Query")
        await s.search("same query")
        assert remote.calls == ["Same  Query"]


class TestReaderWiring:
    @staticmethod
    def reader(remote, handler=lambda req: html_response(), max_chars=6000):
        return PageReader(max_chars=max_chars, resolver=resolver(PUBLIC),
                          transport=httpx.MockTransport(handler), remote=remote)

    async def test_hosted_page_used_and_truncated(self):
        remote = FakeRemote(page=("https://example.com/b", "Gliders", "word " * 100))
        local = []
        reader = self.reader(remote, handler=lambda r: local.append(r) or html_response(),
                             max_chars=50)
        page = await reader.read("https://example.com/a")
        assert page.url == "https://example.com/b" and page.title == "Gliders"
        assert page.truncated and len(page.text) <= 50
        assert local == []

    async def test_falls_back_to_local_read_when_hosted_fails(self):
        page = await self.reader(FakeRemote(fail=True)).read("https://example.com/darts")
        assert "seven salvaged metal" in page.text

    async def test_empty_hosted_text_reads_locally(self):
        remote = FakeRemote(page=("https://example.com/darts", "", ""))
        page = await self.reader(remote).read("https://example.com/darts")
        assert "seven salvaged metal" in page.text

    async def test_private_address_blocked_before_hosted_call(self):
        remote = FakeRemote(page=("x", "x", "x"))
        reader = PageReader(resolver=resolver({"router.lan": ["192.168.1.1"]}), remote=remote)
        with pytest.raises(WebReadError, match="private"):
            await reader.read("http://router.lan/admin")
        assert remote.calls == []


def test_timeouts_suit_voice():
    assert tf.SEARCH_TIMEOUT <= 6 and tf.FETCH_TIMEOUT <= 6
