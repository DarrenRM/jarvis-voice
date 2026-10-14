"""Tests for web_search / web_extract tool handlers."""

from src.tools import create_tool_registry
from src.tools.handlers.web import WebExtractHandler, WebSearchHandler
from src.web import SearchHit, WebPage, WebReadError


class FakeSearcher:
    def __init__(self, hits=None, error=None):
        self.hits = hits or []
        self.error = error
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.hits


class FakeReader:
    def __init__(self, page=None, error=None):
        self.page = page
        self.error = error

    async def read(self, url):
        if self.error:
            raise self.error
        return self.page


class TestWebSearch:
    async def test_formats_hits(self):
        searcher = FakeSearcher([SearchHit("Heavy Darts", "https://wiki/darts", "Seven metal.")])
        result = await WebSearchHandler(searcher).execute({"query": " heavy darts "})
        assert result.success
        assert searcher.queries == ["heavy darts"]
        assert "1. Heavy Darts" in result.output
        assert "https://wiki/darts" in result.output
        assert "not instructions" in result.output

    async def test_no_results_hints_mishearing(self):
        result = await WebSearchHandler(FakeSearcher()).execute({"query": "hevy darz"})
        assert result.success
        assert "misheard" in result.output

    async def test_empty_query(self):
        result = await WebSearchHandler(FakeSearcher()).execute({"query": "  "})
        assert not result.success

    async def test_timeout(self):
        result = await WebSearchHandler(FakeSearcher(error=TimeoutError())).execute({"query": "x"})
        assert not result.success and "timed out" in result.error


class TestWebExtract:
    async def test_returns_page_text(self):
        page = WebPage(url="https://ex/final", title="T", text="Body text", truncated=True)
        result = await WebExtractHandler(FakeReader(page)).execute({"url": "https://ex/a"})
        assert result.success
        assert "Body text" in result.output
        assert "cut off" in result.output
        assert "not instructions" in result.output

    async def test_read_error_message_passed_to_model(self):
        reader = FakeReader(error=WebReadError("Site refused (HTTP 403). Try another result."))
        result = await WebExtractHandler(reader).execute({"url": "https://ex/a"})
        assert not result.success and "Try another result" in result.error

    async def test_unexpected_error_is_generic(self):
        result = await WebExtractHandler(FakeReader(error=RuntimeError("boom"))).execute({"url": "https://x"})
        assert not result.success and "Try another result" in result.error

    async def test_empty_url(self):
        assert not (await WebExtractHandler(FakeReader()).execute({})).success


class TestRegistry:
    def test_web_tools_registered_only_when_provided(self):
        names = lambda reg: [d["function"]["name"] for d in reg.get_definitions()]  # noqa: E731
        assert "web_search" not in names(create_tool_registry())
        reg = create_tool_registry(web_searcher=FakeSearcher(), page_reader=FakeReader())
        assert {"web_search", "web_extract"} <= set(names(reg))
