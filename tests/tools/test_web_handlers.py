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
    def __init__(self, page=None, error=None, per_url=None, delay=0.0):
        self.page = page
        self.error = error
        self.per_url = per_url or {}  # url -> WebPage or Exception
        self.delay = delay
        self.urls = []

    async def read(self, url):
        import asyncio

        self.urls.append(url)
        await asyncio.sleep(self.delay)
        outcome = self.per_url.get(url, self.error or self.page)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def page(url, text="Body text", truncated=False):
    return WebPage(url=url, title=f"T {url}", text=text, truncated=truncated)


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
        one = WebPage(url="https://ex/final", title="T", text="Body text", truncated=True)
        result = await WebExtractHandler(FakeReader(one)).execute({"url": "https://ex/a"})
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
        assert not (await WebExtractHandler(FakeReader()).execute({"urls": ["  "]})).success

    async def test_reads_several_urls_in_parallel(self):
        import time

        reader = FakeReader(per_url={"https://a": page("https://a", "Alpha"),
                                     "https://b": page("https://b", "Beta")}, delay=0.3)
        start = time.perf_counter()
        result = await WebExtractHandler(reader).execute({"urls": ["https://a", "https://b"]})
        assert time.perf_counter() - start < 0.5  # concurrent, not 0.6s sequential
        assert result.success
        assert result.output.index("Alpha") < result.output.index("Beta")  # request order kept
        assert result.metadata["read"] == 2

    async def test_partial_failure_still_succeeds(self):
        reader = FakeReader(per_url={
            "https://blocked": WebReadError("Site refused the request (HTTP 403). Try another result."),
            "https://ok": page("https://ok", "Seventy two degrees"),
        })
        result = await WebExtractHandler(reader).execute({"urls": ["https://blocked", "https://ok"]})
        assert result.success
        assert "Seventy two degrees" in result.output
        assert "HTTP 403" in result.output

    async def test_all_failed_is_error_with_reasons(self):
        reader = FakeReader(error=WebReadError("Page took too long to load. Try another result."))
        result = await WebExtractHandler(reader).execute({"urls": ["https://a", "https://b"]})
        assert not result.success
        assert result.error.count("too long") == 2

    async def test_dedupes_and_caps_urls(self):
        reader = FakeReader(page=page("https://x"))
        urls = ["https://a", "https://a"] + [f"https://u{i}" for i in range(10)]
        await WebExtractHandler(reader).execute({"urls": urls})
        assert len(reader.urls) == 5 and reader.urls.count("https://a") == 1

    async def test_schema_takes_url_list(self):
        params = WebExtractHandler(FakeReader()).definition.parameters
        assert params["required"] == ["urls"]
        assert params["properties"]["urls"]["maxItems"] == 5


class TestRegistry:
    def test_web_tools_registered_only_when_provided(self):
        names = lambda reg: [d["function"]["name"] for d in reg.get_definitions()]  # noqa: E731
        assert "web_search" not in names(create_tool_registry())
        reg = create_tool_registry(web_searcher=FakeSearcher(), page_reader=FakeReader())
        assert {"web_search", "web_extract"} <= set(names(reg))
