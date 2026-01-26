"""Tests for search tool handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.handlers.search import WebSearchHandler, FetchUrlHandler


class MockSearchService:
    """Mock search service for testing."""

    def __init__(self, search_result: str = "Search results here", fetch_result: str = "Page content here"):
        self.search_result = search_result
        self.fetch_result = fetch_result
        self.search_and_fetch = AsyncMock(return_value=search_result)
        self.fetch_url = AsyncMock(return_value=fetch_result)


class TestWebSearchHandler:
    """Tests for WebSearchHandler."""

    @pytest.fixture
    def search_service(self):
        return MockSearchService()

    @pytest.fixture
    def handler(self, search_service):
        return WebSearchHandler(search_service, max_searches=3)

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "web_search"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "web_search"
        assert len(defn.description) > 0
        assert "query" in defn.parameters["properties"]
        assert "query" in defn.parameters["required"]

    def test_initial_search_count_is_zero(self, handler):
        """Search count starts at zero."""
        assert handler.search_count == 0
        assert handler.limit_reached is False

    @pytest.mark.asyncio
    async def test_successful_search(self, handler, search_service):
        """Successful search returns results."""
        result = await handler.execute({"query": "python tutorials"})

        assert result.success is True
        assert result.output == "Search results here"
        assert result.error is None
        assert result.metadata["query"] == "python tutorials"
        search_service.search_and_fetch.assert_called_once_with("python tutorials", fetch_top_n=1)

    @pytest.mark.asyncio
    async def test_search_increments_count(self, handler):
        """Each search increments the count."""
        assert handler.search_count == 0

        await handler.execute({"query": "test 1"})
        assert handler.search_count == 1

        await handler.execute({"query": "test 2"})
        assert handler.search_count == 2

    @pytest.mark.asyncio
    async def test_empty_query_fails(self, handler):
        """Empty query returns error."""
        result = await handler.execute({"query": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_query_fails(self, handler):
        """Whitespace-only query returns error."""
        result = await handler.execute({"query": "   "})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_query_fails(self, handler):
        """Missing query key returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_limit_enforced(self, handler):
        """Search limit is enforced after max_searches."""
        # Do 3 searches (the limit)
        for i in range(3):
            result = await handler.execute({"query": f"search {i}"})
            assert result.success is True

        assert handler.search_count == 3
        assert handler.limit_reached is True

        # 4th search should fail
        result = await handler.execute({"query": "one more"})
        assert result.success is False
        assert "limit reached" in result.error.lower()
        assert result.metadata.get("limit_reached") is True

        # Count should NOT increment on rejected search
        assert handler.search_count == 3

    @pytest.mark.asyncio
    async def test_reset_count(self, handler):
        """reset_count() resets the search count."""
        await handler.execute({"query": "test 1"})
        await handler.execute({"query": "test 2"})
        assert handler.search_count == 2

        handler.reset_count()
        assert handler.search_count == 0
        assert handler.limit_reached is False

    @pytest.mark.asyncio
    async def test_reset_allows_more_searches(self, handler):
        """After reset, more searches are allowed."""
        # Hit the limit
        for i in range(3):
            await handler.execute({"query": f"search {i}"})

        # Reset
        handler.reset_count()

        # Can search again
        result = await handler.execute({"query": "new search"})
        assert result.success is True
        assert handler.search_count == 1

    @pytest.mark.asyncio
    async def test_search_error_still_increments_count(self, handler, search_service):
        """Count increments even on error to prevent retry loops."""
        search_service.search_and_fetch.side_effect = Exception("Network error")

        result = await handler.execute({"query": "test"})
        assert result.success is False
        assert "Network error" in result.error
        assert handler.search_count == 1  # Still incremented

    @pytest.mark.asyncio
    async def test_metadata_includes_search_count(self, handler):
        """Result metadata includes current search count."""
        await handler.execute({"query": "test 1"})
        result = await handler.execute({"query": "test 2"})

        assert result.metadata["search_count"] == 2
        assert result.metadata["limit_reached"] is False

    def test_custom_max_searches(self, search_service):
        """Can configure different max searches."""
        handler = WebSearchHandler(search_service, max_searches=5)
        assert handler._max_searches == 5


class TestFetchUrlHandler:
    """Tests for FetchUrlHandler."""

    @pytest.fixture
    def search_service(self):
        return MockSearchService()

    @pytest.fixture
    def web_search_handler(self, search_service):
        return WebSearchHandler(search_service, max_searches=3)

    @pytest.fixture
    def handler(self, search_service, web_search_handler):
        return FetchUrlHandler(search_service, web_search_handler)

    @pytest.fixture
    def standalone_handler(self, search_service):
        """Handler without linked web search handler."""
        return FetchUrlHandler(search_service)

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "fetch_url"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "fetch_url"
        assert len(defn.description) > 0
        assert "url" in defn.parameters["properties"]
        assert "url" in defn.parameters["required"]
        assert "query" in defn.parameters["properties"]

    @pytest.mark.asyncio
    async def test_successful_fetch(self, handler, search_service):
        """Successful fetch returns content."""
        result = await handler.execute({"url": "https://example.com"})

        assert result.success is True
        assert result.output == "Page content here"
        assert result.metadata["url"] == "https://example.com"
        search_service.fetch_url.assert_called_once_with("https://example.com", "")

    @pytest.mark.asyncio
    async def test_fetch_with_query(self, handler, search_service):
        """Fetch can include query for smart extraction."""
        result = await handler.execute({
            "url": "https://example.com",
            "query": "python tutorials"
        })

        assert result.success is True
        search_service.fetch_url.assert_called_once_with("https://example.com", "python tutorials")
        assert result.metadata["query"] == "python tutorials"

    @pytest.mark.asyncio
    async def test_empty_url_fails(self, handler):
        """Empty URL returns error."""
        result = await handler.execute({"url": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_url_fails(self, handler):
        """Missing URL key returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_counts_toward_search_limit(self, handler, web_search_handler):
        """Fetch increments shared search count."""
        assert web_search_handler.search_count == 0

        await handler.execute({"url": "https://example.com"})
        assert web_search_handler.search_count == 1

        await handler.execute({"url": "https://example2.com"})
        assert web_search_handler.search_count == 2

    @pytest.mark.asyncio
    async def test_fetch_respects_shared_limit(self, handler, web_search_handler):
        """Fetch blocked when shared search limit reached."""
        # Use up limit with searches
        for i in range(3):
            await web_search_handler.execute({"query": f"search {i}"})

        assert handler.limit_reached is True

        # Fetch should fail
        result = await handler.execute({"url": "https://example.com"})
        assert result.success is False
        assert "limit reached" in result.error.lower()

    @pytest.mark.asyncio
    async def test_standalone_handler_no_limit(self, standalone_handler):
        """Standalone handler without web search has no limit."""
        assert standalone_handler.limit_reached is False

        # Can fetch many times
        for i in range(10):
            result = await standalone_handler.execute({"url": f"https://example{i}.com"})
            assert result.success is True

    @pytest.mark.asyncio
    async def test_fetch_failure_detected(self, handler, search_service):
        """Detects Firecrawl not running error."""
        search_service.fetch_url.return_value = "Could not fetch content from https://example.com. Firecrawl may not be running."

        result = await handler.execute({"url": "https://example.com"})

        assert result.success is False
        assert "Could not fetch" in result.error

    @pytest.mark.asyncio
    async def test_fetch_exception_handled(self, handler, search_service):
        """Fetch exceptions are handled gracefully."""
        search_service.fetch_url.side_effect = Exception("Connection refused")

        result = await handler.execute({"url": "https://example.com"})

        assert result.success is False
        assert "Connection refused" in result.error

    def test_reset_count_is_noop(self, handler, web_search_handler):
        """reset_count on fetch handler is a no-op (managed by web search)."""
        web_search_handler._search_count = 5
        handler.reset_count()
        assert web_search_handler._search_count == 5  # Unchanged


class TestSearchHandlerIntegration:
    """Integration tests for search handlers working together."""

    @pytest.fixture
    def search_service(self):
        return MockSearchService()

    @pytest.fixture
    def handlers(self, search_service):
        web_search = WebSearchHandler(search_service, max_searches=3)
        fetch_url = FetchUrlHandler(search_service, web_search)
        return web_search, fetch_url

    @pytest.mark.asyncio
    async def test_mixed_search_and_fetch_share_limit(self, handlers):
        """Searches and fetches share the same limit."""
        web_search, fetch_url = handlers

        # 1 search
        await web_search.execute({"query": "test"})
        assert web_search.search_count == 1

        # 1 fetch
        await fetch_url.execute({"url": "https://example.com"})
        assert web_search.search_count == 2

        # 1 more search
        await web_search.execute({"query": "test 2"})
        assert web_search.search_count == 3

        # Both should now be at limit
        assert web_search.limit_reached is True
        assert fetch_url.limit_reached is True

        # Neither can do more
        search_result = await web_search.execute({"query": "blocked"})
        fetch_result = await fetch_url.execute({"url": "https://blocked.com"})

        assert search_result.success is False
        assert fetch_result.success is False

    @pytest.mark.asyncio
    async def test_reset_affects_both(self, handlers):
        """Reset on web search allows both to work again."""
        web_search, fetch_url = handlers

        # Hit limit
        for i in range(3):
            await web_search.execute({"query": f"search {i}"})

        assert fetch_url.limit_reached is True

        # Reset via web search
        web_search.reset_count()

        # Both work again
        assert fetch_url.limit_reached is False

        search_result = await web_search.execute({"query": "new"})
        fetch_result = await fetch_url.execute({"url": "https://new.com"})

        assert search_result.success is True
        assert fetch_result.success is True
