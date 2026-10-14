"""Tests for PageReader: network guards and extraction. No real network."""

import httpx
import pytest

from src.web import PageReader, WebReadError
from src.web import reader as reader_module

ARTICLE = """<html><head><title>Heavy Darts - Wiki</title></head><body>
<nav>Home | Items | Crafting</nav>
<article><h1>Heavy Darts</h1>
<p>Heavy darts are crafted at a fabricator using seven salvaged metal and two scrap.
They deal extra damage to armored targets and are a staple of mid game builds.</p>
<p>Players usually unlock the recipe after reaching the second research tier.</p>
</article><footer>Copyright</footer></body></html>"""

PUBLIC = {"example.com": ["93.184.215.14"], "other.example": ["93.184.215.15"]}


def resolver(table):
    async def resolve(host, port):
        if host not in table:
            raise OSError("not found")
        return table[host]
    return resolve


def make_reader(handler, table=None, max_chars=6000):
    return PageReader(
        max_chars=max_chars,
        resolver=resolver(table or PUBLIC),
        transport=httpx.MockTransport(handler),
    )


def html_response(body=ARTICLE, **headers):
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8", **headers},
                          text=body)


class TestExtraction:
    async def test_reads_article_text_and_title(self):
        page = await make_reader(lambda req: html_response()).read("https://example.com/darts")
        assert page.title == "Heavy Darts - Wiki"
        assert "seven salvaged metal" in page.text
        assert "Copyright" not in page.text
        assert page.url == "https://example.com/darts"
        assert not page.truncated

    async def test_plain_text_passed_through(self):
        reader = make_reader(lambda req: httpx.Response(
            200, headers={"content-type": "text/plain"}, text="just text"))
        page = await reader.read("https://example.com/a.txt")
        assert page.text == "just text"

    async def test_truncates_long_pages(self):
        reader = make_reader(lambda req: httpx.Response(
            200, headers={"content-type": "text/plain"}, text="word " * 1000), max_chars=100)
        page = await reader.read("https://example.com/long")
        assert page.truncated and len(page.text) <= 100
        assert not page.text.endswith(" ")

    async def test_empty_page_suggests_next_step(self):
        empty = "<html><body><script></script></body></html>"
        reader = make_reader(lambda req: html_response(empty))
        with pytest.raises(WebReadError, match="JavaScript"):
            await reader.read("https://example.com/spa")


class TestRedirects:
    async def test_follows_public_redirect(self):
        def handler(req):
            if req.url.host == "example.com":
                return httpx.Response(301, headers={"location": "https://other.example/final"})
            return html_response()
        page = await make_reader(handler).read("https://example.com/start")
        assert page.url == "https://other.example/final"

    async def test_relative_redirect(self):
        def handler(req):
            if req.url.path == "/old":
                return httpx.Response(302, headers={"location": "/new"})
            return html_response()
        page = await make_reader(handler).read("https://example.com/old")
        assert page.url == "https://example.com/new"

    async def test_redirect_to_private_address_blocked(self):
        seen = []

        def handler(req):
            seen.append(str(req.url))
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
        with pytest.raises(WebReadError, match="private or local"):
            await make_reader(handler).read("https://example.com/")
        assert seen == ["https://example.com/"]  # metadata endpoint never requested

    async def test_too_many_redirects(self):
        reader = make_reader(lambda req: httpx.Response(302, headers={"location": "/loop"}))
        with pytest.raises(WebReadError, match="Too many redirects"):
            await reader.read("https://example.com/loop")


class TestAddressGuard:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/",
        "http://localhost:8080/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
    ])
    async def test_private_targets_never_requested(self, url):
        requested = []
        table = {"localhost": ["127.0.0.1"]}
        reader = make_reader(lambda req: requested.append(req) or html_response(), table)
        with pytest.raises(WebReadError):
            await reader.read(url)
        assert requested == []

    async def test_hostname_resolving_to_private_blocked(self):
        table = {"evil.example": ["93.184.215.14", "10.0.0.1"]}
        reader = make_reader(lambda req: html_response(), table)
        with pytest.raises(WebReadError, match="private or local"):
            await reader.read("https://evil.example/")

    @pytest.mark.parametrize(
        "url", ["file:///C:/Windows/win.ini", "ftp://example.com/x", "example.com"]
    )
    async def test_non_http_schemes_rejected(self, url):
        with pytest.raises(WebReadError):
            await make_reader(lambda req: html_response()).read(url)

    async def test_unknown_host(self):
        with pytest.raises(WebReadError, match="Could not find"):
            await make_reader(lambda req: html_response()).read("https://nope.invalid/")


class TestLimits:
    async def test_mime_allowlist(self):
        reader = make_reader(lambda req: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
        with pytest.raises(WebReadError, match="application/pdf"):
            await reader.read("https://example.com/doc.pdf")

    async def test_byte_cap(self, monkeypatch):
        monkeypatch.setattr(reader_module, "MAX_BYTES", 1000)
        reader = make_reader(lambda req: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"x" * 5000))
        with pytest.raises(WebReadError, match="too large"):
            await reader.read("https://example.com/big")

    async def test_http_error_status(self):
        reader = make_reader(lambda req: httpx.Response(403, text="nope"))
        with pytest.raises(WebReadError, match="HTTP 403"):
            await reader.read("https://example.com/")

    async def test_total_budget(self, monkeypatch):
        import asyncio

        monkeypatch.setattr(reader_module, "TOTAL_BUDGET", 0.05)

        async def slow(host, port):
            await asyncio.sleep(1)
            return ["93.184.215.14"]
        reader = PageReader(resolver=slow, transport=httpx.MockTransport(lambda r: html_response()))
        with pytest.raises(WebReadError, match="too long"):
            await reader.read("https://example.com/")
