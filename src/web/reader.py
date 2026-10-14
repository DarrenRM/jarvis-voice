"""Read a web page as plain text: httpx fetch + local Trafilatura extraction.

Guards:
- http/https only; every redirect hop is resolved and rejected if it points at a
  private, loopback, link-local or otherwise non-public address
- at most MAX_REDIRECTS manual redirects
- per-request timeout plus a total time budget
- byte cap on the download and a MIME allowlist

DNS is checked before each hop, not pinned for the connection, so a hostile
DNS server could still rebind between check and connect. Fine for a home bot.
"""

import asyncio
import ipaddress
import logging
import socket
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 6
REQUEST_TIMEOUT = 15.0
TOTAL_BUDGET = 30.0
ALLOWED_MIME = {"text/html", "application/xhtml+xml", "text/plain", "text/markdown"}
HTML_MIME = {"text/html", "application/xhtml+xml"}
USER_AGENT = "Mozilla/5.0 (compatible; JarvisVoiceBot/1.0; page reader)"

Resolver = Callable[[str, int], Awaitable[list[str]]]


class WebReadError(Exception):
    """Page could not be read. The message is written for the model to act on."""


@dataclass
class WebPage:
    url: str  # final URL after redirects
    title: str
    text: str
    truncated: bool


async def _system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class PageReader:
    """Fetches one URL and returns its main text."""

    def __init__(
        self,
        max_chars: int = 15000,
        resolver: Optional[Resolver] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """
        Args:
            max_chars: Extracted text longer than this is cut
            resolver: DNS lookup override (tests)
            transport: httpx transport override (tests)
        """
        self._max_chars = max_chars
        self._resolve = resolver or _system_resolver
        self._transport = transport

    async def read(self, url: str) -> WebPage:
        """Fetch and extract a page.

        Raises:
            WebReadError: with a message telling the model what to try next
        """
        try:
            async with asyncio.timeout(TOTAL_BUDGET):
                final_url, mime, source = await self._fetch(url)
        except TimeoutError:
            raise WebReadError("Page took too long to load. Try another result.") from None
        except httpx.HTTPStatusError as e:
            raise WebReadError(
                f"Site refused the request (HTTP {e.response.status_code}). Try another result."
            ) from None
        except httpx.HTTPError as e:
            raise WebReadError(
                f"Could not connect ({type(e).__name__}). Try another result."
            ) from None

        if mime in HTML_MIME:
            title, text = await asyncio.to_thread(_extract_html, source, final_url)
        else:
            title, text = final_url, source
        text = text.strip()
        if not text:
            raise WebReadError(
                "No readable text on that page (it probably needs JavaScript). "
                "Try another result or answer from the search snippets."
            )

        truncated = len(text) > self._max_chars
        if truncated:
            text = text[: self._max_chars].rsplit(" ", 1)[0]
        return WebPage(url=final_url, title=title or final_url, text=text, truncated=truncated)

    async def _fetch(self, url: str) -> tuple[str, str, str]:
        started = time.monotonic()
        target = url
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT},
            transport=self._transport,
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await self._check_url(target)
                async with client.stream("GET", target) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        target = str(response.url.join(location))
                        continue
                    response.raise_for_status()
                    mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if mime not in ALLOWED_MIME:
                        raise WebReadError(
                            f"Unsupported document type ({mime or 'unknown'}). Try another result."
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes(16384):
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            raise WebReadError("Page is too large to read. Try another result.")
                    source = bytes(body).decode(response.encoding or "utf-8", errors="replace")
                    logger.info(
                        f"[TIMING] web_extract fetch: {time.monotonic() - started:.2f}s "
                        f"{len(body)} bytes {target}"
                    )
                    return str(response.url), mime, source
        raise WebReadError("Too many redirects. Try another result.")

    async def _check_url(self, url: str) -> None:
        try:
            parsed = httpx.URL(url)
        except httpx.InvalidURL:
            raise WebReadError("That is not a valid link.") from None
        if parsed.scheme not in ("http", "https") or not parsed.host:
            raise WebReadError("Only http and https links can be read.")

        host = parsed.host
        try:
            addresses = [str(ipaddress.ip_address(host))]
        except ValueError:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                addresses = await self._resolve(host, port)
            except OSError:
                raise WebReadError(f"Could not find the site {host}. Try another result.") from None

        if not addresses or not all(_is_public(a) for a in addresses):
            logger.warning(f"Blocked non-public address for {host}: {addresses}")
            raise WebReadError("That link points to a private or local network address; blocked.")


def _extract_html(source: str, url: str) -> tuple[str, str]:
    import trafilatura
    from lxml import html

    title = ""
    try:
        title = " ".join(html.fromstring(source).xpath("//title/text()")).strip()
    except Exception:
        pass
    text = trafilatura.extract(
        source,
        url=url,
        output_format="markdown",
        include_links=False,  # URLs are noise for a voice answer
        include_tables=True,
        include_comments=False,
    ) or ""
    return title, text
