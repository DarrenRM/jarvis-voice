"""Firecrawl client for self-hosted content extraction."""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Browser-like headers to avoid being blocked by websites
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


class FirecrawlClient:
    """Client for self-hosted Firecrawl instance.

    Firecrawl extracts content from URLs and converts to markdown,
    which is ideal for feeding to LLMs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3002",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize Firecrawl client.

        Args:
            base_url: Base URL of self-hosted Firecrawl instance
            api_key: Optional API key (not needed for self-hosted without auth)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def scrape(self, url: str) -> str:
        """Scrape URL and return markdown content.

        Args:
            url: URL to scrape

        Returns:
            Markdown content from the page, or empty string on error
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "url": url,
            "formats": ["markdown"],
            # Pass browser-like headers to the target site
            "headers": BROWSER_HEADERS,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/scrape",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Firecrawl returns {success: true, data: {markdown: "..."}}
                        if data.get("success"):
                            content = data.get("data", {}).get("markdown", "")
                            logger.info(f"Firecrawl scraped {url}: {len(content)} chars")
                            return content
                        else:
                            error_msg = data.get("error", "unknown error")
                            logger.warning(f"Firecrawl failed for {url}: {error_msg}")
                            return ""

                    logger.warning(f"Firecrawl returned status {response.status} for {url}")
                    return ""

        except aiohttp.ClientConnectorError:
            logger.warning(f"Firecrawl not available at {self.base_url}")
            return ""
        except asyncio.TimeoutError:
            logger.warning(f"Firecrawl timeout for {url}")
            return ""
        except Exception as e:
            logger.error(f"Firecrawl error for {url}: {e}")
            return ""

    async def is_available(self) -> bool:
        """Check if Firecrawl service is available.

        Returns:
            True if service responds, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status in (200, 404)  # 404 is OK, means server is up
        except Exception:
            return False
