"""Firecrawl client for self-hosted content extraction."""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


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
                            return data.get("data", {}).get("markdown", "")
                        else:
                            logger.warning(f"Firecrawl returned success=false: {data}")
                            return ""

                    logger.warning(f"Firecrawl returned status {response.status}")
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
