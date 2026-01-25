"""Firecrawl client for self-hosted content extraction."""

import asyncio
import json
import logging
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Generic extraction schema - returns structured facts for any content type
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key facts as brief data points (e.g. 'Craft time: 20s', 'Temperature: 72F')"
        },
        "content_type": {
            "type": "string",
            "description": "Type of content: recipe, article, wiki, weather, news, etc"
        }
    }
}

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

    async def extract(self, url: str, query: str) -> Optional[dict[str, Any]]:
        """Extract structured data from URL using LLM.

        Args:
            url: URL to extract from
            query: What information to extract (e.g., "crafting recipe", "weather forecast")

        Returns:
            Extracted data dict with 'facts' and 'content_type', or None on error
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "url": url,
            "formats": ["extract"],
            "extract": {
                "prompt": f"Extract relevant information about: {query}",
                "schema": EXTRACT_SCHEMA,
            },
            "headers": BROWSER_HEADERS,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/scrape",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout + 30),  # Extra time for LLM
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success"):
                            extract = data.get("data", {}).get("extract")
                            if extract and extract.get("facts"):
                                logger.info(f"Firecrawl extracted {len(extract['facts'])} facts from {url}")
                                return extract
                            logger.warning(f"Firecrawl extraction returned no facts for {url}")
                            return None
                        else:
                            error_msg = data.get("error", "unknown error")
                            logger.warning(f"Firecrawl extract failed for {url}: {error_msg}")
                            return None

                    logger.warning(f"Firecrawl extract returned status {response.status} for {url}")
                    return None

        except aiohttp.ClientConnectorError:
            logger.warning(f"Firecrawl not available at {self.base_url}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Firecrawl extract timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Firecrawl extract error for {url}: {e}")
            return None

    async def smart_extract(
        self, url: str, query: str, max_markdown_length: int = 8000
    ) -> str:
        """Get page content as markdown for LLM processing.

        Args:
            url: URL to extract from
            query: What information to extract (used for logging)
            max_markdown_length: Max chars of markdown to return

        Returns:
            Markdown content truncated to max length, or error message
        """
        # Skip LLM extraction - it returns useless generic facts
        # Just get markdown and let the calling LLM find what it needs
        markdown = await self.scrape(url)
        if markdown:
            if len(markdown) > max_markdown_length:
                markdown = markdown[:max_markdown_length] + "\n\n[Content truncated...]"
            logger.info(f"Returning {len(markdown)} chars markdown for {url}")
            return markdown

        return f"Could not fetch content from {url}"

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
