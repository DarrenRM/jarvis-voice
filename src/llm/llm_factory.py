"""Factory for creating LLM clients based on configuration."""

import logging
from typing import TYPE_CHECKING

from .base import BaseLLMClient
from .claude_client import ClaudeClient
from .groq_llama_client import GroqLlamaClient
from ..search.search_service import SearchService

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


def create_llm_client(settings: "Settings") -> BaseLLMClient:
    """Factory function to create the appropriate LLM client.

    Creates either a Claude client (using Claude Code SDK) or a Groq Llama
    client based on the LLM_PROVIDER setting.

    Args:
        settings: Application settings

    Returns:
        LLM client instance based on LLM_PROVIDER setting

    Raises:
        ValueError: If provider is unknown or required API key is missing
    """
    provider = settings.llm_provider.lower()

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY required for Groq LLM provider. "
                "Set it in your .env file."
            )

        # Create search service for tool use
        search_service = SearchService(
            firecrawl_url=settings.firecrawl_url,
            firecrawl_api_key=settings.firecrawl_api_key,
            max_results=settings.search_max_results,
        )

        logger.info(
            f"Creating Groq Llama client (model: {settings.groq_llm_model})"
        )
        return GroqLlamaClient(
            api_key=settings.groq_api_key,
            model=settings.groq_llm_model,
            max_tokens=settings.groq_max_tokens,
            search_service=search_service,
        )

    elif provider == "claude":
        logger.info(f"Creating Claude client (model: {settings.claude_model})")
        return ClaudeClient(model=settings.claude_model)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Valid options are 'claude' or 'groq'."
        )
