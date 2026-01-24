"""Factory for creating LLM clients based on configuration."""

import logging
from typing import TYPE_CHECKING

from .base import BaseLLMClient
from .claude_client import ClaudeClient
from .groq_client import GroqClient
from ..search.search_service import SearchService

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)

# Map provider names to Groq model IDs
GROQ_MODEL_MAP = {
    "llama": "llama-3.3-70b-versatile",
    "gpt-oss": "openai/gpt-oss-120b",
    "groq": "llama-3.3-70b-versatile",  # backwards compat
}


def create_llm_client(settings: "Settings") -> BaseLLMClient:
    """Factory function to create the appropriate LLM client.

    Creates either a Claude client or a Groq client based on LLM_PROVIDER:
    - claude: Claude via Claude Code SDK
    - llama: Llama 3.3 70B via Groq
    - gpt-oss: GPT-OSS 120B via Groq
    - groq: (deprecated) alias for llama

    Args:
        settings: Application settings

    Returns:
        LLM client instance based on LLM_PROVIDER setting

    Raises:
        ValueError: If provider is unknown or required API key is missing
    """
    provider = settings.llm_provider.lower()

    # Groq-based providers (llama, gpt-oss, groq)
    if provider in GROQ_MODEL_MAP:
        if not settings.groq_api_key:
            raise ValueError(
                f"GROQ_API_KEY required for {provider} provider. "
                "Set it in your .env file."
            )

        model = GROQ_MODEL_MAP[provider]

        # Create search service for tool use
        search_service = SearchService(
            firecrawl_url=settings.firecrawl_url,
            firecrawl_api_key=settings.firecrawl_api_key,
            max_results=settings.search_max_results,
        )

        logger.info(f"Creating Groq client (provider: {provider}, model: {model})")
        return GroqClient(
            api_key=settings.groq_api_key,
            model=model,
            max_tokens=settings.groq_max_tokens,
            search_service=search_service,
        )

    elif provider == "claude":
        logger.info(f"Creating Claude client (model: {settings.claude_model})")
        return ClaudeClient(model=settings.claude_model)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Valid options are 'claude', 'llama', or 'gpt-oss'."
        )
