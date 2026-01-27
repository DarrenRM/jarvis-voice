"""Factory for creating LLM clients based on configuration."""

import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from .base import BaseLLMClient
from .claude_client import ClaudeClient
from .groq_client import GroqClient
from ..search.search_service import SearchService
from ..image import NanoBananaClient
from ..tools.definitions import create_tool_registry

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)

# Map provider names to Groq model IDs
GROQ_MODEL_MAP = {
    "llama": "llama-3.3-70b-versatile",
    "gpt-oss": "openai/gpt-oss-120b",
    "groq": "llama-3.3-70b-versatile",  # backwards compat
}


def create_llm_client(
    settings: "Settings",
    on_image_generated: Optional[Callable[[bytes, str], Awaitable[str]]] = None,
    on_user_tagged: Optional[Callable[[str, str], Awaitable[str]]] = None,
    on_message_posted: Optional[Callable[[str], Awaitable[str]]] = None,
) -> BaseLLMClient:
    """Factory function to create the appropriate LLM client.

    Creates either a Claude client or a Groq client based on LLM_PROVIDER:
    - claude: Claude via Claude Code SDK
    - llama: Llama 3.3 70B via Groq
    - gpt-oss: GPT-OSS 120B via Groq
    - groq: (deprecated) alias for llama

    Args:
        settings: Application settings
        on_image_generated: Callback for posting generated images (Groq only)
        on_user_tagged: Callback for tagging users in Discord (Groq only)
        on_message_posted: Callback for posting messages to Discord (Groq only)

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
            max_content_length=settings.search_max_content_length,
        )

        # Create image service if API key is configured
        image_service = None
        if settings.nanobanana_api_key:
            logger.info("Image generation enabled (Nano Banana)")
            image_service = NanoBananaClient(api_key=settings.nanobanana_api_key)

        # Create Claude service for delegation if enabled
        claude_service = None
        if settings.claude_delegation_enabled:
            try:
                claude_service = ClaudeClient(model=settings.claude_model, max_tokens=1024)
                logger.info(f"Claude delegation enabled (using {settings.claude_model} for deep analysis)")
            except Exception as e:
                logger.warning(f"Claude delegation unavailable: {e}")

        # Create tool registry for new modular tool system
        tool_registry = create_tool_registry(
            search_service=search_service,
            image_service=image_service,
            claude_service=claude_service,
            on_image_generated=on_image_generated,
            on_user_tagged=on_user_tagged,
            on_message_posted=on_message_posted,
            claude_timeout=settings.claude_timeout,
        )

        logger.info(f"Creating Groq client (provider: {provider}, model: {model})")
        logger.info(f"Image callback provided: {on_image_generated is not None}")
        logger.info(f"Tag callback provided: {on_user_tagged is not None}")
        return GroqClient(
            api_key=settings.groq_api_key,
            model=model,
            max_tokens=settings.groq_max_tokens,
            search_service=search_service,
            image_service=image_service,
            on_image_generated=on_image_generated,
            on_user_tagged=on_user_tagged,
            on_message_posted=on_message_posted,
            claude_service=claude_service,
            tool_registry=tool_registry,
        )

    elif provider == "claude":
        logger.info(f"Creating Claude client (model: {settings.claude_model})")
        return ClaudeClient(model=settings.claude_model)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Valid options are 'claude', 'llama', or 'gpt-oss'."
        )
