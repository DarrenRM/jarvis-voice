"""Factory for creating the LLM service from configuration."""

import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from ..tools import ToolExecutor, create_tool_registry
from .cerebras_client import CerebrasClient
from .service import LLMService
from .system_prompt_builder import create_default_prompt_builder

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


def create_llm_client(
    settings: "Settings",
    on_image_generated: Optional[Callable[[bytes, str, str], Awaitable[str]]] = None,
    on_user_tagged: Optional[Callable[[str, str], Awaitable[str]]] = None,
    on_message_posted: Optional[Callable[[str], Awaitable[str]]] = None,
) -> LLMService:
    """Create the LLM service (Cerebras) with its tools.

    Args:
        settings: Application settings
        on_image_generated: Callback for posting generated images
        on_user_tagged: Callback for tagging users in Discord
        on_message_posted: Callback for posting messages to Discord

    Raises:
        ValueError: If CEREBRAS_API_KEY is missing
    """
    if not settings.cerebras_api_key:
        raise ValueError("CEREBRAS_API_KEY is required. Set it in your .env file.")

    image_service = None
    if settings.image_generation_enabled:
        if settings.nanobanana_api_key:
            from ..image import NanoBananaClient

            logger.info("Image generation enabled (Nano Banana)")
            image_service = NanoBananaClient(api_key=settings.nanobanana_api_key)
        else:
            logger.warning("IMAGE_GENERATION_ENABLED but NANOBANANA_API_KEY not set, disabling")

    web_searcher = page_reader = None
    if settings.web_tools_enabled:
        from ..web import PageReader, WebSearcher

        web_searcher = WebSearcher(max_results=settings.web_search_max_results)
        page_reader = PageReader(max_chars=settings.web_extract_max_chars)

    tool_registry = create_tool_registry(
        image_service=image_service,
        on_image_generated=on_image_generated,
        on_user_tagged=on_user_tagged,
        on_message_posted=on_message_posted,
        web_searcher=web_searcher,
        page_reader=page_reader,
    )

    client = CerebrasClient(
        api_key=settings.cerebras_api_key,
        model=settings.cerebras_model,
        reasoning_effort=settings.cerebras_reasoning_effort,
        max_completion_tokens=settings.llm_max_completion_tokens,
        timeout=settings.llm_timeout_seconds,
    )

    logger.info(
        f"Creating Cerebras client (model: {settings.cerebras_model}, "
        f"reasoning: {settings.cerebras_reasoning_effort}, "
        f"tools: {[d['function']['name'] for d in tool_registry.get_definitions()]})"
    )
    return LLMService(
        client=client,
        tool_executor=ToolExecutor(tool_registry),
        prompt_builder=create_default_prompt_builder(web_tools=settings.web_tools_enabled),
        image_service=image_service,
    )
