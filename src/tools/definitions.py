"""Factory for a pre-configured ToolRegistry.

Each handler owns its own schema (``handler.definition``); that is what the
model receives.
"""

from typing import Any, Awaitable, Callable, Optional

from .handlers.discord import PostMessageHandler, PostTableHandler, TagUserHandler
from .handlers.image import EditImageHandler, GenerateImageHandler
from .handlers.web import WebExtractHandler, WebSearchHandler
from .registry import ToolRegistry

# Type alias for Discord async callbacks
DiscordCallback = Callable[..., Awaitable[str]]
ImageCallback = Callable[[bytes, str, str], Awaitable[str]]


def create_tool_registry(
    image_service: Optional[Any] = None,
    on_image_generated: Optional[ImageCallback] = None,
    on_user_tagged: Optional[DiscordCallback] = None,
    on_message_posted: Optional[DiscordCallback] = None,
    web_searcher: Optional[Any] = None,
    page_reader: Optional[Any] = None,
) -> ToolRegistry:
    """Create a ToolRegistry with all available handlers registered.

    Args:
        web_searcher: WebSearcher; registers web_search when provided
        page_reader: PageReader; registers web_extract when provided
        image_service: NanoBananaClient for image generation. Image tools are
            only registered when this is provided.
        on_image_generated: Callback for posting generated images
        on_user_tagged: Callback for tagging users
        on_message_posted: Callback for posting messages

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    if web_searcher:
        registry.register(WebSearchHandler(web_searcher))
    if page_reader:
        registry.register(WebExtractHandler(page_reader))

    if image_service:
        registry.register(GenerateImageHandler(image_service, on_image_generated))
        registry.register(EditImageHandler(image_service, on_image_generated))

    registry.register(TagUserHandler(on_user_tagged))
    registry.register(PostMessageHandler(on_message_posted))
    registry.register(PostTableHandler(on_message_posted))

    return registry
