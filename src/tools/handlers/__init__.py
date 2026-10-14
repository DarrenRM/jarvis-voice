"""Tool handlers for LLM function calling."""

from .discord import PostMessageHandler, PostTableHandler, TagUserHandler
from .image import EditImageHandler, GenerateImageHandler
from .web import WebExtractHandler, WebSearchHandler

__all__ = [
    # Discord handlers
    "TagUserHandler",
    "PostMessageHandler",
    "PostTableHandler",
    # Image handlers
    "GenerateImageHandler",
    "EditImageHandler",
    # Web handlers
    "WebSearchHandler",
    "WebExtractHandler",
]
