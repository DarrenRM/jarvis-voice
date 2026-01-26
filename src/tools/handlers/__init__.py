"""Tool handlers for LLM function calling."""

from .search import WebSearchHandler, FetchUrlHandler
from .discord import TagUserHandler, PostMessageHandler, PostTableHandler
from .image import GenerateImageHandler, EditImageHandler
from .research import DeepResearchHandler

__all__ = [
    # Search handlers
    "WebSearchHandler",
    "FetchUrlHandler",
    # Discord handlers
    "TagUserHandler",
    "PostMessageHandler",
    "PostTableHandler",
    # Image handlers
    "GenerateImageHandler",
    "EditImageHandler",
    # Research handler
    "DeepResearchHandler",
]
