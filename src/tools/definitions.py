"""Tool definitions for LLM API calls.

This module contains the JSON schema definitions for all tools,
as well as a factory function for creating a pre-configured ToolRegistry.
"""

from typing import Any, Callable, Awaitable, Optional

from .base import ToolHandler
from .registry import ToolRegistry
from .handlers.search import WebSearchHandler, FetchUrlHandler
from .handlers.image import GenerateImageHandler, EditImageHandler
from .handlers.discord import TagUserHandler, PostMessageHandler, PostTableHandler
from .handlers.research import DeepResearchHandler


# Tool definitions in LLM API format
# These match the expected format for OpenAI/Groq function calling
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. USE THIS for: game wikis, recipes, crafting guides, item info, build guides, current events, news, weather, prices, sports, or ANY factual question you're not 100% certain about. If it is complex or multi-step use the deep_research tool instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read the content of a specific URL. Use when you need to read a webpage the user mentioned or to get detailed information from a search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a NEW image from a text description. Use when user asks you to create, generate, draw, make, or wants an image/picture/photo/infographic. For editing or changing or adding details to an existing image, use edit_image instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Edit or modify the most recently generated image. Use when user says 'edit that', 'change the image', 'modify it', 'update that picture', 'give it X', 'make it Y', or references a previous image they want changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change about the image (e.g., 'add silly hats to everyone', 'make it more colorful', 'change the background to a beach')"
                    }
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tag_user",
            "description": "Tag/mention a Discord user in chat with a message. Use when user says 'tag', 'message', 'tell', 'ping', or 'let X know'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Name of the person to tag (e.g., 'Jordan', 'Pat', 'Sam')"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send to the tagged user"
                    }
                },
                "required": ["target_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "post_to_chat",
            "description": "Post a plain text message to the Discord text channel. Use for simple messages, quotes, or single items. For comparisons, rankings, weather across locations, or data with multiple items - use post_table instead (it's much more readable). IMPORTANT: Do NOT include emotion tags like [happy], [laughing], [sigh] etc. in the message - those are for voice only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message content to post. Use Discord markdown (bold, lists, etc.) but NO emotion tags like [happy] or [laughing]."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "post_table",
            "description": "Post a formatted table to the Discord text channel. ALWAYS USE THIS instead of post_to_chat when presenting: comparisons between things, rankings, lists or ingredients, specs/stats, prices, or ANY data with 2+ items that share attributes. Tables are much easier to read than plain text lists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Optional title shown above the table"
                    },
                    "headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column headers"
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "Table rows (each row is array of cell values)"
                    }
                },
                "required": ["headers", "rows"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Perform deep research and analysis. ALWAYS USE THIS when user explicitly says 'deep research', 'think hard', 'analyze this', or asks for in-depth analysis - these phrases override all other rules. Also use for: complex reasoning, code questions, comparing multiple options, or synthesizing 3+ sources. DO NOT USE for simple factual questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to research or analyze. Include the user's question and focus areas."
                    },
                    "context": {
                        "type": "string",
                        "description": "Gathered information (search results, prior tool outputs) to consider."
                    }
                },
                "required": ["task"]
            }
        }
    }
]


# Type alias for Discord async callbacks
DiscordCallback = Callable[..., Awaitable[str]]
ImageCallback = Callable[[bytes, str, str], Awaitable[str]]


def create_tool_registry(
    search_service: Optional[Any] = None,
    image_service: Optional[Any] = None,
    claude_service: Optional[Any] = None,
    on_image_generated: Optional[ImageCallback] = None,
    on_user_tagged: Optional[DiscordCallback] = None,
    on_message_posted: Optional[DiscordCallback] = None,
    max_searches: int = 3,
) -> ToolRegistry:
    """Create a ToolRegistry with all handlers registered.

    Args:
        search_service: SearchService for web search/fetch
        image_service: NanoBananaClient for image generation
        claude_service: ClaudeClient for deep research delegation
        on_image_generated: Callback for posting generated images
        on_user_tagged: Callback for tagging users
        on_message_posted: Callback for posting messages
        max_searches: Maximum searches per request (default 3)

    Returns:
        Configured ToolRegistry with all handlers
    """
    registry = ToolRegistry()

    # Search handlers
    if search_service:
        web_search = WebSearchHandler(search_service, max_searches=max_searches)
        fetch_url = FetchUrlHandler(search_service, web_search_handler=web_search)
        registry.register(web_search)
        registry.register(fetch_url)

    # Image handlers
    generate_image = GenerateImageHandler(image_service, on_image_generated)
    edit_image = EditImageHandler(image_service, on_image_generated)
    registry.register(generate_image)
    registry.register(edit_image)

    # Discord handlers
    tag_user = TagUserHandler(on_user_tagged)
    post_message = PostMessageHandler(on_message_posted)
    post_table = PostTableHandler(on_message_posted)
    registry.register(tag_user)
    registry.register(post_message)
    registry.register(post_table)

    # Research handler
    deep_research = DeepResearchHandler(claude_service)
    registry.register(deep_research)

    return registry


def get_tool_names() -> list[str]:
    """Get list of all tool names.

    Returns:
        List of tool names
    """
    return [defn["function"]["name"] for defn in TOOL_DEFINITIONS]


def get_tool_definition(name: str) -> Optional[dict[str, Any]]:
    """Get tool definition by name.

    Args:
        name: Tool name to look up

    Returns:
        Tool definition dict, or None if not found
    """
    for defn in TOOL_DEFINITIONS:
        if defn["function"]["name"] == name:
            return defn
    return None
