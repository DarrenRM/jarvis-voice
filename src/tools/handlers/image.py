"""Image tool handlers for generation and editing."""

import logging
from typing import Any, Callable, Awaitable, Optional, Protocol

from ..base import ToolDefinition, ToolResult, ToolHandler

logger = logging.getLogger(__name__)


class ImageService(Protocol):
    """Protocol for image generation service."""

    def has_chat(self) -> bool:
        """Check if there's an active chat session (image was generated)."""
        ...

    async def generate(self, prompt: str) -> tuple[bytes, str]:
        """Generate image from prompt. Returns (bytes, mime_type)."""
        ...

    async def edit(self, instruction: str) -> tuple[bytes, str]:
        """Edit previous image. Returns (bytes, mime_type)."""
        ...


# Type alias for image callback: (bytes, prompt, mime_type) -> result string
ImageCallback = Callable[[bytes, str, str], Awaitable[str]]


class GenerateImageHandler(ToolHandler):
    """Handler for generating images from text prompts."""

    def __init__(
        self,
        image_service: Optional[ImageService] = None,
        on_image_generated: Optional[ImageCallback] = None,
    ):
        """Initialize generate image handler.

        Args:
            image_service: Image generation service (e.g., NanoBananaClient)
            on_image_generated: Async callback(bytes, prompt, mime_type) -> str
        """
        self._image_service = image_service
        self._on_image_generated = on_image_generated

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_image",
            description="Generate an image from a text description. Use when the user asks for an image, picture, illustration, or visual content. The image will be posted to the Discord text channel.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate. Be specific about style, composition, colors, and subjects.",
                    },
                },
                "required": ["prompt"],
            },
        )

    def set_service(self, service: ImageService) -> None:
        """Set the image service."""
        self._image_service = service

    def set_callback(self, callback: ImageCallback) -> None:
        """Set the callback for posting generated images."""
        self._on_image_generated = callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute image generation.

        Args:
            args: Must contain "prompt" with image description

        Returns:
            ToolResult with generation result or error
        """
        if not self._image_service:
            return ToolResult(
                success=False,
                output="",
                error="Image generation not available - NANOBANANA_API_KEY not configured",
            )

        prompt = args.get("prompt", "").strip()
        if not prompt:
            return ToolResult(
                success=False,
                output="",
                error="Image prompt cannot be empty",
            )

        logger.info(f"Generating image: {prompt[:100]}...")

        try:
            image_bytes, mime_type = await self._image_service.generate(prompt)
            logger.info(f"Image generated: {len(image_bytes)} bytes, type: {mime_type}")

            if self._on_image_generated:
                logger.info("Calling on_image_generated callback...")
                result = await self._on_image_generated(image_bytes, prompt, mime_type)
                logger.info(f"Callback returned: {result}")

                return ToolResult(
                    success=True,
                    # Inform model that editing is now available
                    output=f"{result} You can now edit this image using the edit_image tool if the user wants changes.",
                    metadata={
                        "prompt": prompt,
                        "size_bytes": len(image_bytes),
                        "mime_type": mime_type,
                    },
                )

            return ToolResult(
                success=False,
                output="",
                error="Image generated but no channel to post to",
                metadata={
                    "prompt": prompt,
                    "size_bytes": len(image_bytes),
                    "mime_type": mime_type,
                },
            )

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Image generation failed: {str(e)}",
                metadata={"prompt": prompt},
            )


class EditImageHandler(ToolHandler):
    """Handler for editing previously generated images."""

    def __init__(
        self,
        image_service: Optional[ImageService] = None,
        on_image_generated: Optional[ImageCallback] = None,
    ):
        """Initialize edit image handler.

        Args:
            image_service: Image generation service (e.g., NanoBananaClient)
            on_image_generated: Async callback(bytes, prompt, mime_type) -> str
        """
        self._image_service = image_service
        self._on_image_generated = on_image_generated

    @property
    def name(self) -> str:
        return "edit_image"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_image",
            description="Edit the previously generated image. Use when the user wants to modify, change, or add something to an image you just created.",
            parameters={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change in the image (e.g., 'add silly hats', 'make the background blue', 'remove the tree')",
                    },
                },
                "required": ["instruction"],
            },
        )

    def set_service(self, service: ImageService) -> None:
        """Set the image service."""
        self._image_service = service

    def set_callback(self, callback: ImageCallback) -> None:
        """Set the callback for posting edited images."""
        self._on_image_generated = callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute image editing.

        Args:
            args: Must contain "instruction" with edit description

        Returns:
            ToolResult with edit result or error
        """
        if not self._image_service:
            return ToolResult(
                success=False,
                output="",
                error="Image editing not available - NANOBANANA_API_KEY not configured",
            )

        # CRITICAL: Check if there's a previous image to edit
        if not self._image_service.has_chat():
            return ToolResult(
                success=False,
                output="",
                error="No previous image to edit. Ask the user what image they'd like you to generate first.",
            )

        instruction = args.get("instruction", "").strip()
        if not instruction:
            return ToolResult(
                success=False,
                output="",
                error="Edit instruction cannot be empty",
            )

        logger.info(f"Editing image: {instruction[:100]}...")

        try:
            image_bytes, mime_type = await self._image_service.edit(instruction)
            logger.info(f"Image edited: {len(image_bytes)} bytes, type: {mime_type}")

            if self._on_image_generated:
                logger.info("Calling on_image_generated callback for edited image...")
                result = await self._on_image_generated(
                    image_bytes, f"Edited: {instruction}", mime_type
                )
                logger.info(f"Callback returned: {result}")

                return ToolResult(
                    success=True,
                    output=f"{result} The image has been edited. You can make more edits using edit_image.",
                    metadata={
                        "instruction": instruction,
                        "size_bytes": len(image_bytes),
                        "mime_type": mime_type,
                    },
                )

            return ToolResult(
                success=False,
                output="",
                error="Image edited but no channel to post to",
                metadata={
                    "instruction": instruction,
                    "size_bytes": len(image_bytes),
                    "mime_type": mime_type,
                },
            )

        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Image editing failed: {str(e)}",
                metadata={"instruction": instruction},
            )
