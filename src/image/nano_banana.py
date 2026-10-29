"""Nano Banana (Gemini) image generation and editing client."""

import asyncio
import io
import logging

logger = logging.getLogger(__name__)


class NanoBananaClient:
    """Client for Gemini image generation and multi-turn editing.

    Uses Gemini's chat interface for image generation, which enables
    multi-turn editing (e.g., "edit that to add silly hats").
    """

    MODEL = "gemini-2.5-flash-image"  # pinned on purpose: agents must not change it

    def __init__(self, api_key: str):
        """Initialize the Nano Banana client.

        Args:
            api_key: Google AI API key (NANOBANANA_API_KEY)
        """
        self.api_key = api_key
        self._client = None
        self._chat = None  # Persistent chat session for multi-turn editing

    def _ensure_client(self):
        """Lazy initialization of the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def has_chat(self) -> bool:
        """Check if there's an active chat session (image was generated).

        Returns:
            True if an image has been generated and can be edited
        """
        return self._chat is not None

    def reset_chat(self) -> None:
        """Reset the chat context.

        Call this when the voice session ends to clear image editing context.
        """
        self._chat = None
        logger.debug("Image chat context reset")

    def _extract_image(self, response) -> tuple[bytes, str]:
        """Extract image bytes from a Gemini response, converting to PNG.

        Uses Gemini's as_image() helper for proper format handling,
        then converts to PNG bytes for consistent Discord uploads.

        Args:
            response: Gemini chat response

        Returns:
            Tuple of (PNG image bytes, mime_type "image/png")

        Raises:
            ValueError: If no image found in response
        """
        # Check for valid response structure
        if not response.candidates:
            raise ValueError("No candidates in response - request may have been blocked")

        candidate = response.candidates[0]
        if candidate.content is None:
            # Check if there's a block reason
            finish_reason = getattr(candidate, 'finish_reason', None)
            raise ValueError(f"No content in response (finish_reason: {finish_reason}) - likely blocked by safety filters")

        if not candidate.content.parts:
            raise ValueError("No parts in response content")

        for part in candidate.content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                original_mime = getattr(part.inline_data, 'mime_type', 'unknown')
                logger.info(f"Gemini returned image with MIME type: {original_mime}")

                # Use as_image() to get PIL Image, then convert to PNG bytes
                try:
                    pil_image = part.as_image()
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    png_bytes = buffer.getvalue()
                    logger.info(f"Converted to PNG: {len(png_bytes)} bytes")
                    return png_bytes, "image/png"
                except Exception as e:
                    logger.warning(f"as_image() failed ({e}), using raw bytes")
                    return part.inline_data.data, original_mime

        raise ValueError("No image in response")

    async def generate(self, prompt: str) -> tuple[bytes, str]:
        """Generate a NEW image from a text prompt.

        This starts a fresh chat session, discarding any previous image context.
        Use edit() to modify a previously generated image.

        Args:
            prompt: Text description of the image to generate

        Returns:
            Tuple of (image bytes, mime_type)

        Raises:
            Exception: If image generation fails
        """
        try:
            self._ensure_client()

            logger.info(f"Generating image: {prompt[:100]}...")

            loop = asyncio.get_running_loop()

            def _generate():
                from google.genai import types

                # Create a new chat session for fresh image generation
                self._chat = self._client.chats.create(
                    model=self.MODEL,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

                response = self._chat.send_message(prompt)
                return self._extract_image(response)

            image_bytes, mime_type = await loop.run_in_executor(None, _generate)
            logger.info(f"Image generated: {len(image_bytes)} bytes, type: {mime_type}")
            return image_bytes, mime_type

        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            raise RuntimeError("Image generation not available - google-genai not installed")
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    async def edit(self, instruction: str) -> tuple[bytes, str]:
        """Edit the previously generated image.

        Uses the existing chat session to maintain context, allowing
        Gemini to understand references like "edit that" or "change it".

        Args:
            instruction: What to change (e.g., "add silly hats to everyone")

        Returns:
            Tuple of (edited image bytes, mime_type)

        Raises:
            ValueError: If no previous image exists to edit
            Exception: If image editing fails
        """
        if not self._chat:
            raise ValueError("No previous image to edit. Generate an image first.")

        try:
            logger.info(f"Editing image: {instruction[:100]}...")

            loop = asyncio.get_running_loop()

            def _edit():
                response = self._chat.send_message(instruction)
                return self._extract_image(response)

            image_bytes, mime_type = await loop.run_in_executor(None, _edit)
            logger.info(f"Image edited: {len(image_bytes)} bytes, type: {mime_type}")
            return image_bytes, mime_type

        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise
