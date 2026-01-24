"""Nano Banana (Gemini) image generation and editing client."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NanoBananaClient:
    """Client for Gemini image generation and multi-turn editing.

    Uses Gemini's chat interface for image generation, which enables
    multi-turn editing (e.g., "edit that to add silly hats").
    """

    MODEL = "gemini-2.0-flash-exp"

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

    def _extract_image(self, response) -> bytes:
        """Extract image bytes from a Gemini response.

        Args:
            response: Gemini chat response

        Returns:
            Image bytes

        Raises:
            ValueError: If no image found in response
        """
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                return part.inline_data.data
        raise ValueError("No image in response")

    async def generate(self, prompt: str) -> bytes:
        """Generate a NEW image from a text prompt.

        This starts a fresh chat session, discarding any previous image context.
        Use edit() to modify a previously generated image.

        Args:
            prompt: Text description of the image to generate

        Returns:
            PNG image as bytes

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

            image_bytes = await loop.run_in_executor(None, _generate)
            logger.info(f"Image generated: {len(image_bytes)} bytes")
            return image_bytes

        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            raise RuntimeError("Image generation not available - google-genai not installed")
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    async def edit(self, instruction: str) -> bytes:
        """Edit the previously generated image.

        Uses the existing chat session to maintain context, allowing
        Gemini to understand references like "edit that" or "change it".

        Args:
            instruction: What to change (e.g., "add silly hats to everyone")

        Returns:
            Edited image as bytes

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

            image_bytes = await loop.run_in_executor(None, _edit)
            logger.info(f"Image edited: {len(image_bytes)} bytes")
            return image_bytes

        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise
