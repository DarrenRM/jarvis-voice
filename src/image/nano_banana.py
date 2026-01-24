"""Nano Banana (Gemini) image generation client."""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NanoBananaClient:
    """Client for Gemini image generation API (Nano Banana).

    Uses the gemini-2.5-flash-image model for fast image generation.
    """

    MODEL = "gemini-2.5-flash-image"

    def __init__(self, api_key: str):
        """Initialize the Nano Banana client.

        Args:
            api_key: Google AI API key (NANOBANANA_API_KEY)
        """
        self.api_key = api_key
        self._client = None

    def _ensure_client(self):
        """Lazy initialization of the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    async def generate(self, prompt: str) -> bytes:
        """Generate an image from a text prompt.

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

            # Use synchronous API in thread pool since google-genai doesn't have async
            import asyncio
            loop = asyncio.get_running_loop()

            def _generate():
                from google.genai import types

                response = self._client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

                # Extract image from response
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        # Return raw bytes from inline_data
                        return part.inline_data.data

                raise ValueError("No image in response")

            image_bytes = await loop.run_in_executor(None, _generate)
            logger.info(f"Image generated: {len(image_bytes)} bytes")
            return image_bytes

        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            raise RuntimeError("Image generation not available - google-genai not installed")
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
