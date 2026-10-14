"""Dynamic thinking phrase generator using fast LLM.

Generates context-aware thinking phrases based on the user's request.
The name/personalization is handled separately by affirmation phrases.

Examples:
- "I'll check that weather"
- "Let me think of something funny"
- "I'll look that up"
"""

import logging

logger = logging.getLogger(__name__)

# Prompt template for generating dynamic thinking phrases
# Note: No name needed - affirmation phrase handles personalization
THINKING_PHRASE_PROMPT = """Generate a SHORT (3-8 words) content phrase for this request, OR return SKIP if none needed.
"{user_request}"

Return SKIP for simple requests that don't need a thinking phrase:
- Greetings: "how are you?", "what's up?"
- Simple questions: "what's your name?", "tell me a joke"
- Direct conversation: "thanks", "that's funny", "I agree"

Return a phrase for requests that involve work:
- "What's the weather in Hawaii?" -> "I'll check that weather"
- "Look up ingredients for ornithopter" -> "I'll look that up"
- "Can you search for pasta recipes?" -> "I'll find some recipes"
- "What time is it in Tokyo?" -> "let me check"
- "Generate an image of a cat" -> "I'll create that"

Rules:
- Return SKIP for simple conversational requests (no lookup/search/generation needed)
- 3-8 words MAXIMUM if returning a phrase
- Focus on the ACTION, not greeting
- Do NOT include names or "sure" (handled separately)
- Start with "I'll", "let me", or similar action words

Response (SKIP or phrase only, no quotes):"""


class ThinkingPhraseGenerator:
    """Generates context-aware thinking phrases via fast LLM call.

    Uses Qwen on Cerebras with reasoning off: a phrase this short doesn't need
    it, and every millisecond here is dead air.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-3.8-27b",
        base_url: str = "https://api.cerebras.ai/v1",
    ):
        """Initialize the generator.

        Args:
            api_key: Cerebras API key.
            model: Model to use for generation (default: qwen-3.8-27b).
            base_url: OpenAI-compatible endpoint (default: Cerebras).
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of the OpenAI-compatible client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url, timeout=5.0, max_retries=0
            )

    async def generate(self, transcribed_text: str) -> str:
        """Generate a dynamic thinking phrase based on the user's request.

        Note: This generates only the content-aware part (e.g., "I'll check that weather").
        The personalization (e.g., "Sure, {name}...") is handled separately by
        affirmation phrases that play immediately.

        Args:
            transcribed_text: The user's transcribed speech.

        Returns:
            A context-aware thinking phrase (no name included).

        Raises:
            Exception: If LLM call fails (caller should handle gracefully).
        """
        self._ensure_client()

        prompt = THINKING_PHRASE_PROMPT.format(user_request=transcribed_text)

        logger.debug(f"[THINKING] Generating phrase for: {transcribed_text[:50]}...")

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=30,  # Very short response
            temperature=0.7,  # Some variety
            reasoning_effort="none",
        )

        phrase = response.choices[0].message.content.strip()

        # Clean up: remove quotes if LLM added them
        phrase = phrase.strip('"\'')

        logger.info(f"[THINKING] Generated phrase: {phrase}")
        return phrase

    async def warm_up(self) -> None:
        """Open the connection ahead of the first request (first call is ~2s slower)."""
        self._ensure_client()
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1,
                reasoning_effort="none",
            )
        except Exception as e:
            logger.warning(f"[WARMUP] Thinking phrase warm-up failed: {e}")
