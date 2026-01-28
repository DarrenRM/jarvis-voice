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

    Uses Groq's Llama 3.3 70B for fast inference to generate personalized
    thinking phrases based on what the user actually asked.
    """

    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        """Initialize the generator.

        Args:
            groq_api_key: Groq API key for LLM calls.
            model: Model to use for generation (default: llama-3.3-70b-versatile).
        """
        self._api_key = groq_api_key
        self._model = model
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self._api_key)

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
            max_tokens=30,  # Very short response
            temperature=0.7,  # Some variety
        )

        phrase = response.choices[0].message.content.strip()

        # Clean up: remove quotes if LLM added them
        phrase = phrase.strip('"\'')

        logger.info(f"[THINKING] Generated phrase: {phrase}")
        return phrase
