"""Greeting service for voice channel users.

Handles greeting users who join the voice channel with LLM-generated
personalized greetings, including cooldown management.
"""

import logging
import time
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import discord
    from ..llm import BaseLLMClient
    from ..tts import TTSSynthesizer
    from ..config import Settings

logger = logging.getLogger(__name__)


class GreetingService:
    """Manages greetings for users joining voice channels.

    Features:
    - Cooldown tracking per user
    - Silence detection before greeting
    - State recheck after async waits (critical for avoiding race conditions)
    - LLM-generated personalized greetings
    """

    def __init__(
        self,
        llm_client: "BaseLLMClient",
        tts_synthesizer: "TTSSynthesizer",
        settings: "Settings",
    ):
        """Initialize greeting service.

        Args:
            llm_client: LLM client for generating greetings.
            tts_synthesizer: TTS for synthesizing greeting audio.
            settings: Application settings.
        """
        self._llm = llm_client
        self._tts = tts_synthesizer
        self._settings = settings
        self._cooldowns: dict[int, float] = {}

    @property
    def cooldown_seconds(self) -> float:
        """Get cooldown duration in seconds."""
        return self._settings.greeting_cooldown_seconds

    @property
    def timeout_seconds(self) -> float:
        """Get silence wait timeout in seconds."""
        return self._settings.greeting_timeout_seconds

    @property
    def enabled(self) -> bool:
        """Check if greeting is enabled."""
        return self._settings.greeting_enabled

    def is_on_cooldown(self, user_id: int) -> bool:
        """Check if user is on greeting cooldown.

        Args:
            user_id: Discord user ID to check.

        Returns:
            True if user was greeted within cooldown period.
        """
        last_greeting = self._cooldowns.get(user_id, 0)
        return (time.time() - last_greeting) < self.cooldown_seconds

    def _update_cooldown(self, user_id: int) -> None:
        """Update cooldown timestamp for user.

        Args:
            user_id: Discord user ID to update.
        """
        self._cooldowns[user_id] = time.time()

    def clear_cooldown(self, user_id: int) -> None:
        """Clear cooldown for a user (for testing).

        Args:
            user_id: Discord user ID to clear.
        """
        self._cooldowns.pop(user_id, None)

    def clear_all_cooldowns(self) -> None:
        """Clear all cooldowns (for testing)."""
        self._cooldowns.clear()

    async def greet(
        self,
        member: "discord.Member",
        get_current_state: Callable[[], str],
        wait_for_silence: Callable[[], Awaitable[bool]],
        play_audio: Callable[[np.ndarray], Awaitable[None]],
    ) -> bool:
        """Greet a user who joined the voice channel.

        Only greets if:
        - Greeting is enabled in settings
        - Bot is idle (not listening/processing)
        - User hasn't been greeted within cooldown period
        - Silence is detected within timeout period
        - Bot is still idle after waiting (CRITICAL - prevents race conditions)

        Args:
            member: Discord member who joined.
            get_current_state: Callback to get current session state.
            wait_for_silence: Callback to wait for channel silence.
            play_audio: Callback to play synthesized audio.

        Returns:
            True if greeting was played, False otherwise.
        """
        if not self.enabled:
            return False

        # Check 1: Bot must be idle
        if get_current_state() != "idle":
            logger.info(f"[GREETING] Skipping - bot is busy ({get_current_state()})")
            return False

        # Check 2: Cooldown
        if self.is_on_cooldown(member.id):
            logger.info(f"[GREETING] Skipping {member.display_name} - cooldown active")
            return False

        # Wait for silence (this may take several seconds)
        found_silence = await wait_for_silence()
        if not found_silence:
            logger.info(
                f"[GREETING] Skipping {member.display_name} - "
                f"no silence in {self.timeout_seconds}s"
            )
            return False

        # CRITICAL: Re-check state after async wait
        # State may have changed during silence wait (wake word detected, etc.)
        if get_current_state() != "idle":
            logger.info("[GREETING] Skipping - bot became busy during wait")
            return False

        # Resolve name: prefer configured mapping, fall back to display name
        name = self._settings.user_names.get(member.id, member.display_name)

        # Generate greeting via LLM
        greeting = await self._generate_greeting(name)
        if not greeting:
            return False

        # Update cooldown BEFORE playing (in case play fails, still on cooldown)
        self._update_cooldown(member.id)

        # Synthesize and play
        logger.info(f"[GREETING] Greeting {name}: {greeting}")
        audio = await self._tts.synthesize(greeting)
        if len(audio) > 0:
            await play_audio(audio)
            return True

        return False

    async def _generate_greeting(self, name: str) -> Optional[str]:
        """Generate a short greeting for the user via LLM.

        Uses chat() not quick_response() because chat() uses the full
        system prompt which includes personality and roast info.

        Args:
            name: Display name of user to greet.

        Returns:
            Greeting text or None if generation failed.
        """
        prompt = (
            f"Generate a very short greeting (4 words max) for {name} "
            f"who just joined the voice channel."
        )

        try:
            response = await self._llm.chat(prompt)
            # Ensure it's short for TTS
            if len(response) > 100:
                response = response[:100].rsplit(' ', 1)[0]
            return response.strip()
        except Exception as e:
            logger.error(f"[GREETING] LLM error: {e}")
            return None
