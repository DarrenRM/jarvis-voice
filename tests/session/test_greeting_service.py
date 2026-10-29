"""Tests for greeting service."""

import time

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.session.greeting_service import GreetingService


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="Hey there! Welcome to the channel!")
    return llm


@pytest.fixture
def mock_tts():
    """Create a mock TTS synthesizer."""
    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.array([0.1, 0.2, 0.3], dtype=np.float32))
    return tts


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.greeting_enabled = True
    settings.greeting_cooldown_seconds = 300.0  # 5 minutes
    settings.greeting_timeout_seconds = 10.0
    return settings


@pytest.fixture
def mock_member():
    """Create a mock Discord member."""
    member = MagicMock()
    member.id = 12345
    member.display_name = "TestUser"
    return member


@pytest.fixture
def greeting_service(mock_llm, mock_tts, mock_settings):
    """Create a greeting service with mocks."""
    return GreetingService(mock_llm, mock_tts, mock_settings)


class TestGreetingServiceProperties:
    """Tests for service properties."""

    def test_cooldown_seconds(self, greeting_service, mock_settings):
        """Should return cooldown from settings."""
        assert greeting_service.cooldown_seconds == 300.0

    def test_timeout_seconds(self, greeting_service, mock_settings):
        """Should return timeout from settings."""
        assert greeting_service.timeout_seconds == 10.0

    def test_enabled(self, greeting_service, mock_settings):
        """Should return enabled from settings."""
        assert greeting_service.enabled is True


class TestCooldownManagement:
    """Tests for cooldown tracking."""

    def test_not_on_cooldown_initially(self, greeting_service):
        """New user should not be on cooldown."""
        assert greeting_service.is_on_cooldown(99999) is False

    def test_on_cooldown_after_greeting(self, greeting_service):
        """User should be on cooldown after update."""
        greeting_service._update_cooldown(12345)
        assert greeting_service.is_on_cooldown(12345) is True

    def test_cooldown_expires(self, greeting_service, mock_settings):
        """Cooldown should expire after duration."""
        mock_settings.greeting_cooldown_seconds = 0.1  # 100ms for testing

        greeting_service._update_cooldown(12345)
        assert greeting_service.is_on_cooldown(12345) is True

        # Wait for cooldown to expire
        time.sleep(0.15)
        assert greeting_service.is_on_cooldown(12345) is False

    def test_clear_cooldown(self, greeting_service):
        """clear_cooldown should remove cooldown for user."""
        greeting_service._update_cooldown(12345)
        assert greeting_service.is_on_cooldown(12345) is True

        greeting_service.clear_cooldown(12345)
        assert greeting_service.is_on_cooldown(12345) is False

    def test_clear_all_cooldowns(self, greeting_service):
        """clear_all_cooldowns should remove all cooldowns."""
        greeting_service._update_cooldown(11111)
        greeting_service._update_cooldown(22222)

        greeting_service.clear_all_cooldowns()

        assert greeting_service.is_on_cooldown(11111) is False
        assert greeting_service.is_on_cooldown(22222) is False


class TestGreetMethod:
    """Tests for the greet() method."""

    @pytest.mark.asyncio
    async def test_greet_success(self, greeting_service, mock_member, mock_llm, mock_tts):
        """Greeting should succeed when all conditions met."""
        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is True
        mock_llm.chat.assert_called_once()
        mock_tts.synthesize.assert_called_once()
        play_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_greet_disabled(self, greeting_service, mock_member, mock_settings):
        """Should skip if greeting disabled."""
        mock_settings.greeting_enabled = False

        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_skipped_when_not_idle_initially(
        self, greeting_service, mock_member
    ):
        """Should skip if bot not idle initially."""
        get_state = MagicMock(return_value="listening")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        wait_silence.assert_not_called()  # Should not even wait
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_skipped_when_processing(self, greeting_service, mock_member):
        """Should skip if bot is processing."""
        get_state = MagicMock(return_value="processing")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_greet_skipped_on_cooldown(self, greeting_service, mock_member):
        """Should skip if user on cooldown."""
        greeting_service._update_cooldown(mock_member.id)

        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        wait_silence.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_skipped_no_silence(self, greeting_service, mock_member):
        """Should skip if no silence detected."""
        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=False)  # No silence found
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_skipped_state_changed_during_wait(
        self, greeting_service, mock_member
    ):
        """CRITICAL: Should skip if state changes during silence wait."""
        # State is idle initially, then changes to listening during wait
        call_count = [0]

        def get_state_with_change():
            call_count[0] += 1
            if call_count[0] == 1:
                return "idle"  # First check: idle
            return "listening"  # After wait: now listening

        get_state = MagicMock(side_effect=get_state_with_change)
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        # Should have checked state twice
        assert get_state.call_count == 2
        # Should not have played audio
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_skipped_state_changed_to_processing(
        self, greeting_service, mock_member
    ):
        """Should skip if state becomes processing during wait."""
        states = ["idle", "processing"]
        get_state = MagicMock(side_effect=states)
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_updates_cooldown(self, greeting_service, mock_member):
        """Should update cooldown after successful greeting."""
        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        assert greeting_service.is_on_cooldown(mock_member.id) is False

        await greeting_service.greet(mock_member, get_state, wait_silence, play_audio)

        assert greeting_service.is_on_cooldown(mock_member.id) is True

    @pytest.mark.asyncio
    async def test_greet_skipped_when_llm_fails(
        self, greeting_service, mock_member, mock_llm
    ):
        """Should skip if LLM fails to generate greeting."""
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM error"))

        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_with_empty_audio(
        self, greeting_service, mock_member, mock_tts
    ):
        """Should return False if TTS returns empty audio."""
        mock_tts.synthesize = AsyncMock(return_value=np.array([], dtype=np.float32))

        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )

        assert result is False
        play_audio.assert_not_called()


class TestGenerateGreeting:
    """Tests for greeting generation."""

    @pytest.mark.asyncio
    async def test_generate_greeting_uses_name(self, greeting_service, mock_llm):
        """Should include user name in prompt."""
        await greeting_service._generate_greeting("Alice")

        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args[0][0]
        assert "Alice" in call_args

    @pytest.mark.asyncio
    async def test_generate_greeting_truncates_long_response(
        self, greeting_service, mock_llm
    ):
        """Should truncate response over 100 chars."""
        long_response = "A" * 150 + " word"
        mock_llm.chat = AsyncMock(return_value=long_response)

        result = await greeting_service._generate_greeting("Test")

        assert len(result) <= 100

    @pytest.mark.asyncio
    async def test_generate_greeting_returns_none_on_error(
        self, greeting_service, mock_llm
    ):
        """Should return None on LLM error."""
        mock_llm.chat = AsyncMock(side_effect=Exception("API error"))

        result = await greeting_service._generate_greeting("Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_greeting_strips_whitespace(
        self, greeting_service, mock_llm
    ):
        """Should strip whitespace from response."""
        mock_llm.chat = AsyncMock(return_value="  Hello there!  \n")

        result = await greeting_service._generate_greeting("Test")

        assert result == "Hello there!"


class TestGreetingServiceEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_multiple_users_independent_cooldowns(self, greeting_service):
        """Different users should have independent cooldowns."""
        greeting_service._update_cooldown(11111)

        assert greeting_service.is_on_cooldown(11111) is True
        assert greeting_service.is_on_cooldown(22222) is False

    @pytest.mark.asyncio
    async def test_greet_same_user_twice_blocked(
        self, greeting_service, mock_member, mock_tts
    ):
        """Second greeting to same user should be blocked by cooldown."""
        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        # First greeting
        result1 = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )
        assert result1 is True

        # Reset mocks for second call
        mock_tts.synthesize.reset_mock()
        play_audio.reset_mock()

        # Second greeting (should be blocked)
        result2 = await greeting_service.greet(
            mock_member, get_state, wait_silence, play_audio
        )
        assert result2 is False
        play_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_greet_different_users_both_succeed(
        self, greeting_service, mock_tts
    ):
        """Greeting different users should both succeed."""
        member1 = MagicMock()
        member1.id = 11111
        member1.display_name = "User1"

        member2 = MagicMock()
        member2.id = 22222
        member2.display_name = "User2"

        get_state = MagicMock(return_value="idle")
        wait_silence = AsyncMock(return_value=True)
        play_audio = AsyncMock()

        result1 = await greeting_service.greet(
            member1, get_state, wait_silence, play_audio
        )
        result2 = await greeting_service.greet(
            member2, get_state, wait_silence, play_audio
        )

        assert result1 is True
        assert result2 is True
