"""Tests for configuration parsing."""

import os
import pytest


class TestUserNamesConfig:
    """Tests for user_names configuration field."""

    def test_default_user_names(self, monkeypatch):
        """Default user_names is empty (configured per-server via USER_NAMES)."""
        # Clear any existing USER_NAMES env var
        monkeypatch.delenv("USER_NAMES", raising=False)
        # Set required env vars
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        # Force reimport to get fresh settings
        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings(_env_file=None)

        assert s.user_names == {}

    def test_parse_json_from_env(self, monkeypatch):
        """USER_NAMES env var parses JSON correctly."""
        monkeypatch.setenv("USER_NAMES", '{"111": "TestUser", "222": "AnotherUser"}')
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings(_env_file=None)

        assert s.user_names == {111: "TestUser", 222: "AnotherUser"}

    def test_empty_json_object(self, monkeypatch):
        """Empty JSON object is valid."""
        monkeypatch.setenv("USER_NAMES", '{}')
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings(_env_file=None)

        assert s.user_names == {}

    def test_user_names_int_keys(self, monkeypatch):
        """JSON string keys are converted to int."""
        monkeypatch.setenv("USER_NAMES", '{"123456789": "User"}')
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings(_env_file=None)

        # Keys should be int, not string
        assert 123456789 in s.user_names
        assert "123456789" not in s.user_names


class TestSettingsSingleton:
    """Tests for settings singleton behavior."""

    def test_get_settings_returns_settings(self, monkeypatch):
        """get_settings returns a Settings instance."""
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import get_settings, Settings

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_singleton(self, monkeypatch):
        """get_settings returns same instance."""
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestRequiredSettings:
    """Tests for required configuration fields."""

    def test_discord_token_required(self, monkeypatch, tmp_path):
        """DISCORD_TOKEN is required when no .env file exists."""
        # Create empty .env to override the real one
        monkeypatch.delenv("DISCORD_TOKEN", raising=False)
        monkeypatch.chdir(tmp_path)  # Change to temp dir without .env

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # Explicitly disable .env file


class TestOptionalSettings:
    """Tests for optional configuration fields with defaults."""

    def test_default_llm_settings(self, monkeypatch, tmp_path):
        """Cerebras model, reasoning and feature flags have expected defaults."""
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")
        for var in ("CEREBRAS_MODEL", "CEREBRAS_REASONING_EFFORT",
                    "IMAGE_GENERATION_ENABLED", "DEEP_RESEARCH_ENABLED"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)  # Avoid .env leakage

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings(_env_file=None)  # Explicitly disable .env file

        assert s.cerebras_model == "qwen-3.8-27b"
        assert s.cerebras_reasoning_effort == "high"
        assert s.image_generation_enabled is False

    def test_default_tts_voice(self, monkeypatch):
        """TTS voice has a default."""
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings()

        assert s.tts_voice is not None
        assert isinstance(s.tts_voice, str)

    def test_default_log_level(self, monkeypatch):
        """Log level defaults to INFO."""
        monkeypatch.setenv("DISCORD_TOKEN", "fake_token")

        import importlib
        import src.config
        importlib.reload(src.config)

        from src.config import Settings
        s = Settings()

        assert s.log_level == "INFO"
