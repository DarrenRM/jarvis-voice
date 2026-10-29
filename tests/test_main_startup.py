"""A fresh setup fails at startup with a clear message and a non-zero exit.

Before: missing keys only surfaced at voice-join time (or as a silent bot), and
Discord's privileged-intents refusal was swallowed with exit code 0.
"""

import discord
import pytest

import src.main as main_module
from src.config import Settings

API_KEYS = ("GROQ_API_KEY", "CEREBRAS_API_KEY", "INWORLD_API_KEY")


def make_settings(monkeypatch, **env) -> Settings:
    for key in API_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def all_keys(monkeypatch) -> Settings:
    return make_settings(
        monkeypatch, GROQ_API_KEY="g", CEREBRAS_API_KEY="c", INWORLD_API_KEY="i"
    )


class TestRequiredKeys:
    def test_nothing_missing_when_every_key_is_set(self, monkeypatch):
        assert main_module.missing_settings(all_keys(monkeypatch)) == []

    def test_lists_every_missing_key(self, monkeypatch):
        settings = make_settings(monkeypatch)
        assert main_module.missing_settings(settings) == list(API_KEYS)

    def test_env_example_placeholder_counts_as_missing(self, monkeypatch):
        settings = make_settings(
            monkeypatch, GROQ_API_KEY="g", CEREBRAS_API_KEY="c",
            INWORLD_API_KEY="your_inworld_api_key_here",
        )
        assert main_module.missing_settings(settings) == ["INWORLD_API_KEY"]

    def test_main_exits_non_zero_before_starting(self, monkeypatch, capsys):
        settings = make_settings(monkeypatch, GROQ_API_KEY="g")
        monkeypatch.setattr(main_module, "get_settings", lambda: settings)

        def no_logging(level):
            raise AssertionError("should stop before logging or loading anything")

        monkeypatch.setattr(main_module, "setup_logging", no_logging)

        with pytest.raises(SystemExit) as exited:
            main_module.main()

        assert exited.value.code == 1
        out = capsys.readouterr().out
        assert "CEREBRAS_API_KEY" in out and "INWORLD_API_KEY" in out
        assert "GROQ_API_KEY" not in out


class FakeBot:
    """Logs in, then Discord refuses the gateway connection."""

    def __init__(self, settings, **components):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def login(self, token):
        pass

    async def connect(self):
        raise discord.PrivilegedIntentsRequired(None)


class TestPrivilegedIntents:
    async def test_connection_error_is_raised_not_swallowed(self, monkeypatch):
        monkeypatch.setattr(main_module, "ClaudeBot", FakeBot)
        monkeypatch.setattr(main_module.signal, "signal", lambda *args: None)

        with pytest.raises(discord.PrivilegedIntentsRequired):
            await main_module.run_bot(all_keys(monkeypatch), None)

    def test_main_explains_the_intents_and_exits_non_zero(self, monkeypatch, caplog):
        settings = all_keys(monkeypatch)
        monkeypatch.setattr(main_module, "get_settings", lambda: settings)

        class Listener:
            def stop(self):
                pass

        monkeypatch.setattr(main_module, "setup_logging", lambda level: Listener())
        monkeypatch.setattr(main_module, "preload_components", lambda s: None)

        async def refused(settings, components):
            raise discord.PrivilegedIntentsRequired(None)

        monkeypatch.setattr(main_module, "run_bot", refused)

        with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exited:
            main_module.main()

        assert exited.value.code == 1
        assert "Presence Intent" in caplog.text and "Server Members Intent" in caplog.text
