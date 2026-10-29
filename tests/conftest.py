"""Test setup: run offline with no .env and no real keys."""

import os

# Settings requires a Discord token. A dummy one lets a fresh clone run the suite;
# tests never connect to Discord. A real DISCORD_TOKEN in the shell is left alone.
os.environ.setdefault("DISCORD_TOKEN", "test-token")
