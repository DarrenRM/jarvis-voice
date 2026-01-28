# Discord Voice Claude Bot

Voice-activated AI assistant for Discord.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy and configure environment (note env is up to date and example might be behind)
cp .env.example .env
# Edit .env with your tokens

# Run the bot
python -m src.main
```

## Configuration

### Required

```bash
DISCORD_TOKEN=your_bot_token_here
```

### LLM Provider Selection

Switch between providers (we use gpt-oss primarily) with a single env var:

```bash
# Claude (default) - uses Claude Code SDK, no API key needed
LLM_PROVIDER=claude
CLAUDE_MODEL=haiku  # haiku, sonnet, or opus

# Llama 3.3 70B via Groq - fast inference
LLM_PROVIDER=llama
GROQ_API_KEY=your_groq_api_key_here

# GPT-OSS 120B via Groq - larger context (131K), better reasoning
LLM_PROVIDER=gpt-oss
GROQ_API_KEY=your_groq_api_key_here
```

### STT (Speech-to-Text)

Uses Groq Whisper for fast, accurate transcription.

```bash
GROQ_API_KEY=your_groq_api_key_here  # Required for STT (shared with LLM if using Groq)
```

### TTS (Text-to-Speech)

```bash
INWORLD_API_KEY=your_inworld_api_key
TTS_VOICE=sarah # alex is default
```

## Tools (Groq Providers)

When using `LLM_PROVIDER=llama` or `LLM_PROVIDER=gpt-oss`, the bot has tool capabilities:

| Tool | Description | Setup |
|------|-------------|-------|
| `web_search` | DuckDuckGo search | Works out of the box | only sees results without firecrawl
| `fetch_url` | Extract page content | Optional: Firecrawl via `docker-compose.firecrawl.yml` |
| `generate_image` | Image generation | `NANOBANANA_API_KEY` (Gemini) |
| `deep_research` | Complex research via Claude | `CLAUDE_DELEGATION_ENABLED=true` (default) |
| `send_discord_message` | Post to text channel | Automatic |
| `get_channel_members` | List voice users | Automatic |

Required for Firecrawl setup: `docker compose -f docker-compose.firecrawl.yml up -d`


## Voice Interaction

1. Say the wake word (default: "hey jarvis")
2. Wait for acknowledgment (optional)
3. Ask your question
4. Bot responds with TTS

## Architecture

```
Discord Voice --> [Per-User Session] --> [Wake Word] openwakeword
                                              |
                                         [VAD] silero
                                              |
                                         [STT] Groq Whisper
                                              |
                                         [LLM] Claude / Groq
                                              |
                                         [TTS] Inworld --> Discord Voice
```

**Features:** Auto-join, per-user sessions, speaker isolation, greeting on join, Claude delegation for Groq.

## File Structure

```
src/
  audio/         # Audio utilities (ring buffer, resampling)
  bot/           # Discord bot, commands, voice session
  core/          # Shared types and interfaces
  detection/     # Wake word (openwakeword) and VAD (silero)
  image/         # Image generation (Nano Banana / Gemini)
  llm/           # LLM providers (Claude SDK, Groq GPT OSS 120B)
  pipeline/      # Voice processing pipeline orchestration
  search/        # Web search (DuckDuckGo, Firecrawl)
  session/       # Per-user voice session state
  sink/          # Discord audio sink handling
  stt/           # Speech-to-text (Groq Whisper)
  tools/         # Tool definitions for Groq providers
  tts/           # Text-to-speech (Inworld)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```
