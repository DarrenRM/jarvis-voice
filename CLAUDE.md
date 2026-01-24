# Discord Voice Claude Bot

Voice-activated AI assistant for Discord using Claude, Llama 3.3 70B, or GPT-OSS 120B.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy and configure environment
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

Switch between providers with a single env var:

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

```bash
STT_PROVIDER=groq      # groq (fast), deepgram, or whisper (offline)
GROQ_API_KEY=...       # shared with LLM if using Groq for both
DEEPGRAM_API_KEY=...   # if using Deepgram
```

### TTS (Text-to-Speech)

```bash
INWORLD_API_KEY=your_inworld_api_key
TTS_VOICE=alex
```

## Web Search (Llama/GPT-OSS)

When using `LLM_PROVIDER=llama` or `LLM_PROVIDER=gpt-oss`, the bot has web search capabilities via:
- **DuckDuckGo** - free web search (no API key needed)
- **Firecrawl** - URL content extraction (optional, self-hosted)

### Basic Setup (DuckDuckGo only)

Just set `LLM_PROVIDER=llama` or `LLM_PROVIDER=gpt-oss` - web search works out of the box using DuckDuckGo snippets.

### Full Setup with Firecrawl (Optional)

Firecrawl extracts full page content from URLs, giving the LLM more context. **This is optional** - search works fine without it using DuckDuckGo snippets.

```bash
# One-time setup: clone Firecrawl repo
git clone https://github.com/mendableai/firecrawl.git firecrawl-repo

# Build and start (first time takes a few minutes)
docker compose -f docker-compose.firecrawl.yml up -d --build

# Verify it's running
curl http://localhost:3002/

# Stop when done
docker compose -f docker-compose.firecrawl.yml down

# Subsequent starts (no rebuild needed)
docker compose -f docker-compose.firecrawl.yml up -d
```

Configuration (all optional, have defaults):
```bash
FIRECRAWL_URL=http://localhost:3002
FIRECRAWL_API_KEY=              # not needed for self-hosted
SEARCH_MAX_RESULTS=3
```

### How Search Works

1. User asks a question requiring current info
2. Model decides to call `web_search` tool
3. DuckDuckGo returns search results with snippets
4. Optionally, model calls `fetch_url` to get full page content via Firecrawl
5. Model synthesizes answer from results

Without Firecrawl running, `fetch_url` returns an error but search still works with snippets only.

## Image Generation (Llama/GPT-OSS)

When using `LLM_PROVIDER=llama` or `LLM_PROVIDER=gpt-oss`, the bot can generate images using Gemini's Nano Banana API.

### Setup

```bash
NANOBANANA_API_KEY=your_google_ai_api_key
IMAGE_CHANNEL_NAME=general  # Channel to post images (default: "general")
```

Get your API key at https://aistudio.google.com/apikey

### How It Works

1. User asks for an image via voice: "Generate an image of a cat wearing a top hat"
2. Model calls the `generate_image` tool
3. Image is generated via Gemini API (gemini-2.5-flash-image)
4. Image is posted to the configured Discord text channel
5. Model confirms via voice: "Done, I've posted the image to chat"

**Note:** Image generation is only available for Groq providers (Llama/GPT-OSS). Claude uses the Claude Code SDK which doesn't support custom tools.

## Discord Commands

| Command | Description |
|---------|-------------|
| `!join` | Join your voice channel and start listening |
| `!leave` | Leave voice channel |
| `!status` | Show bot status |

## Voice Interaction

1. Say the wake word (default: "hey jarvis")
2. Wait for acknowledgment
3. Ask your question
4. Bot responds with TTS

## Architecture

```
User Speech
    |
[Wake Word Detection] - openwakeword
    |
[VAD] - silero-vad (voice activity detection)
    |
[STT] - Groq Whisper / Deepgram / local Whisper
    |
[LLM] - Claude / Llama 3.3 70B / GPT-OSS 120B
    |     (with tools for Groq: web search, image generation)
    |
    +---> [Image Gen] - Nano Banana (Gemini) --> Discord Text Channel
    |
[TTS] - Inworld
    |
Discord Voice Output
```

## File Structure

```
src/
  bot/           # Discord bot and voice session
  config.py      # Pydantic settings
  detection/     # Wake word and VAD
  image/         # Image generation (for Groq models)
    nano_banana.py     # Gemini API client
  llm/           # LLM clients
    base.py            # Abstract base class
    claude_client.py   # Claude Code SDK client
    groq_client.py     # Groq client (Llama, GPT-OSS)
    llm_factory.py     # Factory for provider selection
  search/        # Web search pipeline (for Groq models)
    duckduckgo.py      # DuckDuckGo search
    firecrawl.py       # Firecrawl scraper
    search_service.py  # Unified service
  stt/           # Speech-to-text
  tts/           # Text-to-speech
```

## Comparing Providers

| Feature | Claude | Llama 3.3 70B | GPT-OSS 120B |
|---------|--------|---------------|--------------|
| Provider | `claude` | `llama` | `gpt-oss` |
| API | Claude Code SDK | Groq | Groq |
| Context | 200K | ~8K | 131K |
| Speed | Moderate | ~750 tok/s | ~500 tok/s |
| Tool use | Native | Native | Native |
| Search | Claude's built-in | DuckDuckGo + Firecrawl | DuckDuckGo + Firecrawl |
| Image Gen | Not supported | Nano Banana (Gemini) | Nano Banana (Gemini) |
| API key | None (Claude auth) | GROQ_API_KEY | GROQ_API_KEY |
| Quality | Excellent | Good | Better |

## Troubleshooting

### Llama/GPT-OSS: "GROQ_API_KEY required"
Set `GROQ_API_KEY` in your `.env` file. Get one at https://console.groq.com

### Firecrawl: "Could not fetch content"
Firecrawl isn't running. Either:
- Start it: `docker compose -f docker-compose.firecrawl.yml up -d`
- Or ignore it - search still works with snippets only

### Claude: "Claude Code SDK not available"
Run `claude auth login` to authenticate, or install the SDK: `pip install claude-code-sdk`

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```
