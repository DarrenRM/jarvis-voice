# Discord Voice Bot

A voice assistant that sits in a Discord voice channel. Say the wake word, ask a question out loud, and it answers back in voice.

```
Discord voice -> wake word (openwakeword) -> VAD (Silero) -> STT (Groq Whisper)
              -> LLM (Qwen 3.8 27B on Cerebras) -> TTS (Inworld) -> Discord voice
```

Each person in the channel gets their own session, and only the person who said the wake word is listened to for the question. The bot can also search the web, post in text channels, generate images, and say hi when people join.

> **⚠️ Heads up: this needs a cleanup pass.** Most of this code was written by early-2026 AI coding agents. It works, but it has real technical debt, bugs, and some architectural mistakes (for example, the voice session class does far too much). Before building on it, have a current frontier coding agent review the whole codebase and refactor it. Treat it as a working prototype, not a clean foundation.

## What you need

- **Python 3.11+** (3.12 recommended)
- **A Discord bot token** (see below)
- **Cerebras API key**: <https://cloud.cerebras.ai>. Runs the LLM (Qwen 3.8 27B). Use pay-as-you-go: the free tier's 5 requests/minute is too low for voice.
- **Groq API key**: free tier at <https://console.groq.com>. Used for speech-to-text and the short "thinking" phrases.
- **Inworld API key**: <https://studio.inworld.ai>. Used for the bot's voice.
- Optional: **Gemini API key** (`NANOBANANA_API_KEY`) for image generation (also set `IMAGE_GENERATION_ENABLED=true`).
- Linux/macOS: install `ffmpeg` and `libopus` (e.g. `apt install ffmpeg libopus0`). Windows works out of the box.

## 1. Create the Discord bot

1. Go to <https://discord.com/developers/applications> and click **New Application**.
2. **Bot** tab: click **Reset Token** and copy the token. This is your `DISCORD_TOKEN`.
3. On the same tab, turn on **Message Content Intent** under *Privileged Gateway Intents*.
4. **OAuth2 → URL Generator**: tick the `bot` scope, then the permissions **View Channels, Send Messages, Attach Files, Read Message History, Connect, Speak, Use Voice Activity**.
5. Open the generated URL and add the bot to your server.

## 2. Install

```bash
git clone <this repo>
cd claude-bot
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -e .
```

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env`. These are the settings you'll want to change:

```bash
DISCORD_TOKEN=...
CEREBRAS_API_KEY=...
GROQ_API_KEY=...
INWORLD_API_KEY=...

TTS_VOICE=sarah               # any Inworld voice: alex, sarah, luna, hades, ...

# Optional
AUTO_JOIN_CHANNEL=General     # voice channel to join on startup (default: first one with people)
IMAGE_GENERATION_ENABLED=true # plus NANOBANANA_API_KEY=... for image generation
```

### Set up the people in your server

The bot works out who is talking and addresses them by name. Names are also used when you say things like *"tag Bob and tell him dinner's ready."* Out of the box it uses each person's Discord display name. To give people friendlier names (handy when display names are like `xX_n00bslayer_Xx`), map their user IDs:

1. In Discord: **User Settings → Advanced → Developer Mode** on.
2. Right-click each person → **Copy User ID**.
3. Add them to `.env` as JSON (IDs as strings):

```bash
USER_NAMES={"123456789012345678": "Alice", "234567890123456789": "Bob"}
```

You can also give the bot some optional running jokes. It will *rarely* work them into answers, and if someone asks it to roast a person directly, it roasts the asker instead:

```bash
USER_ROASTS={"Alice": "always goes AFK mid-raid.", "Bob": "has never won a game."}
```

Leave `USER_ROASTS` empty (the default) to turn this off.

### Everything else

`.env.example` lists every other option with comments (VAD sensitivity, timeouts, keep-alive, greeting, and so on). The defaults are fine to start with.

The personality prompt still has some leftovers from the original server. It's tuned for Dune: Awakening (it knows about `awakening.wiki` and has gaming jokes). To change it, edit `src/llm/system_prompt_builder.py`.

## 4. Run

```bash
python -m src.main
```

The bot logs in and automatically joins a voice channel. Then:

1. Say **"hey jarvis"**
2. Ask your question (you can say it in one breath, e.g. *"hey jarvis what's the weather in Tokyo"*)
3. Wait for the spoken answer

Text commands in any channel:

| Command   | What it does                     |
|-----------|----------------------------------|
| `!join`   | Join the voice channel you're in |
| `!leave`  | Leave voice                      |
| `!status` | Show session status              |

## Tools the bot can use

| Tool | What it does | Setup |
|------|--------------|-------|
| `web_search` | DuckDuckGo search (titles, links, snippets) | Nothing needed |
| `web_extract` | Read one page's main text locally (Trafilatura) | Nothing needed |
| `tag_user` | Mention someone in the text channel | Nothing needed |
| `post_to_chat` / `post_table` | Post a message or table to the text channel | Nothing needed |
| `generate_image` / `edit_image` | Make or edit an image and post it | `IMAGE_GENERATION_ENABLED=true` + `NANOBANANA_API_KEY` |

Web lookup runs entirely on your machine: no API keys, no hosted scraping service. The page reader only fetches public http(s) addresses (private/local network addresses are blocked on every redirect) and caps size and time. Pages that need JavaScript can't be read; the bot falls back to the search snippets. Turn the web tools off with `WEB_TOOLS_ENABLED=false`.

## Keep-alive

Discord kicks idle voice bots after about an hour, so every 50 minutes the bot says something short and slightly annoyed. To change it:

```bash
KEEP_ALIVE_ENABLED=true
KEEP_ALIVE_INTERVAL_SECONDS=3000
KEEP_ALIVE_PROMPT="Say something short to stay connected"
```

## Troubleshooting

- **Bot joins but never responds:** check the logs for wake word detections. Try lowering `WAKE_WORD_CONFIDENCE` (e.g. `0.4`) and say "hey jarvis" clearly.
- **It cuts you off mid-sentence:** raise `MIN_SILENCE_MS` / `SPEECH_PAD_MS`.
- **No audio comes out:** check `INWORLD_API_KEY`, and on Linux check that `ffmpeg` and `libopus` are installed.
- **Commands do nothing:** Message Content Intent isn't turned on in the developer portal.

## Project layout

```
src/
  bot/        Discord client, commands, per-channel voice session
  detection/  Wake word + VAD
  stt/        Speech-to-text
  llm/        Cerebras client, tool loop, system prompt
  tts/        Inworld TTS
  tools/      Tool handlers (web, images, Discord actions)
  web/        DuckDuckGo search + local page reader
  image/      Image generation
  pipeline/   Voice pipeline orchestration
  config.py   All settings (read from .env)
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```
