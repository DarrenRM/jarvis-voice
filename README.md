# Jarvis: a voice assistant for Discord calls

Say "Hey Jarvis" in a Discord voice channel, ask a question out loud, and hear the answer back in the call.

## What it is

Jarvis joins a voice channel and listens to everyone in it. It was built for a group's game nights: ask for a crafting recipe, a patch change or a stat while you keep playing, get a spoken answer, and get the recipe or table posted in the text chat too. It looks things up on the web, tags people, and (when enabled) generates images.

The repository and Python package are still called `claude-bot` after the first version, which used Claude. It now runs on Qwen.

## What's interesting

- **One wake-word detector per speaker.** Each person in the call gets their own openWakeWord detector (created on demand, capped at 10). "Hey Jarvis" is attributed to whoever said it at the moment it fires, and only that person's audio is transcribed for the question. See `src/sink/`.
- **Knowing when not to answer.** After a reply, the person it answered can carry on without the wake word. Each follow-up first goes through an addressee check (is this for the bot, or for the friend next to them?), and anything uncertain counts as not addressed. Follow-up windows only count down while the room is quiet. See `src/llm/addressee_judge.py`.
- **Speech starts before the answer is finished.** A short filler phrase plays as soon as the speaker stops. The model's reasoning never reaches TTS or the chat history, and the answer is split into sentences and synthesized as it streams. In the e2e harness a good turn starts audio about 0.6 to 0.9 s after the question ends.
- **A monitor that watches real turns.** Detectors flag error replies, give-ups, slow or silent turns, tool failures and TTS repeats, and post them to a Discord channel where the owner reacts with a check or a cross. Real incidents are then replayed offline through the real sink and session code (`tests/irl`) so fixed bugs stay fixed.
- **A workaround for an unfinished library path.** Discord now requires DAVE end-to-end encryption, and py-cord 2.8's voice-receive path for it is unfinished (py-cord issue 3139). `src/bot/pycord_dave_patch.py` replaces the broken decrypt step and drops undecodable packets instead of crashing.

## How it works

```
Discord voice (per-speaker audio)
  -> wake-word detector per speaker -> VAD (Silero) -> speech-to-text (Whisper on Groq)
  -> Qwen on Cerebras, with a tool loop (web search and page reading, chat posts, tagging, images)
  -> sentence-by-sentence TTS (Inworld) -> playback into the channel
```

- `src/bot/` holds the Discord client and one `VoiceSession` per channel that wires the rest together.
- `src/pipeline/` has the transcription, response and TTS stages; `src/llm/` has the Cerebras client, the tool loop and the prompt builder.
- `src/web/` searches (TinyFish if keyed, otherwise ddgs) and reads pages locally. Web content is treated as untrusted, and the page reader rejects private, loopback and link-local addresses on every redirect hop.
- Per-game patch notes can be dropped in `data/game_notes/`. They go into the prompt while someone in the call is playing that game, so answers about a new release don't come from stale model memory. The notes themselves are not in the repo.
- Everything is configured through environment variables (`src/config.py`, pydantic-settings).

## Running it locally

Developed and run on Windows with Git Bash. On Linux or macOS, install `ffmpeg` and `libopus` first. Python 3.11 or newer.

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
cp .env.example .env            # then fill in the keys below
.venv/Scripts/python -m src.main
```

Without `uv`: `python -m venv .venv`, activate it, then `pip install -e ".[dev]"`. On Linux or macOS the interpreter is `.venv/bin/python`.

Create a Discord application, copy the bot token, turn on Message Content Intent, and invite the bot with permission to view channels, send messages, attach files, read message history, connect, speak and use voice activity.

Required environment variables (names only; `.env.example` lists every option):

- `DISCORD_TOKEN`
- `CEREBRAS_API_KEY` for the Qwen model
- `GROQ_API_KEY` for Whisper speech-to-text
- `INWORLD_API_KEY` for the bot's voice

Optional: `USER_NAMES` (map Discord user IDs to friendly names), `TINYFISH_API_KEY` (hosted web search), `TYPESAFE_API_KEY` (a faster addressee check), `NANOBANANA_API_KEY` with `IMAGE_GENERATION_ENABLED` (images), and `MONITOR_ENABLED` with the other `MONITOR_*` settings (live monitor, off by default).

Tests run offline and need no keys or `.env`:

```bash
.venv/Scripts/python -m pytest -q
```

The end-to-end audio harness (`scripts/e2e/run_voice_e2e.py`) drives the real pipeline with spoken scenarios and needs the live Groq, Cerebras and Inworld keys. It never connects to Discord unless run with `--listen`.

## Status

In use at every game night by four regulars since January 2026. Python on py-cord, with a wake detector for each speaker; Whisper speech-to-text, a Qwen model on Cerebras, and streaming speech. The suite has 867 automated tests, and the audio harness covers 45 scripted scenarios and can replay clips saved from live turns. The recordings are real voices from game nights and are kept private, not in this repository.

## How it was built

Designed and directed by Darren Mills, built with AI coding agents. See `AGENTS.md` and `CLAUDE.md` for the rules the agents work under. Most commits carry the agents' `Co-Authored-By` trailers.

## License

No license yet.
