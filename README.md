# Jarvis: a voice assistant for Discord calls

Say "Hey Jarvis" in a Discord voice channel, ask a question out loud, and hear the answer back in the call.

## What it is

Jarvis joins a voice channel and listens to everyone in it. It was built for a group's game nights: ask for a crafting recipe, a patch change or a stat while you keep playing, get a spoken answer, and get the recipe or table posted in the text chat too. It looks things up on the web, tags people, and (when enabled) generates images.

The first version ran on Claude, which is where the `claude-bot` command and the `discord-voice-claude-bot` package name come from. It now runs on Qwen on Cerebras.

## What's interesting

- **One wake-word detector per speaker.** Each person in the call gets their own openWakeWord detector (created on demand, capped at 10). "Hey Jarvis" is attributed to whoever said it at the moment it fires, and only that person's audio is transcribed for the question. See `src/sink/`.
- **Hearing people out.** People pause mid-question ("um... so..."), so the bot waits for 2 s of quiet before it treats a question as finished. If the speaker carries on before the answer starts playing, it cancels that answer and starts over with everything they said. See `_carried_on` in `src/bot/voice_session.py` and the replays in `tests/irl/test_carry_on_irl.py`.
- **Knowing when not to answer.** After a reply, the person it answered can carry on without the wake word. Each follow-up first goes through an addressee check (is this for the bot, or for the friend next to them?), and anything uncertain counts as not addressed. With speaking detection on (as in `.env.example`), follow-up windows only count down while the room is quiet. See `src/llm/addressee_judge.py`.
- **Speech starts before the answer is finished.** A short opening phrase ("Sure, Sam...") starts as soon as the question is over, while Whisper is still transcribing. The model's reasoning never reaches TTS or the chat history, and the answer is split into sentences and synthesized as it streams. In the scripted e2e runs, first audio starts a median 2.1 s after the speaker's last word. 2.0 s of that is the deliberate end-of-question pause, so the bot is speaking about 0.1 s after it decides the question is over.
- **A monitor that watches real turns.** Detectors flag error replies, give-ups, slow or silent turns, tool failures and TTS repeats, and post them to a Discord channel where the owner marks each one. See "How it was built" for what happens next.
- **Voice receive under Discord's DAVE encryption.** Discord now requires DAVE end-to-end encryption, and py-cord 2.8's voice-receive path for it is unfinished (py-cord issue 3139). `src/bot/pycord_dave_patch.py` replaces the broken decrypt step and drops undecodable packets instead of crashing. It applies only to the py-cord releases it was written against, and `pyproject.toml` pins those.

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

Create a Discord application and copy the bot token. On its Bot page, turn on the three privileged intents the bot requests: Presence Intent (which game each speaker is playing), Server Members Intent (keeps members cached for that lookup) and Message Content Intent (the `!join`, `!leave` and `!status` commands). Without them Discord refuses the connection, and the bot exits with a message naming the switches to turn on. Invite the bot with permission to view channels, send messages, attach files, read message history, add reactions (the monitor puts ✅ and ❌ on its posts), connect, speak and use voice activity.

Required environment variables (names only; `.env.example` lists every option). The bot checks them at startup and exits with a list of any that are missing:

- `DISCORD_TOKEN`
- `CEREBRAS_API_KEY` for the Qwen model
- `GROQ_API_KEY` for Whisper speech-to-text
- `INWORLD_API_KEY` for the bot's voice

Optional: `USER_NAMES` (map Discord user IDs to friendly names), `TINYFISH_API_KEY` (hosted web search), `TYPESAFE_API_KEY` (TypeSafe Jev makes the follow-up addressee check instead of Qwen, and the key also turns on the automatic chat posts of recipes, steps and stats), `NANOBANANA_API_KEY` with `IMAGE_GENERATION_ENABLED` (images), and `MONITOR_ENABLED` with the other `MONITOR_*` settings (live monitor, off by default). Posts, tags and images go to `IMAGE_CHANNEL_NAME` (default `general`).

Two settings make the bot speak unprompted and are on by default: it greets people who join its voice channel (`GREETING_*`), and after 50 idle minutes it says a short line so Discord doesn't drop the connection (`KEEP_ALIVE_*`).

The `hey_jarvis` wake word is openWakeWord's pretrained model, downloaded at startup on first run. openWakeWord's pretrained models are licensed CC BY-NC-SA 4.0 (non-commercial).

Tests run offline and need no keys or `.env`:

```bash
.venv/Scripts/python -m pytest -q
```

The end-to-end audio harness (`scripts/e2e/run_voice_e2e.py`) drives the real pipeline with spoken scenarios and needs the live Groq, Cerebras and Inworld keys. It never connects to Discord unless run with `--listen`.

## Status

In use at game nights by four regulars throughout 2026. The suite has 802 automated tests, and the audio harness covers 45 scripted scenarios and can replay clips saved from live turns. The recordings are real voices from game nights and are kept private, not in this repository.

## How it was built

Darren Mills designed Jarvis and directed the AI coding agents that built it. He set the product rules (hear people out, stay quiet when unsure who is being addressed, never read links aloud) and designed the loop that checks the bot against real game nights:

1. The live monitor (`src/monitor/`) flags problem turns in a Discord channel.
2. Darren marks each one: a check means something went wrong, a cross means no problem, or nothing that disrupted the game.
3. Confirmed incidents become timeline replays in `tests/irl`, run offline through the real sink and session code, and spoken `irl_*` scenarios in `scripts/e2e`. Each fix is checked against what actually happened, and stays fixed.

`AGENTS.md` and `CLAUDE.md` hold the rules the agents work under. Most commits carry the agents' `Co-Authored-By` trailers, and `docs/archive/` keeps the January plans he directed.

## License

MIT. See `LICENSE`.
