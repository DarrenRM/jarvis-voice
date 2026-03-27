# AGENTS.md

This repository is a Discord voice assistant with real-time audio, wake word detection, STT, LLM orchestration, tool use, and TTS. Agent work here should optimize for small, well-scoped changes that preserve live voice-flow behavior.

## First Steps

Before editing:

1. Read `CLAUDE.md` for product context and env expectations.
2. Check the current git status and do not overwrite unrelated user changes.
3. Skim the relevant module under `src/` and its matching tests under `tests/`.

## Working Agreements

- Prefer focused patches over broad refactors unless explicitly requested.
- Preserve provider flexibility: `claude`, `llama`, and `gpt-oss` are all supported.
- Keep configuration env-driven; do not hardcode secrets, tokens, model names, or local paths.
- Treat `.env` as user-owned. Update `.env.example` when configuration shape changes.
- Avoid breaking streaming and async behavior with blocking calls on hot paths.
- If touching audio, session, or Discord voice code, be extra careful about race conditions and cleanup behavior.

## Useful Commands

```bash
# Install
pip install -e ".[dev]"

# Run the bot
python -m src.main

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## Code Map

- `src/bot/`: Discord bot integration, voice client/session management
- `src/audio/`: audio utilities such as playback and resampling
- `src/detection/`: wake word, VAD, speaking detection
- `src/stt/`: speech-to-text providers
- `src/llm/`: LLM clients, prompt/system prompt builders, orchestration
- `src/tools/`: tool definitions, registry, execution, handlers
- `src/search/`: DuckDuckGo and Firecrawl-backed search/fetch
- `src/tts/`: text filtering and speech synthesis/streaming
- `src/pipeline/`: voice-processing pipeline stages
- `src/session/`: per-user conversational/session behavior
- `src/core/`: shared abstractions and state helpers
- `src/image/`: image generation integration
- `src/sink/`: Discord audio sink and per-user audio state

## Change Guidance

- For provider/tool work:
  keep definitions, registry wiring, handler behavior, and prompt/orchestration expectations aligned.
- For config changes:
  update `src/config.py`, `CLAUDE.md`, and `.env.example` together when needed.
- For prompt changes:
  review both the prompt builder and any tool-calling implications.
- For Discord/session changes:
  verify join/leave lifecycle, per-user isolation, and cleanup paths still make sense.

## Validation Expectations

- Run targeted tests when possible, and `pytest` for broader changes.
- Run `ruff check src/ tests/` after code edits.
- If you cannot run validation, say so clearly in your handoff.

## Notes

- There is a local `firecrawl-repo/` directory in the workspace. Do not modify it unless the task explicitly involves that dependency.
- Logs, local settings, and environment files may be dirty in the worktree. Leave unrelated changes alone.
