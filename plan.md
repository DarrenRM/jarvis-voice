# Discord Voice Claude Bot - Implementation Plan

## Goal
Voice-activated AI assistant for Discord with **latency and UX as #1 priority**.

## Key UX Insight
"Hey Claude" -> **Immediate** TTS acknowledgment (pre-cached audio, <150ms) -> Then process question

---

## Latency-Optimized Architecture

### Target Latencies
| Touchpoint | Target | Strategy |
|------------|--------|----------|
| Wake word -> Acknowledgment | <150ms | Pre-loaded WAV files (no TTS generation) |
| Question end -> Processing | <100ms | Silero VAD for endpoint detection |
| Claude first token -> TTS starts | <500ms | Streaming TTS with sentence chunking |
| **Total: Wake to speaking** | **<3s** | Parallel pipelines |

### VRAM Budget (RTX 3080 Ti - 12GB)
| Component | VRAM | Notes |
|-----------|------|-------|
| Whisper Tiny (wake word) | ~100MB | Always running |
| Whisper Medium (transcription) | ~2GB | On standby |
| Qwen3-TTS-0.6B | ~3GB | Smaller model for lower latency |
| Silero VAD | ~10MB | ONNX runtime |
| **Total** | **~5.1GB** | Comfortable headroom |

---

## State Machine

```
IDLE (listening for wake word)
  |-- wake word detected
  v
ACKNOWLEDGED (play "Yes?" immediately)
  |-- audio finished
  v
LISTENING (VAD-bounded recording)
  |-- VAD end-of-speech
  v
PROCESSING (Claude thinking)
  |-- first text chunk
  v
SPEAKING (streaming TTS playback)
  |-- Claude asks question --> CLARIFYING
  |-- finished --> IDLE
```

**Interrupt handling**: VAD detects user speech during SPEAKING -> stop playback -> LISTENING

---

## Project Structure

```
discord-voice-claude-bot/
├── pyproject.toml
├── .env.example
├── src/
│   ├── main.py                 # Entry point
│   ├── config.py               # Pydantic settings
│   ├── bot/
│   │   ├── discord_bot.py      # Bot class, commands
│   │   ├── voice_handler.py    # Voice channel management
│   │   └── state_machine.py    # Conversation states
│   ├── audio/
│   │   ├── pipeline.py         # Discord audio capture
│   │   ├── ring_buffer.py      # Thread-safe buffer
│   │   └── playback.py         # Discord playback
│   ├── detection/
│   │   ├── wake_word.py        # OpenWakeWord / Whisper Tiny
│   │   └── vad.py              # Silero VAD
│   ├── stt/
│   │   └── transcriber.py      # Faster-Whisper
│   ├── llm/
│   │   ├── claude_client.py    # Claude Code SDK
│   │   └── prompt_builder.py   # Context enrichment
│   ├── tts/
│   │   ├── synthesizer.py      # Qwen3-TTS
│   │   └── streaming.py        # Sentence chunking buffer
│   └── context/
│       └── enrichment.py       # Discord Rich Presence
├── assets/audio/               # Pre-recorded acknowledgments
└── tests/
```

---

## Implementation Phases

### Phase 1: Project Setup & Audio Pipeline
- [ ] Create project structure with pyproject.toml
- [ ] Set up Pycord with voice receive support
- [ ] Implement ring buffer for audio capture
- [ ] Integrate Silero VAD for speech detection
- [ ] Test Discord audio capture end-to-end

### Phase 2: Wake Word & STT
- [ ] Integrate OpenWakeWord (or Whisper Tiny fallback)
- [ ] Create pre-recorded acknowledgment audio files
- [ ] Implement Faster-Whisper transcription
- [ ] Connect VAD -> transcription triggering
- [ ] Test: "Hey Claude" -> instant "Yes?" playback

### Phase 3: Claude Code Integration
- [ ] Integrate Claude Code SDK (streaming mode)
- [ ] Implement Discord Rich Presence game detection
- [ ] Build context-aware prompt construction
- [ ] Add conversation session management

### Phase 4: TTS & Streaming Playback
- [ ] Integrate Qwen3-TTS
- [ ] Implement streaming TTS buffer (sentence chunking)
- [ ] Build Discord playback with streaming audio source
- [ ] Connect: Claude stream -> TTS chunks -> Discord playback

### Phase 5: State Machine & Polish
- [ ] Implement full state machine with all transitions
- [ ] Add interrupt handling (user speaks during playback)
- [ ] Error handling and recovery
- [ ] Timeout handling for each state

### Phase 6: Testing & Optimization
- [ ] Latency benchmarking at each stage
- [ ] Unit tests for components
- [ ] Integration tests
- [ ] Performance tuning

---

## Key Dependencies

```toml
[project]
dependencies = [
    "py-cord[voice]>=2.6.0",      # Discord (Pycord fork with voice receive)
    "faster-whisper>=1.0.0",       # STT
    "onnxruntime-gpu>=1.17.0",     # Silero VAD
    "openwakeword>=0.6.0",         # Wake word detection
    "qwen-tts>=0.1.0",             # TTS
    "claude-code-sdk>=0.1.0",      # Claude Code
    "pydantic-settings>=2.0.0",    # Config
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
]
```

---

## Critical Design Decisions (Confirmed)

1. **Pre-cached acknowledgments** - No TTS generation for "Yes?", instant WAV playback (<50ms)
2. **OpenWakeWord** - Custom "Hey Claude" model, <1ms detection, speaker-independent (anyone can trigger)
3. **Whisper Medium** - For full question transcription after wake word
4. **Pycord over discord.py** - Better voice receive support
5. **Streaming TTS** - Sentence chunking allows speaking before full response
6. **Silero VAD** - Low overhead, accurate speech endpoint detection
7. **Qwen3-TTS-0.6B** - Smaller model for lower latency (user confirmed)

---

## Verification Plan

### Manual Testing
1. Start bot, join voice channel via `!join` command
2. Say "Hey Claude" - should hear "Yes?" within 150ms
3. Ask a question - should start hearing response within 3s
4. Interrupt while bot is speaking - should stop and listen
5. Ask about current game - should include game context in response

### Latency Benchmarks
```bash
python scripts/benchmark_latency.py
```
Measures: wake-to-ack, speech-to-transcription, claude-first-token, tts-first-audio

### Integration Tests
- Audio pipeline captures Discord audio correctly
- Wake word triggers with >90% accuracy
- VAD detects speech boundaries accurately
- Claude streaming works with context
- TTS generates and plays correctly

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `.env.example` | Discord token, model settings template |
| `src/main.py` | Entry point, model loading, bot startup |
| `src/config.py` | Pydantic settings for all config |
| `src/bot/state_machine.py` | Core state machine (most critical) |
| `src/audio/pipeline.py` | Discord audio capture |
| `src/detection/wake_word.py` | Wake word with instant acknowledgment |
| `src/tts/streaming.py` | Sentence-chunked streaming TTS |
| `assets/audio/*.wav` | Pre-recorded acknowledgment clips |
