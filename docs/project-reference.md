# Discord Voice Claude Bot - Project Reference

A comprehensive reference document for understanding and maintaining this voice-activated AI assistant for Discord.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Audio Pipeline](#audio-pipeline)
4. [Critical Timing Constants](#critical-timing-constants)
5. [Key Components](#key-components)
6. [Code Patterns](#code-patterns)
7. [LLM Integration](#llm-integration)
8. [Configuration Reference](#configuration-reference)
9. [File Reference](#file-reference)
10. [Design Decisions](#design-decisions)

---

## Overview

Voice-activated AI assistant for Discord voice channels supporting:
- **Claude** (via Claude Code SDK - no API key needed)
- **Llama 3.3 70B** (via Groq - ~750 tok/s)
- **GPT-OSS 120B** (via Groq - 131K context)

The bot listens for a wake word ("hey jarvis"), captures user speech, transcribes it, generates a response, and speaks it back with natural pacing.

### Key Innovations

1. **Pre-buffering**: Captures continuous speech like "hey jarvis what's the weather?" without pause
2. **Parallel TTS streaming**: LLM generates while TTS synthesizes previous sentences
3. **PTT staleness detection**: Responds 500ms after push-to-talk release (not 3.5s)
4. **Speaker isolation**: Filters non-wake-word speakers in multi-user channels
5. **Personalization**: "One second, Darren..." with Discord activity awareness

---

## Architecture

### High-Level Flow

```
User Speech (Discord 48kHz stereo)
       |
       v
[StreamingSink] -----> per-user pre-buffers (1.5s rolling)
       |
[Wake Word] ---------> "hey jarvis" detection (OpenWakeWord)
       |
[VAD] ---------------> speech endpoint detection (Silero)
       |
[Resample] ----------> 48kHz -> 16kHz (resample_poly, 10-50x faster)
       |
[STT] ---------------> Groq Whisper (~189x realtime)
       |
[SpeakerManager] ----> name + activity enrichment
       |
[LLM] ---------------> Claude / Llama / GPT-OSS
       |                   |
       |              [Tools] -> web_search, generate_image, tag_user
       |
[Parallel TTS] ------> Sentence-level streaming (Inworld 24kHz)
       |
[Resample] ----------> 24kHz -> 48kHz
       |
       v
Discord Voice Output
```

### Dual State Machine Architecture

The system uses **two coordinated state machines** running in different contexts:

1. **StreamingSink** (Discord's audio thread)
   - Processes audio 50x/second
   - Runs wake word detection and VAD
   - Thread-safe callbacks to VoiceSession

2. **VoiceSession** (asyncio event loop)
   - Orchestrates pipeline: STT -> LLM -> TTS
   - Manages playback queue and parallel TTS
   - Uses `call_soon_threadsafe()` for thread safety

```
Discord Thread                    Asyncio Event Loop
     |                                   |
[StreamingSink]  --wake_event-->  [VoiceSession]
     |           --speech_end-->        |
     |                                  |
   (audio)       <--state change--    (control)
```

---

## Audio Pipeline

### Sample Rate Architecture

| Stage | Rate | Format | Notes |
|-------|------|--------|-------|
| Discord input | 48kHz stereo | s16le PCM | 20ms frames (3840 bytes) |
| Processing | 16kHz mono | float32 | Wake word, VAD, STT |
| TTS output | 24kHz mono | float32 | Inworld native |
| Discord output | 48kHz stereo | s16le PCM | Upsampled for playback |

### Resampling Performance

The hot path uses `resample_poly` for integer ratios:

```python
# 48kHz -> 16kHz (3:1 ratio) - HOT PATH
resampled = signal.resample_poly(mono, 1, 3)  # 10-50x faster than FFT
```

**Why this matters**: FFT-based resampling holds the GIL during computation. When the user talks during bot playback, `resample_poly` reduces GIL contention that causes audio pops.

### Pre-Buffer System

Maintains 1.5 seconds of audio per user while idle:

```python
# Per-user pre-buffers capture question start
self._per_user_pre_buffers: dict[int, list[np.ndarray]] = {}
self._pre_buffer_max_samples = int(16000 * 1.5)  # 1.5s at 16kHz
```

When wake word detected, ONLY the wake word speaker's buffer is used:

```python
# Include ONLY the wake word speaker's pre-buffer (speaker isolation)
if self._last_user_id and self._last_user_id in self._per_user_pre_buffers:
    pre_buffer = self._per_user_pre_buffers[self._last_user_id]
    self._audio_buffer = pre_buffer.copy()
```

---

## Critical Timing Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `PTT timeout` | 500ms | voice_sink.py:88 | Detect push-to-talk release |
| `Silence debounce` | 3000ms | config.py:69 | Channel silence for responses |
| `VAD min silence` | 500ms | config.py:62 | Detect speech endpoint |
| `Pre-buffer duration` | 1.5s | voice_sink.py:100 | Capture question start |
| `Sentence gap` | 150ms | voice_session.py:730 | Natural pacing between TTS |
| `VAD speech pad` | 300ms | config.py:61 | Prevent clipping word edges |
| `Max question` | 30s | config.py:63 | Cap runaway recording |
| `User silence debounce` | 300ms | speaking_detector.py:208 | Per-user brief pause tolerance |
| `Stale threshold` | 500ms | speaking_detector.py:65 | PTT release detection |

### Timing Trade-offs

**PTT timeout (500ms)**:
- Too low: Network jitter causes false positives
- Too high: Users wait unnecessarily after releasing PTT
- Current: Balances reliability with responsiveness

**Silence debounce (3000ms)**:
- Purpose: Allow natural conversation pauses before declaring channel silent
- Used for: Waiting to start bot response
- NOT used for: Acknowledgment timing (uses immediate check instead)

**Pre-buffer (1.5s)**:
- Purpose: Capture question start in continuous speech
- Trade-off: Long enough for context, short enough to avoid stale audio
- Cleared on gaps >1.0s to prevent stale audio

---

## Key Components

### VoiceSession (src/bot/voice_session.py)

**State Machine**: `IDLE -> LISTENING -> PROCESSING -> IDLE`

**Key Responsibilities**:
- Main pipeline orchestrator
- Parallel TTS with ordered playback queue
- Dynamic timeout extension while user speaks
- Personalized thinking phrases

**Critical Methods**:
- `_run_loop()`: Main state machine loop (line 373)
- `_process_speech()`: STT -> LLM -> TTS pipeline (line 873)
- `_playback_consumer()`: Ordered audio queue player (line 732)
- `_synthesize_and_queue()`: Parallel TTS task spawner (line 805)

### StreamingSink (src/bot/voice_sink.py)

**Hot Path**: Processes audio 50x/second in Discord's thread.

**Key Responsibilities**:
- Real-time audio processing
- Per-user pre-buffer management
- Wake word detection coordination
- Speaker isolation filtering

**Critical Methods**:
- `write()`: Main audio processing entry point (line 307)
- `check_audio_timeout()`: PTT release detection (line 183)
- `check_idle_audio_gap()`: Idle wake word completion (line 241)
- `start_listening()`: Transition with pre-buffer handling (line 135)

### SpeakingDetector (src/detection/speaking_detector.py)

**Hybrid Detection**: Packet tracking + energy gate + VAD confirmation

**Key Innovations**:
- Staleness detection handles PTT release
- Dual interfaces: immediate vs debounced

**Critical Methods**:
- `process_audio()`: Per-user audio processing (line 181)
- `is_anyone_actively_speaking()`: Immediate check, no debounce (line 302)
- `get_state()`: Debounced channel state (line 285)
- `_is_user_stale()`: 500ms no-audio check (line 140)

---

## Code Patterns

### Pattern 1: Pre-Buffer Accumulation

Captures continuous speech like "hey jarvis what's the weather?":

```python
# voice_sink.py - Maintain rolling pre-buffer while idle
if self._state == "idle":
    if user_id not in self._per_user_pre_buffers:
        self._per_user_pre_buffers[user_id] = []

    self._per_user_pre_buffers[user_id].append(audio_16k.copy())

    # Trim to max duration (1.5s)
    user_buffer = self._per_user_pre_buffers[user_id]
    total = sum(len(a) for a in user_buffer)
    while total > self._pre_buffer_max_samples and user_buffer:
        removed = user_buffer.pop(0)
        total -= len(removed)
```

### Pattern 2: PTT Staleness Detection

Responds quickly after push-to-talk release:

```python
# speaking_detector.py - Check if user data is stale
STALE_THRESHOLD_S = 0.5  # 500ms

def _is_user_stale(self, user: UserAudioState, now: float) -> bool:
    """If no audio for 500ms, user is not speaking (PTT released)."""
    if user.last_packet_time is None:
        return True
    return (now - user.last_packet_time) >= self.STALE_THRESHOLD_S

# Only count users as speaking if NOT stale
any_speaking = any(
    user.is_speaking and not self._is_user_stale(user, now)
    for user in self._user_states.values()
)
```

### Pattern 3: Parallel TTS Streaming

LLM streams while TTS synthesizes previous sentences:

```python
# voice_session.py - Producer: spawn TTS tasks in parallel
async for chunk in self.llm.chat_stream(prompt_text):
    buffer += chunk

    # Queue complete sentences
    while '\n' in buffer:
        line, buffer = buffer.split('\n', 1)
        if line.strip():
            sentences_queued += 1
            # Spawn parallel TTS task (don't await)
            task = asyncio.create_task(
                self._synthesize_and_queue(line.strip(), sentences_queued)
            )
            self._tts_tasks.append(task)

# Consumer: play audio in order
async def _playback_consumer(self):
    next_to_play = 1
    pending: dict[int, np.ndarray] = {}

    while True:
        item = await self._audio_queue.get()
        if item is None:
            break

        sentence_num, audio = item
        pending[sentence_num] = audio

        # Play consecutive available sentences
        while next_to_play in pending:
            audio = pending.pop(next_to_play)
            await self._play_audio(audio)
            next_to_play += 1
```

### Pattern 4: Thread-Safe Callbacks

Communication between Discord thread and asyncio:

```python
# voice_session.py - Callback from sink thread
def _on_wake_word(self):
    """Callback when wake word detected (from sink thread)."""
    if self._state == "idle" and self._loop:
        # Thread-safe event setting
        self._loop.call_soon_threadsafe(self._wake_event.set)

def _on_speech_end(self, audio: np.ndarray, user_id: Optional[int] = None):
    """Callback when speech ends (from sink thread)."""
    if self._state == "listening" and self._loop:
        self._pending_audio = audio
        self._loop.call_soon_threadsafe(self._speech_end_event.set)
```

### Pattern 5: Speaker Isolation

Filter audio from non-wake-word speakers:

```python
# voice_sink.py - Only buffer audio from wake word speaker
if self._state == "listening":
    if self._speaker_isolation_enabled and self._last_user_id is not None:
        if user_id != self._last_user_id:
            # Log once per user per session
            if user_id not in self._filtered_users:
                logger.info(f"Filtering audio from user {user_id}")
                self._filtered_users.add(user_id)
            return  # Don't buffer this audio
```

### Pattern 6: Tool Use Loop with Search Limiting

Prevent infinite search loops:

```python
# groq_client.py - Limit search calls
MAX_SEARCH_CALLS = 3
SEARCH_TOOLS = {"web_search", "fetch_url"}
ACTION_TOOLS = {"generate_image", "edit_image", "tag_user"}

for turn in range(self.max_turns):
    search_limit_reached = self._search_calls_this_request >= MAX_SEARCH_CALLS

    if search_limit_reached:
        # Remove search tools, keep action tools only
        available_tools = [t for t in TOOLS if t["function"]["name"] in ACTION_TOOLS]

        # Force model to answer with gathered info
        messages.append({
            "role": "system",
            "content": "You have gathered enough search information. Answer NOW."
        })

    # Track search tools toward limit
    for tool_call in response_message.tool_calls:
        tool_name = tool_call.function.name
        if tool_name in SEARCH_TOOLS:
            self._search_calls_this_request += 1
```

---

## LLM Integration

### Providers

| Provider | Model | Context | Speed | Tools |
|----------|-------|---------|-------|-------|
| Claude | haiku/sonnet/opus | 200K | Moderate | Native (SDK) |
| Llama | llama-3.3-70b | ~8K | ~750 tok/s | 5 custom |
| GPT-OSS | gpt-oss-120b | 131K | ~500 tok/s | 5 custom |

### Tools (Groq Only)

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo + auto-fetch top result via Firecrawl |
| `fetch_url` | Manual URL fetch via Firecrawl |
| `generate_image` | Gemini 2.5 Flash -> Discord channel |
| `edit_image` | Multi-turn editing (persistent Gemini session) |
| `tag_user` | Discord mentions with fuzzy name matching |

### Streaming Architecture

```
LLM Response     TTS Synthesis          Playback Queue
    |                 |                      |
  chunk1 -------> [TTS Task 1] ---------> (1, audio1)
  chunk2 -------> [TTS Task 2] ---------> (2, audio2)
  chunk3 -------> [TTS Task 3] ---------> (3, audio3)
    |                 |                      |
   (parallel)    (parallel)           (ordered playback)
```

First sentence waits for:
1. Thinking phrase to complete
2. Channel silence (if speaking detection enabled)

---

## Configuration Reference

### Required

```bash
DISCORD_TOKEN=your_token
```

### LLM Provider

```bash
LLM_PROVIDER=claude|llama|gpt-oss  # Default: claude
CLAUDE_MODEL=haiku|sonnet|opus     # Default: haiku
GROQ_API_KEY=your_key              # Required for STT + Groq LLM
```

### Wake Word

```bash
WAKE_WORD_MODEL=hey_jarvis         # OpenWakeWord model
WAKE_WORD_CONFIDENCE=0.5           # Detection threshold
```

### Timing Tuning

```bash
VAD_SENSITIVITY=0.5                # Silero VAD threshold
MIN_SILENCE_MS=500                 # Speech endpoint silence
SPEECH_PAD_MS=300                  # Padding around speech
MAX_QUESTION_DURATION=30.0         # Max recording duration
```

### Multi-User Features

```bash
SPEAKING_DETECTION_ENABLED=false   # Enable multi-user awareness
SPEAKER_ISOLATION_ENABLED=true     # Filter non-wake-word speakers
SPEAKING_SILENCE_DEBOUNCE_MS=3000  # Channel silence debounce
USER_NAMES='{"123456": "Name"}'    # Discord ID -> friendly name
```

### Search & Image

```bash
FIRECRAWL_URL=http://localhost:3002  # Self-hosted Firecrawl
SEARCH_MAX_RESULTS=3                 # DuckDuckGo results
NANOBANANA_API_KEY=your_key          # Gemini API key
IMAGE_CHANNEL_NAME=general           # Channel for images
```

### Auto-Join

```bash
AUTO_JOIN_VOICE=true               # Join voice on startup
AUTO_JOIN_CHANNEL=channel_name     # Channel to auto-join
```

---

## File Reference

| File | Lines | Responsibility |
|------|-------|----------------|
| [voice_session.py](../src/bot/voice_session.py) | 1078 | Main orchestrator, state machine, parallel TTS |
| [voice_sink.py](../src/bot/voice_sink.py) | 503 | Real-time audio, pre-buffers, speaker isolation |
| [speaking_detector.py](../src/detection/speaking_detector.py) | 337 | Multi-user awareness, staleness detection |
| [groq_client.py](../src/llm/groq_client.py) | 693 | Groq LLM, tool use loop, search limiting |
| [claude_client.py](../src/llm/claude_client.py) | ~200 | Claude Code SDK integration |
| [synthesizer.py](../src/tts/synthesizer.py) | ~200 | Inworld TTS streaming |
| [groq_whisper.py](../src/stt/groq_whisper.py) | ~100 | Groq Whisper STT |
| [speaker_manager.py](../src/bot/speaker_manager.py) | 122 | Name/activity resolution |
| [discord_service.py](../src/bot/discord_service.py) | 157 | Image posting, user tagging |
| [resampler.py](../src/audio/resampler.py) | 106 | Sample rate conversion |
| [config.py](../src/config.py) | 115 | Pydantic settings |
| [main.py](../src/main.py) | 267 | Entry point, component preloading |

---

## Design Decisions

### Why Pre-Buffering?

**Problem**: Users say "hey jarvis what's the weather?" without pausing after the wake word.

**Solution**: Maintain 1.5s rolling buffer per user. When wake word detected, include the speaker's buffer. This captures the question start.

**Trade-off**: Memory usage for multiple users vs natural interaction.

### Why Staleness Detection?

**Problem**: PTT release stops audio packets abruptly. VAD needs silence to detect speech end, but no more audio arrives.

**Solution**: Track last packet time per user. If no audio for 500ms, mark user as not speaking regardless of VAD state.

**Impact**: Response starts 500ms after PTT release instead of waiting for VAD timeout (3.5s).

### Why Parallel TTS Streaming?

**Problem**: Sequential TTS (wait for full response, then synthesize, then play) adds latency.

**Solution**: Spawn TTS tasks for each sentence as it arrives from LLM. Use ordered queue for seamless playback.

**Impact**: First audio plays ~2-3s faster than sequential approach.

### Why resample_poly?

**Problem**: FFT-based resampling holds GIL during computation. When user talks during playback, this causes audio pops.

**Solution**: Use `resample_poly` for integer ratios (3:1 for 48kHz->16kHz). It's 10-50x faster and releases GIL more frequently.

**Trade-off**: Only works for integer ratios (covered by Discord's standard rates).

### Why Speaker Isolation?

**Problem**: Multi-user channels have cross-talk that corrupts transcription.

**Solution**: Lock to wake word speaker and filter audio from other users during listening.

**Trade-off**: Only one user can be transcribed at a time.

### Why Dual Debounce Modes?

**Problem**: 3s debounce is good for "wait until conversation ends" but bad for "should I play ack now?"

**Solution**: Two interfaces:
- `is_anyone_speaking()`: 3s debounce for waiting on responses
- `is_anyone_actively_speaking()`: Immediate check for ack timing

### Why AudioPipeline Not Integrated?

**Problem**: Created clean STT->LLM->TTS pipeline abstraction but couldn't integrate.

**Reason**: VoiceSession has deep state coupling:
- Thinking phrase playback coordination
- Silence detection for playback timing
- Playback task management
- 5+ callbacks would be needed

**Decision**: Keep AudioPipeline as standalone utility for future use or simpler pipelines.

---

## Recent Changes (2026-01-25)

| Change | Impact |
|--------|--------|
| Extracted SpeakerManager | Cleaner speaker identification |
| Extracted DiscordService | Separated Discord operations |
| Created AudioPipeline | Standalone utility for future use |
| Added sample rate constants | No more magic numbers |
| Removed unused STT providers | Simpler codebase (Groq-only) |
| Added 84 tests | Confidence for future changes |

---

## Troubleshooting

### Audio pops during playback
- Cause: FFT resampling holds GIL
- Fix: Using `resample_poly` now (already fixed)

### Slow response after PTT release
- Cause: VAD waiting for silence that never comes
- Check: `SPEAKING_DETECTION_ENABLED=true` for staleness detection

### Wake word not detected
- Check: `WAKE_WORD_CONFIDENCE` (lower = more sensitive)
- Check: Audio actually reaching sink (look for "SINK STATS" logs)

### Search loops forever
- Cause: Model keeps calling search tools
- Fix: Limit enforced (3 search calls max)

### Image not posting
- Check: `NANOBANANA_API_KEY` configured
- Check: `IMAGE_CHANNEL_NAME` matches a text channel

---

*Generated 2026-01-25. See [architecture-audit.md](./architecture-audit.md) for refactoring history.*
