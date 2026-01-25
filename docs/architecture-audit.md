# Architecture Audit: Separation of Concerns & Code Quality

## Handoff for Implementation

**Goal:** Refactor `VoiceSession` god object and simplify STT to Groq-only.

**Start here:**
1. Read the [Implementation Priority](#implementation-priority) table
2. Read the [Risk Analysis](#risk-analysis--implementation-notes) for each task
3. Read the [Testing Strategy](#testing-strategy) section
4. Execute tasks in the [Recommended Implementation Order](#recommended-implementation-order)

**Key decisions already made:**
- STT simplified to Groq Whisper only (delete Deepgram, local Whisper)
- STT abstraction layer cancelled (not needed with single provider)
- Speaker resolution: verified 3 duplicate blocks are identical - safe to extract
- AudioPipeline: do last, needs integration tests first

**Low-risk tasks (do first):** Priority 0, 1, 7
**Medium-risk tasks (do with tests):** Priority 4, 6, 8
**Cancelled:** Priority 3 (STT abstraction)

---

**Date:** 2025-01-24
**Updated:** 2026-01-24
**Overall Grade:** B+

The codebase is well-organized with clear module boundaries. The LLM layer is excellently abstracted. However, `VoiceSession` has grown into a god object.

> **Decision (2026-01-24):** STT is simplified to Groq Whisper only. Deepgram and local Whisper providers will be removed. This eliminates the need for STT abstraction layers.

---

## Table of Contents

1. [What's Done Well](#whats-done-well)
2. [Separation of Concerns Issues](#separation-of-concerns-issues)
3. [Missing Abstractions](#missing-abstractions)
4. [Code Quality Issues](#code-quality-issues)
5. [Recommended Refactoring](#recommended-refactoring)
6. [Dependency Graph](#dependency-graph)

---

## What's Done Well

### 1. LLM Abstraction (Excellent)

The LLM layer follows best practices:

- `src/llm/base.py` defines clean `BaseLLMClient` abstract interface
- `src/llm/llm_factory.py` uses factory pattern for provider selection
- Easy to add new providers by implementing the interface

```python
class BaseLLMClient(ABC):
    @abstractmethod
    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        pass
```

### 2. Configuration Management

`src/config.py` uses Pydantic settings effectively:

- Environment variable mapping with aliases
- Clear documentation via field names
- Sensible defaults for all settings
- Computed properties for derived values

### 3. Event-Driven Decoupling

Callback patterns keep components loosely coupled:

- `on_wake_word`, `on_speech_end` - sink to session communication
- `on_image_generated`, `on_user_tagged` - LLM to Discord communication

### 4. Module Organization

```
src/
├── bot/        # Discord integration (VoiceSession, VoiceSink)
├── llm/        # LLM clients with abstraction
├── stt/        # Speech-to-text providers
├── tts/        # Text-to-speech
├── detection/  # Wake word, VAD, speaking detection
├── search/     # Web search pipeline (for Groq)
├── image/      # Image generation (Nano Banana)
└── config.py   # Pydantic settings
```

---

## Separation of Concerns Issues

### 1. VoiceSession is a God Object

`src/bot/voice_session.py` (1,171 lines) handles too many responsibilities:

| Responsibility | Lines | Should Be |
|----------------|-------|-----------|
| Audio pipeline orchestration | 945-1126 | `AudioPipeline` class |
| Discord image posting | 260-325 | `DiscordService` class |
| Discord user tagging | 327-375 | `DiscordService` class |
| Speaker name resolution | 389-419 | `SpeakerManager` class |
| User ID to name mappings | 38-44 | Config file or database |
| TTS queue/playback | 857-943 | Part of `AudioPipeline` |
| Timing metrics | Throughout | Logging decorator or separate class |

### 2. Hardcoded User Mappings

`src/bot/voice_session.py:38-44`:

```python
USER_NAMES = {
    100000000000000001: "Darren",
    100000000000000003: "Pat",
    100000000000000002: "Sam",
    100000000000000004: "Jordan",
}
```

**Problems:**
- Hardcoded in source code
- Requires code change to add users
- Not configurable per-deployment

**Solution:** Move to `config.py` or a JSON/YAML data file.

### 3. Duplicate Speaker Resolution Logic

The same speaker lookup logic appears in 3 places:

1. `voice_session.py:621-643` (PTT path)
2. `voice_session.py:657-677` (timeout path)
3. `voice_session.py:430-449` (`_on_speech_end` callback)

This violates DRY and makes maintenance error-prone.

---

## Missing Abstractions

### ~~STT Providers Have No Common Interface~~ (RESOLVED)

> **Decision:** STT simplified to Groq Whisper only. No abstraction needed.

**Current approach in `_process_speech()` (to be simplified):**
```python
if self.groq_client is not None:
    text = await self.groq_client.transcribe_audio(audio)
elif self.deepgram_client is not None:      # REMOVE
    text = await self.deepgram_client.transcribe_audio(audio)
elif self.transcriber is not None:          # REMOVE
    text = await loop.run_in_executor(None, self.transcriber.transcribe, audio)
```

**Target (after cleanup):**
```python
text = await self.groq_client.transcribe_audio(audio)
```

**Files to delete:**
- `src/stt/deepgram_flux.py`
- `src/stt/transcriber.py`

**Files to update:**
- `src/stt/__init__.py` - remove exports
- `src/config.py` - remove `stt_provider` field (or make it ignored)
- `src/main.py` - remove whisper loading logic (lines 55-63)
- `src/bot/voice_session.py` - remove deepgram_client/transcriber params and fallback logic

### TTS Has Single Implementation

Currently only `TTSSynthesizer` exists. If additional TTS providers are added, a similar abstraction would be needed.

---

## Code Quality Issues

### Large Methods

| Method | Lines | Location | Recommendation |
|--------|-------|----------|----------------|
| `_process_speech()` | 180 | `voice_session.py:945-1126` | Split into `_transcribe()`, `_generate_response()`, `_log_timing()` |
| `chat_stream()` | 200 | `groq_client.py:191-398` | Extract tool execution into `_execute_tool_loop()` |
| `write()` | 190 | `voice_sink.py:307-494` | Extract idle/listening handlers |

### Magic Numbers

Sample rate `16000` appears ~20 times without a named constant:

```python
# voice_sink.py:100
self._pre_buffer_max_samples = int(16000 * 1.5)

# voice_session.py:956
audio_duration = len(audio) / 16000
```

**Recommendation:** Add constants in `config.py`:
```python
SAMPLE_RATE_16K = 16000
SAMPLE_RATE_24K = 24000
SAMPLE_RATE_48K = 48000
```

### Inline Timeouts and Thresholds

Various values scattered throughout code:
- `silence_timeout = 0.5` in voice_sink.py
- `max_wait=30.0` in voice_session.py
- `_user_log_cooldown = 5.0` in voice_sink.py

These should be configurable or at least defined as class constants.

---

## Recommended Refactoring

### Priority 1: Extract from VoiceSession

#### 1.1 Create `SpeakerManager`

```python
class SpeakerManager:
    """Manages speaker identification and personalization."""

    def __init__(self, user_names: dict[int, str]):
        self._user_names = user_names

    def resolve_speaker(self, user_id: int, guild: discord.Guild) -> SpeakerInfo:
        """Get speaker name and activity from user ID."""
        pass

    def find_user_by_name(self, name: str, guild: discord.Guild) -> tuple[int, str] | None:
        """Fuzzy match user by name."""
        pass
```

#### 1.2 Create `DiscordService`

```python
class DiscordService:
    """Handles Discord-specific operations."""

    def __init__(self, voice_client: discord.VoiceClient, settings: Settings):
        self._vc = voice_client
        self._settings = settings

    async def post_image(self, image_bytes: bytes, caption: str, mime_type: str) -> str:
        """Post image to configured channel."""
        pass

    async def tag_user(self, target_name: str, message: str) -> str:
        """Tag user in chat with message."""
        pass
```

#### 1.3 Create `AudioPipeline`

```python
class AudioPipeline:
    """Orchestrates STT, LLM, and TTS streaming."""

    def __init__(self, stt: BaseSTTProvider, llm: BaseLLMClient, tts: TTSSynthesizer):
        self._stt = stt
        self._llm = llm
        self._tts = tts

    async def process_speech(self, audio: np.ndarray, speaker: SpeakerInfo) -> None:
        """Full pipeline: transcribe -> LLM -> stream TTS."""
        pass
```

### ~~Priority 2: Add Missing Abstractions~~ (CANCELLED)

> **Decision:** With STT simplified to Groq-only, no abstraction layer is needed.
> Use `GroqWhisperClient` directly.

~~#### 2.1 Create `BaseSTTProvider`~~ - Not needed
~~#### 2.2 Create STT Factory~~ - Not needed

### Priority 3: Code Quality Improvements

#### 3.1 Extract Duplicate Speaker Resolution

```python
def _resolve_current_speaker(self, user_id: int) -> None:
    """Set current speaker info from user ID."""
    self._current_speaker_id = user_id
    self._current_speaker_activity = None

    if user_id and user_id in self.USER_NAMES:
        self._current_speaker_name = self.USER_NAMES[user_id]
    elif user_id and self.voice_client:
        try:
            member = self.voice_client.guild.get_member(user_id)
            self._current_speaker_name = member.display_name if member else None
        except Exception:
            self._current_speaker_name = None
    else:
        self._current_speaker_name = None

    # Get activity if we have voice client
    if user_id and self.voice_client:
        try:
            member = self.voice_client.guild.get_member(user_id)
            self._current_speaker_activity = self._get_member_activity(member)
        except Exception:
            pass
```

#### 3.2 Move User Names to Config

```python
# config.py
class Settings(BaseSettings):
    # ...
    user_names_file: Optional[str] = Field(default=None, alias="USER_NAMES_FILE")

    def load_user_names(self) -> dict[int, str]:
        if self.user_names_file and Path(self.user_names_file).exists():
            return json.loads(Path(self.user_names_file).read_text())
        return {}
```

#### 3.3 Add Sample Rate Constants

```python
# src/audio/constants.py
SAMPLE_RATE_DISCORD = 48000  # Discord native
SAMPLE_RATE_PROCESSING = 16000  # Wake word, VAD, STT
SAMPLE_RATE_TTS = 24000  # TTS output
```

---

## Dependency Graph

### Current Architecture

```
main.py
  └── DiscordBot
        └── VoiceSession (GOD OBJECT - 1,171 lines)
              ├── StreamingSink
              │     ├── WakeWordDetector
              │     ├── VAD
              │     └── SpeakingDetector
              ├── STT (3 implementations, no interface)  ← SIMPLIFY TO 1
              │     ├── GroqWhisperClient                ← KEEP
              │     ├── DeepgramFluxClient               ← DELETE
              │     └── Transcriber                      ← DELETE
              ├── BaseLLMClient (good abstraction)
              │     ├── ClaudeClient
              │     └── GroqClient
              │           ├── SearchService
              │           └── NanoBananaClient
              └── TTSSynthesizer
```

### Proposed Architecture (Revised)

```
main.py
  └── DiscordBot
        └── VoiceSession (slimmed down)
              ├── StreamingSink
              │     ├── WakeWordDetector
              │     ├── VAD
              │     └── SpeakingDetector
              ├── SpeakerManager (NEW - optional)
              ├── DiscordService (NEW - optional)
              ├── GroqWhisperClient (direct, no abstraction)
              ├── BaseLLMClient
              │     ├── ClaudeClient
              │     └── GroqClient
              └── TTSSynthesizer
```

> **Note:** `AudioPipeline` extraction is last priority (needs integration tests first).

---

## Implementation Priority

| Priority | Task | Effort | Impact | Risk | Notes |
|----------|------|--------|--------|------|-------|
| 0 | **Remove unused STT providers** | Low | Medium | 🟢 Low | Delete files, update imports |
| 1 | Extract `_resolve_current_speaker()` method | Low | Medium | 🟢 Low | Verified: 3 blocks are identical |
| 2 | Move `USER_NAMES` to config/data file | Low | Medium | 🟢 Low | Handle missing file gracefully |
| 3 | ~~Add `BaseSTTProvider` abstraction~~ | ~~Medium~~ | ~~High~~ | N/A | **Cancelled** - Groq-only now |
| 4 | Create `DiscordService` class | Medium | High | 🟡 Medium | Callback wiring is tricky |
| 5 | Create `SpeakerManager` class | Medium | Medium | 🟢 Low | Pure data lookups |
| 6 | Create `AudioPipeline` class | High | High | 🟡 Medium | Do last, needs integration tests first |
| 7 | Add sample rate constants | Low | Low | 🟢 Very Low | Just rename magic numbers |
| 8 | Split large methods | Medium | Medium | 🟡 Medium | State sharing between splits |

---

## Risk Analysis & Implementation Notes

### Priority 0: Remove Unused STT Providers
**Risk: 🟢 Low**

Deletion is safe - no other code depends on these:
- `src/stt/deepgram_flux.py` - standalone, not imported elsewhere
- `src/stt/transcriber.py` - standalone, not imported elsewhere

**Checklist:**
- [ ] Delete `src/stt/deepgram_flux.py`
- [ ] Delete `src/stt/transcriber.py`
- [ ] Update `src/stt/__init__.py` to only export `GroqWhisperClient`
- [ ] Remove `deepgram_client` and `transcriber` params from `VoiceSession.__init__`
- [ ] Remove fallback logic in `VoiceSession._process_speech()` (lines 1047-1055)
- [ ] Remove fallback logic in `VoiceSession._ensure_stt_client()` (lines 206-230)
- [ ] Remove whisper loading in `main.py` (lines 55-63)
- [ ] Remove/update `stt_provider` in `config.py`
- [ ] Update CLAUDE.md to remove deepgram/whisper references

### Priority 1: Extract `_resolve_current_speaker()`
**Risk: 🟢 Low**

**Verified:** All 3 blocks are character-for-character identical (excluding comments and logging).

**Implementation:**
```python
def _resolve_current_speaker(self, user_id: int) -> None:
    # Paste any of the 3 blocks here - they're identical
```

**Checklist:**
- [ ] Add method to `VoiceSession`
- [ ] Replace block at lines 626-646 with `self._resolve_current_speaker(self.sink._last_user_id)`
- [ ] Replace block at lines 661-680 with `self._resolve_current_speaker(self.sink._last_user_id)`
- [ ] Replace block at lines 430-450 with `self._resolve_current_speaker(user_id)`
- [ ] Keep logging at call sites (they differ)

### Priority 2: Move `USER_NAMES` to Config
**Risk: 🟢 Low**

**Breakage vectors:**
- File not found → default to empty dict
- Invalid JSON → fail fast with clear error

**Implementation:**
```python
# config.py
user_names: dict[int, str] = Field(default_factory=dict, alias="USER_NAMES")
# Or load from JSON file
```

**Note:** Pydantic can parse JSON from env var: `USER_NAMES='{"123": "Name"}'`

### Priority 4: Create `DiscordService`
**Risk: 🟡 Medium**

**Breakage vectors:**
1. **Callback wiring** - `on_image_generated` and `on_user_tagged` must be connected correctly
2. **voice_client lifecycle** - service must handle voice_client being None
3. **Channel lookup** - `_find_text_channel()` logic must transfer correctly
4. **Error handling** - must propagate errors back to caller

**Implementation notes:**
- Pass `voice_client` to methods, don't store (it changes)
- Return status strings for TTS feedback
- Test image posting and user tagging separately

### Priority 5: Create `SpeakerManager`
**Risk: 🟢 Low**

Pure data lookups, no side effects. Safe to extract.

**Breakage vectors:**
- Guild might be None early in session → handle gracefully

### Priority 6: Create `AudioPipeline`
**Risk: 🟡 Medium** (revised from High)

**Concerns:**
1. **Stateful streaming** - TTS chunks must arrive in order
2. **Interrupt handling** - user can interrupt mid-response
3. **Error recovery** - partial failures need cleanup

**Mitigations:**
- The code is already self-contained in `_process_speech()`
- Extraction is mostly moving code, not changing logic
- With integration tests, risk is manageable

**Prerequisites:**
- [ ] Integration test for full pipeline with mocks
- [ ] Test for interrupt handling
- [ ] Test for error recovery paths

**When to do it:** After Priorities 0-5 are done and tested. Not "deferred indefinitely" - just last in queue.

### Priority 7: Add Sample Rate Constants
**Risk: 🟢 Very Low**

Just find-and-replace. No logic changes.

**Checklist:**
- [ ] Add constants to `config.py` or new `src/audio/constants.py`
- [ ] Replace `16000` with `SAMPLE_RATE_16K` (or `SAMPLE_RATE_PROCESSING`)
- [ ] Replace `24000` with `SAMPLE_RATE_24K` (or `SAMPLE_RATE_TTS`)
- [ ] Replace `48000` with `SAMPLE_RATE_48K` (or `SAMPLE_RATE_DISCORD`)

### Priority 8: Split Large Methods
**Risk: 🟡 Medium**

**`_process_speech()` (180 lines):**
- Splits naturally into: `_transcribe()`, `_generate_response()`, `_log_timing()`
- **Watch out:** Local variables shared across phases (timings, audio buffer)

**`chat_stream()` in groq_client (200 lines):**
- Tool execution loop can be extracted
- **Watch out:** Streaming state, tool call accumulation

**`write()` in voice_sink (190 lines):**
- State handlers can be split
- **Watch out:** Buffer management, thread safety

---

## Conclusion

The codebase has a solid foundation with good patterns in place (LLM abstraction, factory pattern, callback-based decoupling). The main technical debt is in `VoiceSession`, which has accumulated too many responsibilities over time.

### Recommended Implementation Order

| Order | Task | Risk | Rationale |
|-------|------|------|-----------|
| 1 | Remove unused STT providers | 🟢 Low | Simplifies codebase, reduces maintenance |
| 2 | Extract `_resolve_current_speaker()` | 🟢 Low | Quick win, verified identical |
| 3 | Add sample rate constants | 🟢 Very Low | Trivial, improves readability |
| 4 | Move `USER_NAMES` to config | 🟢 Low | Configuration belongs in config |
| 5 | Create `SpeakerManager` | 🟢 Low | Clean extraction, no side effects |
| 6 | Create `DiscordService` | 🟡 Medium | Requires careful callback wiring |
| 7 | Split large methods | 🟡 Medium | Do incrementally with tests |
| 8 | Create `AudioPipeline` | 🟡 Medium | Do last, after integration tests |

### What NOT to Do

- **Don't create STT abstraction** - Groq-only means no polymorphism needed
- **Don't refactor without testing the change** - Voice is hard to test manually, be conservative

---

## Testing Strategy

Existing tests: `test_ring_buffer.py`, `test_state_machine.py` - good patterns to follow.

### Pre-Refactor: Capture Behavior

Before any changes, add tests that capture current behavior so refactoring can't silently break things.

#### 1. Speaker Resolution Tests (Priority 1)

```python
# tests/test_speaker_resolution.py
import pytest
from unittest.mock import Mock, MagicMock

class TestSpeakerResolution:
    """Tests for speaker name and activity resolution."""

    def test_resolve_known_user(self):
        """User in USER_NAMES dict returns mapped name."""
        # Setup: mock guild, mock member
        # Assert: returns "Darren" for known ID

    def test_resolve_unknown_user_with_guild(self):
        """Unknown user falls back to Discord display_name."""
        # Assert: returns member.display_name

    def test_resolve_unknown_user_no_guild(self):
        """No guild returns None for name."""
        # Assert: returns None

    def test_resolve_gets_activity(self):
        """Activity is fetched for all users."""
        # Assert: _current_speaker_activity is set

    def test_resolve_handles_member_not_found(self):
        """Gracefully handles guild.get_member returning None."""
        # Assert: no exception, name is None
```

**Why:** These tests lock in the current behavior. After extracting `_resolve_current_speaker()`, all tests should still pass.

#### 2. STT Fallback Tests (Priority 0)

```python
# tests/test_stt_selection.py
class TestSTTSelection:
    """Tests for STT provider selection - DELETE AFTER CLEANUP."""

    def test_groq_used_when_available(self):
        """Groq client is used when present."""

    def test_fallback_to_deepgram(self):
        """Falls back to Deepgram when Groq unavailable."""

    def test_fallback_to_whisper(self):
        """Falls back to Whisper when both unavailable."""
```

**Why:** Write these BEFORE removing providers. They document current behavior. Delete them after cleanup confirms Groq-only works.

#### 3. Config Parsing Tests (Priority 2)

```python
# tests/test_config.py
class TestUserNamesConfig:
    """Tests for USER_NAMES configuration."""

    def test_parse_json_from_env(self):
        """USER_NAMES env var parses JSON correctly."""
        # USER_NAMES='{"123": "Test"}' -> {123: "Test"}

    def test_empty_default(self):
        """Missing USER_NAMES defaults to empty dict."""

    def test_invalid_json_fails_fast(self):
        """Invalid JSON raises clear error on startup."""
```

### Test Fixtures for Mocking

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

@pytest.fixture
def mock_guild():
    """Mock Discord guild with members."""
    guild = Mock()
    member = Mock()
    member.display_name = "TestUser"
    member.activities = []
    guild.get_member = Mock(return_value=member)
    return guild

@pytest.fixture
def mock_voice_client(mock_guild):
    """Mock Discord voice client."""
    vc = Mock()
    vc.guild = mock_guild
    return vc

@pytest.fixture
def mock_groq_client():
    """Mock Groq Whisper client."""
    client = AsyncMock()
    client.transcribe_audio = AsyncMock(return_value="test transcription")
    return client

@pytest.fixture
def user_names():
    """Test user name mappings."""
    return {
        123456: "TestUser1",
        789012: "TestUser2",
    }
```

### Integration Tests (Post-Refactor)

After extracting classes, add integration tests:

```python
# tests/test_voice_session_integration.py
class TestVoiceSessionIntegration:
    """Integration tests for VoiceSession with mocked externals."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, mock_voice_client, mock_groq_client):
        """Test STT -> LLM -> TTS flow with mocks."""
        # This is the "golden path" test

    @pytest.mark.asyncio
    async def test_speaker_resolution_integration(self):
        """Test speaker resolution works end-to-end."""
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_speaker_resolution.py -v

# Run tests matching pattern
pytest tests/ -v -k "speaker"
```

### Test-Driven Refactoring Workflow

For each refactoring task:

1. **Write tests first** that capture current behavior
2. **Run tests** - all should pass
3. **Refactor** - extract method/class
4. **Run tests** - all should still pass
5. **Commit** - tests prove refactoring is safe

### Coverage Targets

| Component | Current | Target |
|-----------|---------|--------|
| `RingBuffer` | ✅ Tested | Maintain |
| `StateMachine` | ✅ Tested | Maintain |
| Speaker resolution | ❌ None | Add before Priority 1 |
| STT selection | ❌ None | Add before Priority 0 |
| Config parsing | ❌ None | Add before Priority 2 |
| Discord callbacks | ❌ None | Add before Priority 4 |

### Smoke Test Script

For manual verification after refactoring:

```bash
# tests/smoke_test.sh
#!/bin/bash
# Run after each major refactor

echo "1. Testing imports..."
python -c "from src.bot.voice_session import VoiceSession" || exit 1

echo "2. Testing config loading..."
python -c "from src.config import Settings; s = Settings()" || exit 1

echo "3. Testing STT client creation..."
python -c "from src.stt import GroqWhisperClient" || exit 1

echo "4. Running unit tests..."
pytest tests/ -v --tb=short || exit 1

echo "✅ All smoke tests passed"
```
