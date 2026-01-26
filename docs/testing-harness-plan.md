# AI-Graded E2E Voice Bot Testing Harness

## Overview

A comprehensive end-to-end testing harness for the Discord voice bot that uses AI-based grading (Claude) to evaluate scenario outcomes against natural language criteria. Supports three testing levels with increasing realism and cost.

## Key Design Decisions

1. **AI Grading over Assertions** - Natural language criteria evaluated by Claude, not brittle string matching
2. **Multi-Level Testing** - Level 1 (text injection), Level 2 (single speaker audio), Level 3 (multi-speaker)
3. **YAML Scenario DSL** - Declarative test scenarios with timing, actions, and criteria
4. **Injectable Components** - VoiceSession already accepts optional components, perfect for mocking
5. **Event Capture via Hooks** - Hook into callbacks (on_wake_word, on_speech_end) and state changes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Scenario YAML                            │
│  - actions: speak, wait, wait_for_state, inject_text        │
│  - criteria: natural language success conditions            │
│  - speakers: multi-user definitions with voice settings     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TestSession                              │
│  - Loads scenario                                           │
│  - Sets up mock Discord + real/mock components              │
│  - Executes actions with timing                             │
│  - Captures all outputs                                     │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ Audio Gen  │  │ Mock       │  │ Output     │
     │ (TTS)      │  │ Discord    │  │ Capture    │
     └────────────┘  └────────────┘  └────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                   ┌────────────────┐
                   │  VoiceSession  │
                   │  (real impl)   │
                   └────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Grader (Claude)                       │
│  - Receives: scenario, bot response, logs, tool calls       │
│  - Evaluates each criterion with reasoning                  │
│  - Returns structured pass/fail with suggestions            │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
tests/
  e2e/
    __init__.py
    conftest.py                     # pytest fixtures
    harness/
      __init__.py
      scenario_loader.py            # YAML DSL parser (Pydantic)
      audio_generator.py            # TTS-based user speech
      audio_mixer.py                # Multi-speaker mixing
      mock_discord.py               # MockVoiceClient
      test_session.py               # Orchestrator
      output_capture.py             # Event/log capture
      grader.py                     # Claude AI grading
    scenarios/
      basic/
        wake_word_only.yaml
        simple_question.yaml
        barge_in.yaml               # Wake word + immediate question
        timeout.yaml                # Wake word, no follow-up
      tools/
        web_search.yaml
        image_generation.yaml
        deep_research.yaml
      multi_speaker/
        speaker_isolation.yaml
        interruption.yaml
    reports/                        # Test outputs (gitignored)
```

---

## Scenario YAML DSL

### Schema

```yaml
metadata:
  name: string                      # Unique scenario name
  description: string               # What this tests
  level: 1|2|3                      # 1=text, 2=audio, 3=multi-speaker
  tags: [string]                    # For filtering (e.g., "slow", "tools")
  skip: bool                        # Skip this scenario
  skip_reason: string               # Why skipped

speakers:                           # Multi-speaker definitions
  - id: int                         # Discord user ID
    name: string                    # Display name
    voice: string                   # TTS voice for generating speech

context: string                     # Optional LLM context to set

actions:                            # Sequence of test actions
  - action: speak|wait|wait_for_state|wait_for_speech|inject_text
    speaker: string                 # Speaker name (for speak/inject_text)
    text: string                    # What to say
    duration: float                 # Seconds (for wait)
    state: idle|listening|processing  # Target state
    timeout: float                  # Action timeout (default 30s)

criteria:                           # Natural language success criteria
  - description: string             # What to evaluate
    weight: float                   # 0.0-1.0 importance
    required: bool                  # Fail test if not met

expectations:                       # Quick validation (before AI grading)
  response_contains: [string]
  response_not_contains: [string]
  tools_called:
    - name: string
      args_contain: {key: value}
  max_latency_ms: int
```

### Example: Basic Question

```yaml
metadata:
  name: simple_factual_question
  description: User asks a basic factual question after wake word
  level: 2
  tags: [basic, happy-path]

speakers:
  - id: 12345
    name: TestUser
    voice: alex

actions:
  - action: speak
    speaker: TestUser
    text: "Hey Jarvis, what is the capital of France?"

  - action: wait_for_state
    state: idle
    timeout: 30.0

criteria:
  - description: "Bot correctly answers Paris"
    weight: 1.0
    required: true

  - description: "Response is concise (under 3 sentences)"
    weight: 0.5
    required: false

  - description: "No errors or confusion in the response"
    weight: 0.7
    required: true

expectations:
  response_contains: ["Paris"]
  max_latency_ms: 5000
```

### Example: Barge-In (Wake Word + Immediate Question)

```yaml
metadata:
  name: barge_in
  description: User says wake word and question in one breath
  level: 2
  tags: [edge-case]

actions:
  - action: speak
    speaker: TestUser
    text: "Hey Jarvis what's two plus two"

  - action: wait_for_state
    state: idle
    timeout: 30.0

criteria:
  - description: "Bot understood both wake word and question together"
    required: true

  - description: "Bot answered correctly (4 or four)"
    required: true

  - description: "Single coherent response, not fragmented"
    required: true
```

### Example: Multi-Speaker Isolation

```yaml
metadata:
  name: speaker_isolation
  description: Bot only responds to wake word speaker, filters others
  level: 3
  tags: [multi-speaker]

speakers:
  - id: 12345
    name: Alice
    voice: ashley
  - id: 67890
    name: Bob
    voice: blake

actions:
  - action: speak
    speaker: Alice
    text: "Hey Jarvis, what time is it?"

  - action: speak
    speaker: Bob
    text: "Actually, tell me the weather instead"

  - action: wait_for_state
    state: idle
    timeout: 30.0

criteria:
  - description: "Bot responds to Alice's time question, not Bob's weather request"
    required: true

  - description: "Response contains time information, not weather"
    required: true

  - description: "Logs show Bob's speech was filtered out"
    required: false
```

### Example: Tool Use

```yaml
metadata:
  name: web_search_tool
  description: Question requiring web search
  level: 2
  tags: [tools]

context: "TestUser is currently Playing Dune: Awakening"

actions:
  - action: speak
    speaker: TestUser
    text: "Hey Jarvis, how do I craft heavy darts in Dune?"

  - action: wait_for_state
    state: idle
    timeout: 45.0

criteria:
  - description: "Bot uses web_search tool to find crafting info"
    required: true

  - description: "Response includes specific crafting requirements"
    required: true

  - description: "Bot infers game from user's activity context"
    required: false

expectations:
  tools_called:
    - name: web_search
```

---

## Key Components

### 1. MockVoiceClient (`mock_discord.py`)

```python
class MockVoiceClient:
    """Mock Discord VoiceClient for testing."""

    def inject_audio(self, audio: np.ndarray, user_id: int):
        """Inject audio as if from Discord.

        Converts 16kHz mono to Discord's 48kHz stereo format.
        """
        discord_bytes = resample_16k_mono_to_48k_stereo(audio)
        # Send in 20ms chunks (Discord packet size)
        for chunk in chunks(discord_bytes, 3840):
            self._sink.write(chunk, user_id)

    def get_played_audio(self) -> list[np.ndarray]:
        """Get all audio that was played back (bot responses)."""
```

### 2. AudioGenerator (`audio_generator.py`)

```python
class AudioGenerator:
    """Generate user speech using TTS (same as bot uses)."""

    async def generate(self, text: str, speaker_id: int, voice: str = "alex") -> GeneratedAudio:
        """Generate audio for text, with caching."""
        # Uses Inworld TTS, resamples 24kHz -> 16kHz
```

### 3. OutputCapture (`output_capture.py`)

Captures everything during a test run:
- State changes (idle -> listening -> processing -> idle)
- Wake word detection events
- STT transcriptions
- LLM response chunks
- Tool calls with arguments and results
- TTS text before synthesis
- Errors and exceptions
- Timing of all events

### 4. AIGrader (`grader.py`)

```python
class AIGrader:
    """Grade scenarios using Claude."""

    async def grade(self, scenario: Scenario, capture: TestCapture) -> GradeResult:
        """Send all captured data to Claude for evaluation."""

        prompt = f"""You are evaluating a Discord voice bot test.

## Scenario: {scenario.metadata.name}
{scenario.metadata.description}

## What Happened
Bot Response: {capture.full_response}
Tool Calls: {format_tool_calls(capture.tool_calls)}
Event Timeline: {format_events(capture.events)}
Errors: {capture.errors}

## Criteria to Evaluate
{format_criteria(scenario.criteria)}

Evaluate each criterion. Return JSON with pass/fail and reasoning."""
```

### 5. TestSession (`test_session.py`)

Orchestrator that:
1. Sets up MockVoiceClient and OutputCapture
2. Creates VoiceSession with real or mocked components
3. Installs capture hooks
4. Executes scenario actions with timing
5. Waits for completion
6. Runs quick validation (expectations)
7. Calls AI grader
8. Returns structured results

---

## Testing Levels

| Level | What | Speed | Cost | Realism |
|-------|------|-------|------|---------|
| 1 | Text injection (bypass STT) | Fast | Low | Low |
| 2 | Full audio, single speaker | Medium | Medium | High |
| 3 | Multi-speaker audio mixing | Slow | High | Highest |

### Level 1: Text Injection

- Directly inject transcribed text into VoiceSession
- Bypass audio generation and STT
- Test LLM responses and tool use
- Good for: Fast iteration, CI, testing conversation logic

### Level 2: Single Speaker Audio

- Generate user speech via TTS
- Inject into StreamingSink
- Full pipeline: wake word -> VAD -> STT -> LLM -> TTS
- Good for: Integration testing, wake word edge cases

### Level 3: Multi-Speaker

- Generate multiple audio streams
- Mix with timing offsets
- Tag per-user for speaker isolation testing
- Good for: Speaker isolation, interruption handling

---

## pytest Integration

### conftest.py Fixtures

```python
@pytest.fixture
def test_session():
    """Fresh test session for each test."""
    return TestSession()

@pytest.fixture
def mock_session():
    """Session with mocked external services (fast, deterministic)."""
    return TestSession(mock_llm=True, mock_tts=True, mock_stt=True)
```

### YAML Auto-Collection

```python
def pytest_collect_file(parent, file_path):
    """Collect YAML scenario files as test items."""
    if file_path.suffix == ".yaml" and "scenarios" in str(file_path):
        return YamlScenarioFile.from_parent(parent, path=file_path)
```

### Running Tests

```bash
# All E2E tests
pytest tests/e2e/

# Only Level 1 (fast, text injection)
pytest tests/e2e/ -k "level1"

# Only basic scenarios
pytest tests/e2e/scenarios/basic/

# With detailed output
pytest tests/e2e/ -v --tb=long
```

---

## Integration Points

| Component | File | Hook Point |
|-----------|------|------------|
| Audio injection | `voice_sink.py:22` | `StreamingSink.write()` |
| State changes | `voice_session.py:31` | `VoiceSession` state property |
| Wake word | `voice_sink.py:41` | `on_wake_word` callback |
| Speech end | `voice_sink.py:42` | `on_speech_end` callback |
| STT | `voice_session.py:83` | `groq_client` parameter |
| LLM | `voice_session.py:84` | `llm` parameter |
| TTS | `voice_session.py:85` | `tts` parameter |
| Tool calls | `groq_client.py` | `_execute_tool()` method |

---

## AI Grading Prompt Template

```
You are evaluating test run #{run_id} for a Discord voice bot.

## Scenario
Name: {name}
Description: {description}

## What Happened

### User Said (transcribed):
{user_transcripts}

### Bot Responded:
{bot_response}

### Tool Calls:
{tool_calls_formatted}

### Event Timeline:
{events_formatted}

### Errors:
{errors or "None"}

### Timing:
- Total duration: {duration_ms}ms

## Success Criteria
{criteria_numbered}

## Your Task

1. Evaluate each criterion as PASS or FAIL with specific reasoning
2. Reference specific quotes, log lines, or timestamps
3. Flag unexpected issues not in criteria
4. Rate overall: PASS (all required criteria met) or FAIL

Respond in JSON:
{
  "criteria_results": [
    {"description": "...", "passed": bool, "reasoning": "..."}
  ],
  "overall_passed": bool,
  "overall_reasoning": "...",
  "unexpected_issues": [...],
  "suggestions": [...]
}
```

---

## Cost Considerations

### Per-Test Costs

| Component | Cost |
|-----------|------|
| TTS (input gen) | ~$0.001 per utterance |
| STT | Free (Groq) |
| LLM (bot response) | ~$0.002-0.01 depending on provider |
| AI Grading (Claude Sonnet) | ~$0.01 per scenario |

### Cost Reduction Strategies

1. **Audio Caching** - Cache generated audio by text+voice key
2. **Batch Grading** - Grade 5-10 scenarios in one Claude call
3. **Quick Validation** - Fail fast on `expectations` before AI grading
4. **Level Filtering** - Run Level 1 in CI, Level 2-3 locally
5. **Grade Caching** - Cache grades for unchanged scenarios (hash DSL + code)

### CI Budget

```yaml
env:
  E2E_BUDGET_DOLLARS: 1.00  # Max per run
  E2E_LEVEL: 1              # Text injection only in CI
```

---

## Phased Implementation

### Phase 1: Foundation (Core Infrastructure)

**Files to create:**
- `tests/e2e/__init__.py`
- `tests/e2e/harness/__init__.py`
- `tests/e2e/harness/scenario_loader.py` - Pydantic models, YAML loading
- `tests/e2e/harness/output_capture.py` - Event capture infrastructure
- `tests/e2e/harness/mock_discord.py` - MockVoiceClient basics

**Deliverable:** Can load scenarios and capture events manually

### Phase 2: Audio Pipeline

**Files to create:**
- `tests/e2e/harness/audio_generator.py` - TTS-based speech generation
- `tests/e2e/harness/audio_mixer.py` - Multi-track mixing
- `tests/e2e/harness/test_session.py` - Orchestrator

**Deliverable:** Can run single-speaker audio tests

### Phase 3: AI Grading + pytest

**Files to create:**
- `tests/e2e/harness/grader.py` - Claude integration
- `tests/e2e/conftest.py` - Fixtures and YAML collection
- `tests/e2e/scenarios/basic/*.yaml` - 5 basic scenarios
- `tests/e2e/scenarios/tools/*.yaml` - 3 tool scenarios

**Deliverable:** Full E2E suite with AI grading

### Phase 4: Polish + CI

**Tasks:**
- Batch grading implementation
- Caching (audio + grades)
- GitHub Actions workflow
- Documentation
- 2 multi-speaker scenarios

**Deliverable:** Production-ready CI integration

---

## Verification Plan

1. **Phase 1 Verification:**
   - Load a YAML scenario, verify Pydantic validation works
   - Create MockVoiceClient, verify audio format conversion
   - Capture events from a manual VoiceSession run

2. **Phase 2 Verification:**
   - Generate audio with AudioGenerator, verify it's valid 16kHz mono
   - Inject audio into MockVoiceClient, verify StreamingSink receives it
   - Run a simple scenario end-to-end, verify state transitions

3. **Phase 3 Verification:**
   - Run AI grader on captured data, verify JSON response parsing
   - Run `pytest tests/e2e/scenarios/basic/` - all should pass
   - Run tool scenario, verify tool call capture

4. **Phase 4 Verification:**
   - Run full suite with `--cov`, verify coverage
   - Run in CI, verify budget limits work
   - Time a full run, verify caching reduces time on second run

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
e2e = [
    "pyyaml>=6.0",
    # scipy already present for resampling
]
```

---

## Example Test Scenarios to Implement

| Scenario | Level | Tests |
|----------|-------|-------|
| Wake word only | 2 | Acknowledgment phrase spoken |
| Simple question | 2 | Correct factual answer |
| Barge-in | 2 | Wake word + question in one breath |
| Timeout | 2 | Wake word, no follow-up -> graceful timeout |
| Web search | 2 | Tool invocation for current info |
| Image generation | 2 | generate_image tool called |
| Deep research | 2 | Complex query delegated |
| Speaker isolation | 3 | Only wake word speaker processed |
| Interruption | 3 | Handle mid-response interruption |
| Follow-up | 2 | Multi-turn conversation context |

---

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add `e2e` optional dependencies |
| `.gitignore` | Add `tests/e2e/reports/` |

## Files to Create

| File | Purpose |
|------|---------|
| `tests/e2e/__init__.py` | Package init |
| `tests/e2e/conftest.py` | pytest fixtures |
| `tests/e2e/harness/__init__.py` | Harness package |
| `tests/e2e/harness/scenario_loader.py` | YAML DSL parser |
| `tests/e2e/harness/audio_generator.py` | TTS speech generation |
| `tests/e2e/harness/audio_mixer.py` | Multi-speaker mixing |
| `tests/e2e/harness/mock_discord.py` | Mock VoiceClient |
| `tests/e2e/harness/test_session.py` | Test orchestrator |
| `tests/e2e/harness/output_capture.py` | Event capture |
| `tests/e2e/harness/grader.py` | AI grading |
| `tests/e2e/scenarios/basic/*.yaml` | Basic scenarios |
| `tests/e2e/scenarios/tools/*.yaml` | Tool scenarios |
| `tests/e2e/scenarios/multi_speaker/*.yaml` | Multi-speaker scenarios |
