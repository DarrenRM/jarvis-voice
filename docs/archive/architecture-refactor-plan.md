# Architecture Refactoring Plan: Discord Voice Bot

**Created**: 2026-01-26
**Status**: Proposal
**Scope**: Major refactoring of voice_session.py, groq_client.py, voice_sink.py
**Related**: [refactor-edge-cases.md](refactor-edge-cases.md) - Critical edge cases to preserve

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Target Architecture](#target-architecture)
4. [Detailed Module Specifications](#detailed-module-specifications)
5. [Git Workflow & Branching Strategy](#git-workflow--branching-strategy)
6. [Refactoring Phases](#refactoring-phases)
7. [Migration Strategy](#migration-strategy)
8. [Testing Strategy](#testing-strategy)
9. [Risk Assessment](#risk-assessment)
10. [Success Criteria](#success-criteria)

---

## Executive Summary

### The Problem

The codebase has accumulated significant architectural debt across three "god classes":

| File | Lines | Responsibilities | Max Method |
|------|-------|-----------------|------------|
| `voice_session.py` | 1,212 | 16+ | 160 lines |
| `groq_client.py` | 997 | 10+ | 254 lines |
| `voice_sink.py` | 604 | 11+ | 212 lines |

**Total: 2,813 lines** that violate Single Responsibility Principle, making the codebase:
- Hard to understand (multiple concerns interleaved)
- Hard to modify (changes ripple unpredictably)
- Hard to test (requires full Discord connection)
- Hard to extend (new features require modifying core classes)

### The Solution

Decompose into **20-25 focused modules** averaging 50-150 lines each, organized around:
- **State Machine**: Explicit state transitions with hooks
- **Pipeline Pattern**: Composable STT → LLM → TTS stages
- **Tool Registry**: Pluggable tool handlers
- **Per-User Management**: Encapsulated user audio state

### Expected Outcomes

- **Maintainability**: Each module has single responsibility
- **Testability**: Unit tests for all components without Discord
- **Extensibility**: Add features without modifying core classes
- **Clarity**: Data flow is explicit and traceable

### Enabling Feature: Barge-in

After refactoring, the previously complex "barge-in" feature (allow user to interrupt bot mid-response) becomes trivial:

```python
# Before: Would require touching 6+ methods across voice_session.py
# After: Two method calls
if user_id == self._current_speaker_id:
    self.pipeline.cancel()
    self.state_machine.transition_to("listening")
```

---

## Current State Analysis

### File Inventory

```
src/
├── bot/
│   ├── voice_session.py     1,212 lines  ← GOD CLASS
│   ├── voice_sink.py          604 lines  ← TOO LARGE
│   ├── discord_bot.py         319 lines  ✓ Acceptable
│   ├── discord_service.py     197 lines  ✓ Good
│   ├── speaker_manager.py     121 lines  ✓ Good
│   ├── voice_source.py         89 lines  ✓ Good
│   └── __init__.py
├── llm/
│   ├── groq_client.py         997 lines  ← GOD CLASS
│   ├── claude_client.py       265 lines  ✓ Acceptable
│   ├── base.py                 67 lines  ✓ Good
│   ├── llm_factory.py          48 lines  ✓ Good
│   └── __init__.py
├── detection/
│   ├── speaking_detector.py   336 lines  ✓ Acceptable
│   ├── wake_word.py           303 lines  ✓ Acceptable
│   ├── vad.py                 220 lines  ✓ Good
│   └── __init__.py
├── tts/
│   ├── synthesizer.py         342 lines  ✓ Acceptable
│   ├── streaming.py           178 lines  ✓ Good
│   └── __init__.py
├── stt/
│   ├── groq_whisper.py        156 lines  ✓ Good
│   └── __init__.py
├── search/
│   ├── firecrawl.py           219 lines  ✓ Good
│   ├── duckduckgo.py          108 lines  ✓ Good
│   ├── search_service.py      148 lines  ✓ Good
│   └── __init__.py
├── image/
│   ├── nano_banana.py         180 lines  ✓ Good
│   └── __init__.py
├── audio/
│   ├── resampler.py            92 lines  ✓ Good
│   └── __init__.py
├── config.py                  168 lines  ✓ Good
└── main.py                    261 lines  ✓ Acceptable
```

### God Class #1: voice_session.py (1,212 lines)

#### Current Responsibilities (16+)

1. **State Machine Management**
   - States: idle, listening, processing
   - Transitions scattered across `_run_loop()`, `_process_speech()`, `_handle_wake_word_only()`
   - No validation of legal transitions

2. **Event Handling**
   - Wake word callback from sink thread
   - Speech end callback from sink thread
   - Thread-safe event signaling via `asyncio.Event`

3. **Audio Buffer Management**
   - `_pending_audio` buffer
   - Buffer clearing and retrieval
   - Duration tracking

4. **Transcription Orchestration**
   - Groq Whisper STT calls
   - Wake word stripping from transcription
   - Empty transcription handling

5. **LLM Request Orchestration**
   - Prompt building with speaker context
   - Streaming response handling
   - Sentence boundary detection

6. **Parallel TTS Pipeline**
   - Sentence-level parallel synthesis
   - `_audio_queue` with ordering
   - `_synthesize_and_queue()` tasks
   - `_playback_consumer()` background task

7. **Audio Playback**
   - `_play_audio()` with Discord voice client
   - Playback completion waiting
   - Inter-sentence pauses

8. **Thinking Phrase System**
   - Generic and personalized phrases
   - Background synthesis during LLM request
   - `_thinking_task` management

9. **Acknowledgment System**
   - Random ack phrase selection
   - Silence detection before playing
   - `_ack_played` tracking

10. **Follow-up Handling**
    - Wake-word-only detection
    - Follow-up question listening
    - `_handle_wake_word_only()` logic

11. **Speaker Identification**
    - `_current_speaker_id`, `_current_speaker_name`, `_current_speaker_activity`
    - Integration with `SpeakerManager`
    - Activity detection (Discord presence)

12. **Discord Operations**
    - Image posting via `DiscordService`
    - User tagging via `DiscordService`
    - Message posting via `DiscordService`

13. **Greeting System**
    - `greet_user()` on voice channel join
    - Cooldown tracking (`_greeting_cooldowns`)
    - Silence detection before greeting
    - LLM greeting generation

14. **Silence Detection**
    - `_wait_for_silence()` with debounce options
    - Speaking detector integration
    - Timeout handling

15. **Component Initialization**
    - `_init_components()` lazy loading
    - 7+ component lifecycles
    - Shared vs per-session components

16. **Session Lifecycle**
    - `start()` / `stop()` methods
    - Task cancellation on stop
    - Resource cleanup

#### Key Methods Analysis

| Method | Lines | Responsibilities |
|--------|-------|-----------------|
| `_run_loop()` | 130 | State machine, wake detection, timeout handling, error recovery |
| `_process_speech()` | 160 | Transcription, prompt building, LLM streaming, TTS pipeline |
| `_playback_consumer()` | 62 | Queue processing, thinking phrase wait, silence detection |
| `_handle_wake_word_only()` | 80 | Ack playing, follow-up listening, timeout handling |
| `greet_user()` | 58 | Cooldown check, silence wait, LLM greeting, TTS playback |
| `_wait_for_silence()` | 32 | Polling loop, debounce modes, timeout |
| `_play_ack()` | 36 | Silence wait, TTS synthesis, playback |
| `_play_thinking_phrase()` | 33 | TTS synthesis, timing tracking, cancellation |
| `_synthesize_and_queue()` | 28 | Streaming TTS, chunk collection, queue insertion |
| `_play_audio()` | 42 | Source creation, playback, completion wait |

#### Coupling Analysis

```
VoiceSession
    ├── StreamingSink (bidirectional - callbacks + state polling)
    ├── VAD (injected)
    ├── SpeakingDetector (injected)
    ├── GroqWhisperClient (injected)
    ├── BaseLLMClient (injected)
    ├── TTSSynthesizer (injected)
    ├── DiscordService (created)
    ├── SpeakerManager (created)
    ├── discord.VoiceClient (external)
    └── TTSAudioSource (created per playback)
```

**Problem**: Session acts as "god orchestrator" directly managing 10+ components.

---

### God Class #2: groq_client.py (997 lines)

#### Current Responsibilities (10+)

1. **LLM API Communication**
   - Groq API streaming calls
   - Model selection (llama-3.3-70b, gpt-oss-120b)
   - Temperature and token configuration

2. **Conversation History**
   - `_conversation_history` list
   - History truncation (last 10 exchanges)
   - System/user/assistant message formatting

3. **System Prompt Generation**
   - 108-line `_build_system_prompt()` method
   - Personality injection
   - User-specific roasts from config
   - Tool usage examples
   - Current date/time injection

4. **Tool Definition**
   - 8 tools defined inline (180+ lines of JSON schemas)
   - Tool parameter specifications
   - Tool descriptions

5. **Tool Call Detection**
   - JSON parsing from streaming response
   - `tool_calls` field extraction
   - Partial JSON handling

6. **Tool Execution**
   - 132-line `_execute_tool()` switch statement
   - 8 different tool handlers inline
   - Error handling per tool

7. **Search Tools**
   - `web_search` - DuckDuckGo via SearchService
   - `fetch_url` - Firecrawl via SearchService
   - Search limit enforcement (3 max)

8. **Image Tools**
   - `generate_image` - Gemini via NanoBananaClient
   - `edit_image` - Gemini chat context
   - Image extraction and Discord posting

9. **Discord Tools**
   - `tag_user` - User mention via callback
   - `post_to_chat` - Message posting via callback
   - `post_table` - Markdown table formatting

10. **Research Delegation**
    - `deep_research` - Delegates to Claude
    - 63-line delegation prompt builder
    - Result summarization

11. **Error Recovery**
    - Tool name normalization (hallucination fixes)
    - Parsing failure recovery
    - Graceful degradation on tool errors

12. **Response Formatting**
    - Tool result summarization
    - Table formatting for Discord
    - Image result messaging

#### Key Methods Analysis

| Method | Lines | Responsibilities |
|--------|-------|-----------------|
| `chat_stream()` | 254 | Tool loop, streaming, error recovery, history |
| `_execute_tool()` | 132 | Tool dispatch, 8 inline handlers |
| `_build_system_prompt()` | 108 | Personality, roasts, examples, tools |
| `_build_claude_delegation_prompt()` | 63 | Research delegation prompt |
| `_format_table()` | 42 | Discord markdown table |
| `_normalize_tool_name()` | 28 | Hallucination fixes |

#### Tool Definitions (inline, 180+ lines)

```python
TOOLS = [
    {"name": "web_search", ...},      # 25 lines
    {"name": "fetch_url", ...},       # 20 lines
    {"name": "generate_image", ...},  # 22 lines
    {"name": "edit_image", ...},      # 25 lines
    {"name": "tag_user", ...},        # 22 lines
    {"name": "post_to_chat", ...},    # 18 lines
    {"name": "post_table", ...},      # 28 lines
    {"name": "deep_research", ...},   # 20 lines
]
```

---

### God Class #3: voice_sink.py (604 lines)

#### Current Responsibilities (11+)

1. **Discord Audio Streaming**
   - `write()` method called by Discord SDK
   - 48kHz stereo input handling
   - Packet timing tracking

2. **Audio Resampling**
   - 48kHz → 16kHz conversion
   - Stereo → mono conversion
   - Normalization to float32

3. **Per-User Wake Word Detection**
   - Lazy detector creation per user
   - `_per_user_detectors` dictionary
   - Detector lifecycle management

4. **Per-User Audio Buffering**
   - `_per_user_pre_buffers` - Audio before wake word
   - Pre-buffer size limiting (3 seconds)
   - Buffer transfer on wake detection

5. **State Management**
   - `state` property (idle/listening)
   - `start_listening()` / `stop_listening()`
   - State-dependent audio routing

6. **VAD Integration**
   - Speech start/end detection
   - `_speech_started` flag
   - `is_speech_active()` method

7. **Speaking Detection**
   - Multi-user awareness via `SpeakingDetector`
   - `is_anyone_speaking()` with debounce
   - `is_anyone_actively_speaking()` immediate

8. **Speaker Isolation**
   - Filter audio to wake word user only
   - `_wake_word_user_id` tracking
   - Non-wake-user audio rejection

9. **Audio Timeout Detection**
   - `check_audio_timeout()` for PTT
   - `_last_audio_time` tracking
   - 500ms timeout threshold

10. **Wake Word Flush Logic**
    - `check_idle_audio_gap()` for idle detection
    - Silence padding to trigger detection
    - `_per_user_silence_attempts` tracking

11. **User Lifecycle**
    - `cleanup_user()` on disconnect
    - Detector cleanup
    - Buffer cleanup

#### Key Methods Analysis

| Method | Lines | Responsibilities |
|--------|-------|-----------------|
| `write()` | 212 | Audio routing, resampling, wake detection, VAD, buffering |
| `check_idle_audio_gap()` | 52 | Idle flush, silence padding, per-user attempts |
| `start_listening()` | 34 | State change, pre-buffer transfer, isolation |
| `_get_or_create_detector()` | 19 | Lazy detector creation |
| `get_audio_buffer()` | 12 | Buffer retrieval with copy |
| `cleanup_user()` | 15 | Detector and buffer cleanup |

#### Per-User State (scattered across 4 dicts)

```python
self._per_user_detectors: dict[int, WakeWordDetector]
self._per_user_pre_buffers: dict[int, list[np.ndarray]]
self._per_user_last_seen: dict[int, float]
self._per_user_silence_attempts: dict[int, int]
```

**Problem**: Per-user state scattered, making lifecycle management error-prone.

---

## Target Architecture

### Directory Structure

```
src/
├── core/
│   ├── __init__.py
│   ├── state_machine.py          # SessionStateMachine
│   ├── events.py                 # Event definitions
│   └── types.py                  # Shared type definitions
│
├── pipeline/
│   ├── __init__.py
│   ├── base.py                   # PipelineStage ABC
│   ├── voice_pipeline.py         # VoicePipeline orchestrator
│   ├── transcription_stage.py    # STT + wake word stripping
│   ├── response_stage.py         # LLM response generation
│   └── tts_stage.py              # Streaming TTS pipeline
│
├── audio/
│   ├── __init__.py
│   ├── buffer_manager.py         # AudioBufferManager
│   ├── playback_queue.py         # OrderedPlaybackQueue
│   ├── resampler.py              # (existing)
│   └── constants.py              # Sample rates, etc.
│
├── tools/
│   ├── __init__.py
│   ├── registry.py               # ToolRegistry
│   ├── base.py                   # ToolHandler ABC
│   ├── executor.py               # ToolExecutor (dispatch + errors)
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── search.py             # WebSearchHandler, FetchUrlHandler
│   │   ├── image.py              # GenerateImageHandler, EditImageHandler
│   │   ├── discord.py            # TagUserHandler, PostMessageHandler, PostTableHandler
│   │   └── research.py           # DeepResearchHandler
│   └── definitions.py            # Tool JSON schemas
│
├── llm/
│   ├── __init__.py
│   ├── base.py                   # (existing) BaseLLMClient
│   ├── claude_client.py          # (existing, minor changes)
│   ├── groq_client.py            # Slimmed: API calls only
│   ├── chat_orchestrator.py      # Tool loop + streaming coordination
│   ├── prompt_builder.py         # SystemPromptBuilder
│   ├── conversation.py           # ConversationHistory
│   └── llm_factory.py            # (existing)
│
├── session/
│   ├── __init__.py
│   ├── voice_session.py          # Slim orchestrator (~150 lines)
│   ├── speaker_tracker.py        # SpeakerTracker
│   ├── greeting_service.py       # GreetingService
│   └── phrases.py                # ACK_PHRASES, THINKING_PHRASES
│
├── sink/
│   ├── __init__.py
│   ├── streaming_sink.py         # Discord audio integration only
│   ├── user_audio_state.py       # UserAudioState dataclass
│   ├── per_user_manager.py       # PerUserAudioManager
│   └── wake_processor.py         # WakeWordProcessor
│
├── bot/
│   ├── __init__.py
│   ├── discord_bot.py            # (existing, no changes)
│   ├── discord_service.py        # (existing, no changes)
│   ├── voice_client_adapter.py   # VoiceClientAdapter interface
│   ├── speaker_manager.py        # (existing, no changes)
│   └── voice_source.py           # (existing, no changes)
│
├── detection/                    # (existing, no changes)
├── tts/                          # (existing, no changes)
├── stt/                          # (existing, no changes)
├── search/                       # (existing, no changes)
├── image/                        # (existing, no changes)
│
├── config.py                     # (existing, no changes)
└── main.py                       # (existing, minor changes)
```

### Module Count

| Category | Current | Target | Change |
|----------|---------|--------|--------|
| core/ | 0 | 3 | +3 |
| pipeline/ | 1 | 5 | +4 |
| audio/ | 1 | 4 | +3 |
| tools/ | 0 | 9 | +9 |
| llm/ | 4 | 7 | +3 |
| session/ | 0 | 4 | +4 |
| sink/ | 0 | 4 | +4 |
| bot/ | 6 | 7 | +1 |
| **Total New Modules** | - | - | **+31** |

### Line Count Projections

| Module | Est. Lines | Purpose |
|--------|------------|---------|
| **core/** | | |
| state_machine.py | 80-100 | State transitions, hooks, validation |
| events.py | 40-60 | WakeWordEvent, SpeechEndEvent, etc. |
| types.py | 30-50 | SpeakerInfo, AudioChunk, etc. |
| **pipeline/** | | |
| base.py | 30-50 | PipelineStage ABC |
| voice_pipeline.py | 100-120 | Stage composition, cancellation |
| transcription_stage.py | 60-80 | STT + wake word stripping |
| response_stage.py | 80-100 | LLM streaming coordination |
| tts_stage.py | 100-120 | Parallel TTS + queue |
| **audio/** | | |
| buffer_manager.py | 80-100 | Buffer lifecycle, VAD integration |
| playback_queue.py | 60-80 | Ordered playback, cancellation |
| **tools/** | | |
| registry.py | 50-70 | Registration, lookup |
| base.py | 30-40 | ToolHandler ABC |
| executor.py | 60-80 | Dispatch, error handling |
| handlers/search.py | 60-80 | Web search, URL fetch |
| handlers/image.py | 80-100 | Generate, edit |
| handlers/discord.py | 80-100 | Tag, post, table |
| handlers/research.py | 60-80 | Claude delegation |
| definitions.py | 100-120 | Tool JSON schemas |
| **llm/** | | |
| groq_client.py | 150-180 | API calls only (down from 997) |
| chat_orchestrator.py | 120-150 | Tool loop, streaming |
| prompt_builder.py | 80-100 | System prompt templates |
| conversation.py | 50-70 | History management |
| **session/** | | |
| voice_session.py | 120-150 | Orchestrator (down from 1,212) |
| speaker_tracker.py | 50-70 | Speaker state |
| greeting_service.py | 60-80 | Greeting logic |
| phrases.py | 60-80 | Phrase constants |
| **sink/** | | |
| streaming_sink.py | 100-120 | Discord audio only (down from 604) |
| user_audio_state.py | 40-60 | Per-user dataclass |
| per_user_manager.py | 80-100 | Lifecycle management |
| wake_processor.py | 80-100 | Detection logic |

**Total: ~2,000-2,400 lines across 25+ focused modules** (vs 2,813 lines in 3 god classes)

---

## Detailed Module Specifications

### 1. core/state_machine.py

#### Purpose
Centralized state management with explicit transitions and hooks.

#### Interface

```python
from enum import Enum, auto
from typing import Callable, Optional
from dataclasses import dataclass

class SessionState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()

@dataclass
class TransitionContext:
    """Context passed to transition hooks."""
    from_state: SessionState
    to_state: SessionState
    trigger: str  # What caused the transition
    speaker_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

class InvalidTransitionError(Exception):
    """Raised when attempting an illegal state transition."""
    pass

class SessionStateMachine:
    """Manages session state with validated transitions and hooks.

    Legal transitions:
        IDLE -> LISTENING (wake word detected)
        LISTENING -> PROCESSING (speech captured)
        LISTENING -> IDLE (timeout, no speech)
        PROCESSING -> IDLE (response complete)
        PROCESSING -> LISTENING (barge-in interrupt)
    """

    VALID_TRANSITIONS: dict[SessionState, set[SessionState]] = {
        SessionState.IDLE: {SessionState.LISTENING},
        SessionState.LISTENING: {SessionState.PROCESSING, SessionState.IDLE},
        SessionState.PROCESSING: {SessionState.IDLE, SessionState.LISTENING},
    }

    def __init__(self):
        self._state = SessionState.IDLE
        self._on_enter_hooks: dict[SessionState, list[Callable]] = defaultdict(list)
        self._on_exit_hooks: dict[SessionState, list[Callable]] = defaultdict(list)
        self._on_transition_hooks: list[Callable] = []

    @property
    def state(self) -> SessionState:
        """Current state (read-only)."""
        return self._state

    def transition_to(
        self,
        target: SessionState,
        trigger: str = "manual",
        **metadata
    ) -> TransitionContext:
        """Transition to a new state.

        Args:
            target: Destination state
            trigger: What caused this transition (for logging/debugging)
            **metadata: Additional context for hooks

        Returns:
            TransitionContext with full transition details

        Raises:
            InvalidTransitionError: If transition is not allowed
        """
        if target not in self.VALID_TRANSITIONS[self._state]:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.name} to {target.name}"
            )

        context = TransitionContext(
            from_state=self._state,
            to_state=target,
            trigger=trigger,
            metadata=metadata,
        )

        # Execute hooks
        self._run_exit_hooks(context)
        self._state = target
        self._run_enter_hooks(context)
        self._run_transition_hooks(context)

        return context

    def on_enter(self, state: SessionState, hook: Callable[[TransitionContext], None]):
        """Register a hook to run when entering a state."""
        self._on_enter_hooks[state].append(hook)

    def on_exit(self, state: SessionState, hook: Callable[[TransitionContext], None]):
        """Register a hook to run when exiting a state."""
        self._on_exit_hooks[state].append(hook)

    def on_transition(self, hook: Callable[[TransitionContext], None]):
        """Register a hook to run on any transition."""
        self._on_transition_hooks.append(hook)

    def can_transition_to(self, target: SessionState) -> bool:
        """Check if transition to target state is valid."""
        return target in self.VALID_TRANSITIONS[self._state]

    def reset(self):
        """Reset to IDLE state without hooks."""
        self._state = SessionState.IDLE
```

#### Usage Example

```python
# In VoiceSession
self.state_machine = SessionStateMachine()

# Register hooks
self.state_machine.on_enter(SessionState.LISTENING, self._on_start_listening)
self.state_machine.on_exit(SessionState.PROCESSING, self._on_stop_processing)
self.state_machine.on_transition(self._log_transition)

# Transition with validation
try:
    self.state_machine.transition_to(
        SessionState.LISTENING,
        trigger="wake_word",
        speaker_id=user_id,
    )
except InvalidTransitionError:
    logger.warning("Invalid transition attempted")
```

---

### 2. tools/registry.py

#### Purpose
Pluggable tool registration and lookup.

#### Interface

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """Tool schema for LLM."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

class ToolHandler(ABC):
    """Base class for tool handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must match LLM tool call)."""
        pass

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Tool definition for LLM."""
        pass

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

class ToolRegistry:
    """Registry for tool handlers.

    Supports:
    - Registration by name
    - Lookup for execution
    - Definition export for LLM
    - Enable/disable individual tools
    """

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}
        self._disabled: set[str] = set()

    def register(self, handler: ToolHandler) -> None:
        """Register a tool handler."""
        if handler.name in self._handlers:
            raise ValueError(f"Tool '{handler.name}' already registered")
        self._handlers[handler.name] = handler

    def unregister(self, name: str) -> None:
        """Unregister a tool handler."""
        self._handlers.pop(name, None)
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        """Disable a tool (keeps registered but won't execute)."""
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """Re-enable a disabled tool."""
        self._disabled.discard(name)

    def get(self, name: str) -> Optional[ToolHandler]:
        """Get handler by name (None if not found or disabled)."""
        if name in self._disabled:
            return None
        return self._handlers.get(name)

    def get_definitions(self) -> list[dict]:
        """Get all enabled tool definitions for LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": h.definition.name,
                    "description": h.definition.description,
                    "parameters": h.definition.parameters,
                }
            }
            for name, h in self._handlers.items()
            if name not in self._disabled
        ]

    @property
    def enabled_tools(self) -> list[str]:
        """List of enabled tool names."""
        return [n for n in self._handlers if n not in self._disabled]
```

---

### 3. tools/handlers/search.py

#### Purpose
Web search and URL fetching tool handlers.

#### Interface

```python
from ..base import ToolHandler, ToolDefinition, ToolResult
from ...search import SearchService

class WebSearchHandler(ToolHandler):
    """Handler for web_search tool."""

    def __init__(self, search_service: SearchService, max_searches: int = 3):
        self._search = search_service
        self._max_searches = max_searches
        self._search_count = 0

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information. Returns snippets from top results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        )

    async def execute(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(success=False, output="", error="No query provided")

        # Enforce search limit
        if self._search_count >= self._max_searches:
            return ToolResult(
                success=False,
                output="",
                error=f"Search limit ({self._max_searches}) reached"
            )

        self._search_count += 1

        try:
            results = await self._search.search(query)
            return ToolResult(
                success=True,
                output=self._format_results(results),
                metadata={"query": query, "result_count": len(results)}
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def reset_count(self):
        """Reset search count (call at start of each conversation turn)."""
        self._search_count = 0

    def _format_results(self, results: list) -> str:
        # Format search results for LLM consumption
        ...

class FetchUrlHandler(ToolHandler):
    """Handler for fetch_url tool."""

    def __init__(self, search_service: SearchService):
        self._search = search_service

    @property
    def name(self) -> str:
        return "fetch_url"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fetch_url",
            description="Fetch and extract content from a URL.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    }
                },
                "required": ["url"]
            }
        )

    async def execute(self, args: dict) -> ToolResult:
        url = args.get("url", "")
        if not url:
            return ToolResult(success=False, output="", error="No URL provided")

        try:
            content = await self._search.fetch_url(url)
            return ToolResult(
                success=True,
                output=content,
                metadata={"url": url}
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

---

### 4. tools/handlers/image.py

#### Purpose
Image generation and editing tool handlers.

#### Interface

```python
from ..base import ToolHandler, ToolDefinition, ToolResult
from ...image import NanoBananaClient

class GenerateImageHandler(ToolHandler):
    """Handler for generate_image tool."""

    def __init__(
        self,
        image_client: NanoBananaClient,
        on_image_generated: Callable[[bytes, str, str], Awaitable[str]],
    ):
        self._client = image_client
        self._on_generated = on_image_generated

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_image",
            description="Generate an image from a text description and post to Discord.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate"
                    }
                },
                "required": ["prompt"]
            }
        )

    async def execute(self, args: dict) -> ToolResult:
        prompt = args.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        try:
            image_bytes, mime_type = await self._client.generate(prompt)

            # Post to Discord via callback
            result_msg = await self._on_generated(image_bytes, prompt, mime_type)

            return ToolResult(
                success=True,
                output=result_msg,
                metadata={"prompt": prompt, "mime_type": mime_type}
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class EditImageHandler(ToolHandler):
    """Handler for edit_image tool with chat context."""

    def __init__(
        self,
        image_client: NanoBananaClient,
        on_image_generated: Callable[[bytes, str, str], Awaitable[str]],
    ):
        self._client = image_client
        self._on_generated = on_image_generated

    @property
    def name(self) -> str:
        return "edit_image"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_image",
            description="Edit the previously generated image based on instructions.",
            parameters={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "How to modify the image"
                    }
                },
                "required": ["instructions"]
            }
        )

    async def execute(self, args: dict) -> ToolResult:
        instructions = args.get("instructions", "")
        if not instructions:
            return ToolResult(success=False, output="", error="No instructions provided")

        if not self._client.has_chat_context():
            return ToolResult(
                success=False,
                output="",
                error="No previous image to edit. Use generate_image first."
            )

        try:
            image_bytes, mime_type = await self._client.edit(instructions)
            result_msg = await self._on_generated(image_bytes, instructions, mime_type)

            return ToolResult(
                success=True,
                output=result_msg,
                metadata={"instructions": instructions}
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

---

### 5. pipeline/voice_pipeline.py

#### Purpose
Compose and orchestrate STT → LLM → TTS stages with cancellation support.

#### Interface

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Any
from dataclasses import dataclass
import asyncio

@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    speaker_activity: Optional[str] = None
    metadata: dict = field(default_factory=dict)

class PipelineStage(ABC):
    """Base class for pipeline stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging."""
        pass

    @abstractmethod
    async def process(self, input: Any, context: PipelineContext) -> Any:
        """Process input and return output."""
        pass

    async def cancel(self) -> None:
        """Cancel any ongoing processing."""
        pass

class VoicePipeline:
    """Orchestrates the voice processing pipeline.

    Pipeline: Audio → Transcription → Response → TTS → Playback

    Supports:
    - Stage composition
    - Cancellation at any point
    - Error handling and recovery
    - Progress callbacks
    """

    def __init__(
        self,
        transcription: "TranscriptionStage",
        response: "ResponseStage",
        tts: "TTSStage",
    ):
        self._transcription = transcription
        self._response = response
        self._tts = tts
        self._cancelled = False
        self._current_stage: Optional[str] = None
        self._tasks: list[asyncio.Task] = []

    async def process(
        self,
        audio: np.ndarray,
        context: PipelineContext,
        on_progress: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        """Run the full pipeline.

        Args:
            audio: Input audio (16kHz mono float32)
            context: Pipeline context with speaker info
            on_progress: Optional callback for progress updates
        """
        self._cancelled = False

        try:
            # Stage 1: Transcription
            self._current_stage = "transcription"
            if on_progress:
                on_progress("transcription_start", None)

            text = await self._transcription.process(audio, context)

            if self._cancelled:
                return

            if not text:
                if on_progress:
                    on_progress("transcription_empty", None)
                return

            if on_progress:
                on_progress("transcription_complete", text)

            # Stage 2: Response generation (streams to TTS)
            self._current_stage = "response"
            if on_progress:
                on_progress("response_start", None)

            # TTS stage handles streaming from response stage
            await self._tts.process_stream(
                self._response.generate_stream(text, context),
                context,
            )

            if on_progress:
                on_progress("complete", None)

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
            raise
        except Exception as e:
            logger.error(f"Pipeline error in {self._current_stage}: {e}")
            raise
        finally:
            self._current_stage = None

    async def cancel(self) -> None:
        """Cancel the pipeline and all stages."""
        self._cancelled = True

        # Cancel all stages
        await asyncio.gather(
            self._transcription.cancel(),
            self._response.cancel(),
            self._tts.cancel(),
            return_exceptions=True,
        )

        # Cancel any tracked tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently processing."""
        return self._current_stage is not None

    @property
    def current_stage(self) -> Optional[str]:
        """Get the current stage name."""
        return self._current_stage
```

---

### 6. sink/user_audio_state.py

#### Purpose
Encapsulate per-user audio state in a single dataclass.

#### Interface

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time

from ..detection import WakeWordDetector

@dataclass
class UserAudioState:
    """Per-user audio processing state.

    Encapsulates all state previously scattered across 4 dicts:
    - _per_user_detectors
    - _per_user_pre_buffers
    - _per_user_last_seen
    - _per_user_silence_attempts
    """

    user_id: int
    detector: Optional[WakeWordDetector] = None
    pre_buffer: list[np.ndarray] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    silence_attempts: int = 0

    # Pre-buffer limits
    MAX_PRE_BUFFER_SECONDS: float = 3.0
    SAMPLE_RATE: int = 16000

    def add_to_pre_buffer(self, audio: np.ndarray) -> None:
        """Add audio to pre-buffer, enforcing size limit."""
        self.pre_buffer.append(audio)
        self._trim_pre_buffer()

    def get_pre_buffer(self) -> np.ndarray:
        """Get concatenated pre-buffer."""
        if not self.pre_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.pre_buffer)

    def clear_pre_buffer(self) -> None:
        """Clear the pre-buffer."""
        self.pre_buffer.clear()

    def update_last_seen(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = time.time()

    def time_since_seen(self) -> float:
        """Get seconds since last audio received."""
        return time.time() - self.last_seen

    def increment_silence_attempts(self) -> int:
        """Increment and return silence attempt counter."""
        self.silence_attempts += 1
        return self.silence_attempts

    def reset_silence_attempts(self) -> None:
        """Reset silence attempt counter."""
        self.silence_attempts = 0

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.detector:
            self.detector.reset()
            self.detector = None
        self.pre_buffer.clear()

    def _trim_pre_buffer(self) -> None:
        """Trim pre-buffer to max size."""
        max_samples = int(self.MAX_PRE_BUFFER_SECONDS * self.SAMPLE_RATE)
        total_samples = sum(len(chunk) for chunk in self.pre_buffer)

        while total_samples > max_samples and self.pre_buffer:
            removed = self.pre_buffer.pop(0)
            total_samples -= len(removed)
```

---

### 7. sink/per_user_manager.py

#### Purpose
Manage lifecycle of per-user audio states.

#### Interface

```python
from typing import Optional, Iterator
from .user_audio_state import UserAudioState
from ..detection import WakeWordDetector

class PerUserAudioManager:
    """Manages per-user audio state lifecycle.

    Responsibilities:
    - Create/retrieve user states
    - Lazy detector initialization
    - User cleanup on disconnect
    - State iteration for batch operations
    """

    def __init__(
        self,
        wake_model_name: str,
        wake_threshold: float,
        wake_sample_rate: int,
    ):
        self._users: dict[int, UserAudioState] = {}
        self._wake_model_name = wake_model_name
        self._wake_threshold = wake_threshold
        self._wake_sample_rate = wake_sample_rate

    def get_or_create(self, user_id: int) -> UserAudioState:
        """Get existing state or create new one."""
        if user_id not in self._users:
            self._users[user_id] = UserAudioState(user_id=user_id)
        return self._users[user_id]

    def get(self, user_id: int) -> Optional[UserAudioState]:
        """Get state if exists, None otherwise."""
        return self._users.get(user_id)

    def ensure_detector(self, state: UserAudioState) -> WakeWordDetector:
        """Ensure user has a wake word detector (lazy init)."""
        if state.detector is None:
            state.detector = WakeWordDetector(
                model_name=self._wake_model_name,
                threshold=self._wake_threshold,
                sample_rate=self._wake_sample_rate,
            )
        return state.detector

    def cleanup_user(self, user_id: int) -> None:
        """Clean up user state on disconnect."""
        state = self._users.pop(user_id, None)
        if state:
            state.cleanup()

    def cleanup_all(self) -> None:
        """Clean up all users."""
        for state in self._users.values():
            state.cleanup()
        self._users.clear()

    def __iter__(self) -> Iterator[UserAudioState]:
        """Iterate over all user states."""
        return iter(self._users.values())

    def __len__(self) -> int:
        """Number of tracked users."""
        return len(self._users)

    @property
    def user_ids(self) -> set[int]:
        """Set of tracked user IDs."""
        return set(self._users.keys())
```

---

### 8. session/voice_session.py (Refactored)

#### Purpose
Slim orchestrator that coordinates components via composition.

#### Target: ~150 lines (down from 1,212)

```python
"""Voice session orchestrator.

Coordinates state machine, pipeline, and sink components.
All complex logic delegated to focused modules.
"""

import asyncio
import logging
from typing import Optional

import discord
import numpy as np

from ..core import SessionStateMachine, SessionState, TransitionContext
from ..pipeline import VoicePipeline, PipelineContext
from ..sink import StreamingSink
from ..audio import SAMPLE_RATE_PROCESSING
from .speaker_tracker import SpeakerTracker
from .greeting_service import GreetingService
from .phrases import get_random_ack, get_random_thinking

logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages voice assistant session for a Discord voice channel.

    Thin orchestrator that:
    - Manages state machine transitions
    - Coordinates pipeline execution
    - Handles events from sink
    - Delegates all complex logic to focused modules
    """

    def __init__(
        self,
        state_machine: SessionStateMachine,
        pipeline: VoicePipeline,
        speaker_tracker: SpeakerTracker,
        greeting_service: Optional[GreetingService] = None,
    ):
        self._state_machine = state_machine
        self._pipeline = pipeline
        self._speaker_tracker = speaker_tracker
        self._greeting_service = greeting_service

        # Discord
        self._voice_client: Optional[discord.VoiceClient] = None
        self._sink: Optional[StreamingSink] = None

        # Events
        self._wake_event = asyncio.Event()
        self._speech_end_event = asyncio.Event()
        self._pending_audio: Optional[np.ndarray] = None

        # Control
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Register state hooks
        self._register_state_hooks()

    def _register_state_hooks(self):
        """Register callbacks for state transitions."""
        self._state_machine.on_enter(SessionState.LISTENING, self._on_enter_listening)
        self._state_machine.on_exit(SessionState.PROCESSING, self._on_exit_processing)

    def _on_enter_listening(self, ctx: TransitionContext):
        """Hook: Entering listening state."""
        if self._sink:
            self._sink.start_listening()

    def _on_exit_processing(self, ctx: TransitionContext):
        """Hook: Exiting processing state."""
        # Cleanup handled by pipeline
        pass

    # --- Event Callbacks (from sink) ---

    def _on_wake_word(self, user_id: Optional[int] = None):
        """Callback when wake word detected."""
        state = self._state_machine.state
        current_speaker = self._speaker_tracker.current_speaker_id

        if state == SessionState.IDLE:
            # Normal activation
            self._speaker_tracker.set_speaker(user_id, self._voice_client)
            self._loop.call_soon_threadsafe(self._wake_event.set)

        elif state == SessionState.PROCESSING and user_id == current_speaker:
            # Barge-in: same user interrupting
            logger.info(f"Barge-in from user {user_id}")
            asyncio.run_coroutine_threadsafe(
                self._handle_barge_in(user_id),
                self._loop,
            )

    def _on_speech_end(self, audio: np.ndarray, user_id: Optional[int] = None):
        """Callback when speech ends."""
        if self._state_machine.state == SessionState.LISTENING:
            self._pending_audio = audio
            self._speaker_tracker.set_speaker(user_id, self._voice_client)
            self._loop.call_soon_threadsafe(self._speech_end_event.set)

    async def _handle_barge_in(self, user_id: int):
        """Handle barge-in interrupt from current speaker."""
        # Cancel current pipeline
        await self._pipeline.cancel()

        # Transition back to listening
        self._state_machine.transition_to(
            SessionState.LISTENING,
            trigger="barge_in",
            speaker_id=user_id,
        )
        self._wake_event.set()

    # --- Main Loop ---

    async def start(self, voice_client: discord.VoiceClient):
        """Start the voice session."""
        self._voice_client = voice_client
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Create sink with callbacks
        self._sink = StreamingSink(
            on_wake_word=self._on_wake_word,
            on_speech_end=self._on_speech_end,
            # ... other config
        )

        voice_client.start_recording(self._sink, lambda *a: None)
        await self._run_loop()

    async def stop(self):
        """Stop the voice session."""
        self._running = False
        await self._pipeline.cancel()
        # ... cleanup

    async def _run_loop(self):
        """Main session loop."""
        while self._running:
            try:
                # Wait for wake word
                self._state_machine.transition_to(SessionState.IDLE, trigger="loop_start")
                await self._wake_event.wait()
                self._wake_event.clear()

                if not self._running:
                    break

                # Transition to listening
                self._state_machine.transition_to(
                    SessionState.LISTENING,
                    trigger="wake_word",
                )

                # Wait for speech
                audio = await self._wait_for_speech()
                if audio is None:
                    continue

                # Process through pipeline
                self._state_machine.transition_to(
                    SessionState.PROCESSING,
                    trigger="speech_captured",
                )

                context = PipelineContext(
                    speaker_id=self._speaker_tracker.current_speaker_id,
                    speaker_name=self._speaker_tracker.current_speaker_name,
                    speaker_activity=self._speaker_tracker.current_activity,
                )

                await self._pipeline.process(audio, context)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session loop error: {e}")
                await asyncio.sleep(1.0)

    async def _wait_for_speech(self) -> Optional[np.ndarray]:
        """Wait for speech end event with timeout."""
        # Simplified - details in actual implementation
        try:
            await asyncio.wait_for(self._speech_end_event.wait(), timeout=30.0)
            self._speech_end_event.clear()
            return self._pending_audio
        except asyncio.TimeoutError:
            return None

    async def greet_user(self, member: discord.Member):
        """Greet a user who joined the channel."""
        if self._greeting_service:
            await self._greeting_service.greet(member, self._state_machine.state)
```

---

## Git Workflow & Branching Strategy

### Branch Per Phase, Merge When Stable

Each phase gets its own branch. Merge to master only when the bot is fully functional and tested.

```bash
# Phase 1: Core abstractions
git checkout -b refactor/phase1-core-abstractions
# ... implement, test, verify bot works ...
git checkout master && git merge refactor/phase1-core-abstractions

# Phase 2: Pipeline extraction
git checkout -b refactor/phase2-pipeline
# ... implement, test, verify bot works ...
git checkout master && git merge refactor/phase2-pipeline

# Phase 3: Per-user audio
git checkout -b refactor/phase3-sink-extraction
# ... implement, test, verify bot works ...
git checkout master && git merge refactor/phase3-sink-extraction

# Phase 4: Session cleanup
git checkout -b refactor/phase4-session-cleanup
# ... implement, test, verify bot works ...
git checkout master && git merge refactor/phase4-session-cleanup
```

### Why This Approach

| Benefit | Explanation |
|---------|-------------|
| **Safe experimentation** | Master always has working bot |
| **Reviewable chunks** | Each phase is a focused PR, not one massive change |
| **Easy rollback** | Revert a single phase if issues found |
| **Urgent fixes** | Can hotfix master between phases |
| **Clear progress** | Each merge is a milestone |

### Branch Naming Convention

```
refactor/phase1-core-abstractions     # Phase 1
refactor/phase2-pipeline              # Phase 2
refactor/phase3-sink-extraction       # Phase 3
refactor/phase4-session-cleanup       # Phase 4
```

### Pre-Phase Checklist

Before starting each phase:

```bash
# 1. Ensure master is clean
git checkout master
git status  # Should be clean

# 2. Pull latest (if working with others)
git pull origin master

# 3. Create phase branch
git checkout -b refactor/phaseN-description

# 4. Verify bot works before changes
python -m src.main  # Quick smoke test
```

### During Phase Work

```bash
# Commit frequently with clear messages
git add src/tools/registry.py
git commit -m "Add ToolRegistry with register/lookup/definitions"

# Keep commits focused (one logical change per commit)
git add src/tools/handlers/search.py tests/tools/test_search_handler.py
git commit -m "Add WebSearchHandler with unit tests"
```

### Phase Completion Checklist

Before merging each phase:

- [ ] All new modules have unit tests
- [ ] Bot runs and responds to voice commands
- [ ] Feature flag toggling works (old path still works)
- [ ] No regressions in existing functionality
- [ ] Code reviewed (self or peer)

```bash
# Final verification
python -m src.main  # Full test
pytest tests/       # Run all tests

# Merge to master
git checkout master
git merge refactor/phaseN-description
git push origin master

# Clean up branch (optional)
git branch -d refactor/phaseN-description
```

### Handling Urgent Fixes During Refactoring

If a bug is found in production while working on a refactor branch:

```bash
# 1. Stash or commit current work
git stash  # or commit WIP

# 2. Switch to master
git checkout master

# 3. Create hotfix branch
git checkout -b hotfix/fix-description

# 4. Fix, test, merge
# ... make fix ...
git checkout master && git merge hotfix/fix-description

# 5. Return to refactor and rebase
git checkout refactor/phaseN-description
git rebase master  # Get the hotfix

# 6. Continue work
git stash pop  # if you stashed
```

---

## Refactoring Phases

### Phase 1: Core Abstractions (Low Risk)

**Duration**: 1-2 sessions
**Risk**: Low
**Impact**: Foundation for future phases
**Branch**: `refactor/phase1-core-abstractions`

#### Step-by-Step Implementation

##### Step 1.1: Create Directory Structure

```bash
mkdir -p src/core src/tools/handlers tests/core tests/tools
touch src/core/__init__.py src/tools/__init__.py src/tools/handlers/__init__.py
```

##### Step 1.2: Create `core/state_machine.py`

1. Create file with `SessionState` enum and `SessionStateMachine` class
2. Extract states from `voice_session.py:161-162` (idle/listening/processing)
3. Define `VALID_TRANSITIONS` dict
4. Implement `transition_to()` with validation
5. Add hook registration (`on_enter`, `on_exit`, `on_transition`)
6. Write tests in `tests/core/test_state_machine.py`

**Test before continuing:**
```bash
pytest tests/core/test_state_machine.py -v
```

##### Step 1.3: Create `tools/base.py` and `tools/registry.py`

1. Define `ToolDefinition` dataclass (name, description, parameters)
2. Define `ToolResult` dataclass (success, output, error, metadata)
3. Define `ToolHandler` ABC with `name`, `definition`, `execute()`
4. Create `ToolRegistry` with register/get/get_definitions
5. Write tests in `tests/tools/test_registry.py`

**Test before continuing:**
```bash
pytest tests/tools/test_registry.py -v
```

##### Step 1.4: Create Tool Handlers (One at a Time)

**Order matters** - start with simplest, build confidence:

1. **`tools/handlers/search.py`** (WebSearchHandler, FetchUrlHandler)
   - Extract from `groq_client.py:736-742`
   - Port search limit logic (`_search_calls_this_request`)
   - Test with mock SearchService

2. **`tools/handlers/discord.py`** (TagUserHandler, PostMessageHandler, PostTableHandler)
   - Extract from `groq_client.py:669-694`
   - Port `_format_table()` method
   - Test with mock callbacks

3. **`tools/handlers/image.py`** (GenerateImageHandler, EditImageHandler)
   - Extract from `groq_client.py:638-666`
   - Port `has_chat()` check for edit
   - Test with mock NanoBananaClient

4. **`tools/handlers/research.py`** (DeepResearchHandler)
   - Extract from `groq_client.py:697-730`
   - Port `_build_claude_delegation_prompt()`
   - Port timeout logic (90s with tools, 45s without)
   - Test with mock ClaudeClient

**Test after each handler:**
```bash
pytest tests/tools/test_*_handler.py -v
```

##### Step 1.5: Create `tools/executor.py`

1. Create `ToolExecutor` that wraps registry
2. Add error handling (try/except around execute)
3. Add tool name aliasing (`TOOL_ALIASES` from `groq_client.py:184-188`)
4. Add logging
5. Write tests

##### Step 1.6: Create `llm/prompt_builder.py`

1. Extract `_build_system_prompt()` from `groq_client.py:840-957`
2. Make template sections configurable
3. Keep roast data and personality separate
4. Write tests

##### Step 1.7: Wire Up with Feature Flags

1. Add flags to `config.py`:
   ```python
   use_new_tool_registry: bool = False
   use_new_prompt_builder: bool = False
   ```

2. Update `groq_client.py.__init__()` to accept optional ToolRegistry
3. Add conditional in `_execute_tool()`:
   ```python
   if self.settings.use_new_tool_registry and self._tool_registry:
       # new path
   else:
       # old path
   ```

4. Add conditional in `_build_system_prompt()`:
   ```python
   if self.settings.use_new_prompt_builder and self._prompt_builder:
       return self._prompt_builder.build(context=self._context)
   else:
       # old inline prompt
   ```

##### Step 1.8: Integration Testing

1. Enable flag: `USE_NEW_TOOL_REGISTRY=true`
2. Run bot: `python -m src.main`
3. Test voice commands that trigger each tool
4. Verify logs show new handlers being called
5. Disable flag, verify old path still works

##### Step 1.9: Remove Old Code (After Verification)

1. Remove inline tool handlers from `groq_client.py`
2. Remove inline prompt building
3. Remove feature flags
4. Final test

#### Verification Checklist

- [ ] `core/state_machine.py` created with tests
- [ ] `tools/registry.py` and `tools/base.py` created with tests
- [ ] All 8 tool handlers created with tests
- [ ] `tools/executor.py` created with tests
- [ ] `llm/prompt_builder.py` created with tests
- [ ] Feature flags working (toggle between old/new)
- [ ] Full voice interaction works with new code
- [ ] Old code removed, flags removed
- [ ] groq_client.py reduced to ~600 lines

#### Files Changed

| File | Change |
|------|--------|
| `src/core/__init__.py` | New |
| `src/core/state_machine.py` | New |
| `src/tools/__init__.py` | New |
| `src/tools/base.py` | New |
| `src/tools/registry.py` | New |
| `src/tools/executor.py` | New |
| `src/tools/handlers/__init__.py` | New |
| `src/tools/handlers/search.py` | New |
| `src/tools/handlers/image.py` | New |
| `src/tools/handlers/discord.py` | New |
| `src/tools/handlers/research.py` | New |
| `src/llm/prompt_builder.py` | New |
| `src/llm/groq_client.py` | Modified (-400 lines) |
| `src/config.py` | Modified (add flags, then remove) |

---

### Phase 2: Pipeline Extraction (Medium Risk)

**Duration**: 2-3 sessions
**Risk**: Medium (async complexity)
**Impact**: Clear data flow, testable stages
**Branch**: `refactor/phase2-pipeline`

**Prerequisites**: Phase 1 complete and merged to master

#### Step-by-Step Implementation

##### Step 2.1: Create Directory Structure

```bash
mkdir -p src/pipeline tests/pipeline
touch src/pipeline/__init__.py
```

##### Step 2.2: Create `pipeline/base.py`

1. Define `PipelineContext` dataclass:
   - `speaker_id`, `speaker_name`, `speaker_activity`
   - `metadata` dict for extensibility
2. Define `PipelineStage` ABC:
   - `name` property
   - `async process(input, context) -> output`
   - `async cancel()` - default no-op
3. Write tests

##### Step 2.3: Create `audio/playback_queue.py`

**CRITICAL**: This handles the sentence ordering edge case (see `refactor-edge-cases.md` Section 4.4)

1. Create `OrderedPlaybackQueue` class
2. Port ordering logic from `voice_session.py:861-912`:
   - Store `(sentence_num, audio)` tuples
   - `pending` dict for out-of-order arrivals
   - `next_to_play` counter
3. Add `async put(sentence_num, audio)`
4. Add `async get() -> audio` (blocks until next in order)
5. Add `cancel()` - clears queue, unblocks waiters
6. Port the `None` sentinel pattern for clean shutdown
7. Write tests with simulated out-of-order completion

**Edge case test:**
```python
async def test_out_of_order_playback():
    queue = OrderedPlaybackQueue()
    # Sentence 2 arrives before sentence 1
    await queue.put(2, audio_2)
    await queue.put(1, audio_1)
    # Should return in order
    assert await queue.get() == audio_1
    assert await queue.get() == audio_2
```

##### Step 2.4: Create `pipeline/transcription_stage.py`

1. Wrap `GroqWhisperClient`
2. Port wake word stripping from `voice_session.py:996-1010`:
   - Strip "hey jarvis" prefix
   - Handle variations ("hey, jarvis", etc.)
3. Return transcribed text or empty string
4. Write tests with mock STT

##### Step 2.5: Create `llm/chat_orchestrator.py`

**CRITICAL**: This is the most complex extraction. Preserve:
- Tool loop from `groq_client.py:299-469`
- Search limit enforcement (3 max)
- `last_action_success` tracking for error recovery
- Graceful failure message construction

1. Create `ChatOrchestrator` class
2. Accept `ToolExecutor`, `BaseLLMClient` as dependencies
3. Port `chat_stream()` core logic (tool loop)
4. Keep streaming response separate
5. Port error recovery paths (see `refactor-edge-cases.md` Section 5)
6. Write tests with mock LLM and tools

**Edge case tests:**
```python
async def test_search_limit_enforced():
    # After 3 searches, should force answer

async def test_action_success_fallback():
    # If image posted but model output fails, use "Done!" message

async def test_tool_name_aliasing():
    # "web_fetch_url" should map to "fetch_url"
```

##### Step 2.6: Create `pipeline/response_stage.py`

1. Wrap `ChatOrchestrator`
2. Accept `PipelineContext` for speaker info
3. Yield response chunks (async generator)
4. Support cancellation
5. Write tests

##### Step 2.7: Create `pipeline/tts_stage.py`

**CRITICAL**: Preserve parallel synthesis pattern from `voice_session.py:1055-1145`

1. Create `TTSStage` class with:
   - `TTSSynthesizer` dependency
   - `OrderedPlaybackQueue` dependency
2. Port sentence boundary detection from `voice_session.py:1096-1122`:
   - Split on newlines (primary)
   - Split on sentence-ending punctuation (secondary)
3. Port parallel synthesis pattern:
   - Create task per sentence
   - Each task synthesizes and queues with sentence number
4. Port cleanup pattern:
   - `finally` block always queues `None` sentinel
   - Await all TTS tasks before returning
5. Write tests

**Edge case test:**
```python
async def test_tts_error_queues_empty_audio():
    # If TTS fails for sentence 2, should queue empty audio
    # so sentence 3 doesn't get stuck waiting
```

##### Step 2.8: Create `pipeline/voice_pipeline.py`

1. Compose stages: Transcription → Response → TTS
2. Port thinking phrase parallel pattern from `voice_session.py:1055-1064`:
   - Start thinking phrase TTS as background task
   - Start LLM request immediately (parallel)
   - Playback waits for thinking before first sentence
3. Implement `cancel()`:
   - Cancel all stages
   - Cancel any tracked tasks
   - Clear queues
4. Add progress callbacks
5. Write integration tests

##### Step 2.9: Wire Up with Feature Flags

1. Add flags to `config.py`:
   ```python
   use_new_pipeline: bool = False
   use_new_playback_queue: bool = False
   ```

2. Update `voice_session.py` to accept optional `VoicePipeline`
3. Add conditional in `_process_speech()`:
   ```python
   if self.settings.use_new_pipeline and self._pipeline:
       await self._pipeline.process(audio, context)
   else:
       # old inline processing
   ```

##### Step 2.10: Integration Testing

**CRITICAL**: Test these edge cases manually:

1. **Normal conversation** - wake → question → response
2. **Long response** - verify sentence ordering
3. **Tool use** - search, image generation
4. **Cancellation** - stop bot mid-response
5. **Thinking phrase** - verify it plays before response
6. **Error recovery** - unplug network during response

##### Step 2.11: Remove Old Code

1. Remove inline pipeline from `voice_session.py`
2. Remove `_process_speech()` inline logic
3. Remove `_playback_consumer()` (now in queue)
4. Remove `_synthesize_and_queue()` (now in TTS stage)
5. Remove feature flags

#### Verification Checklist

- [ ] `pipeline/base.py` created with tests
- [ ] `audio/playback_queue.py` created with tests (ordering works)
- [ ] `pipeline/transcription_stage.py` created with tests
- [ ] `llm/chat_orchestrator.py` created with tests
- [ ] `pipeline/response_stage.py` created with tests
- [ ] `pipeline/tts_stage.py` created with tests
- [ ] `pipeline/voice_pipeline.py` created with integration tests
- [ ] Thinking phrase parallelism preserved
- [ ] Sentence ordering preserved
- [ ] Cancellation works cleanly
- [ ] All error recovery paths work
- [ ] voice_session.py reduced to ~400 lines

#### Files Changed

| File | Change |
|------|--------|
| `src/pipeline/__init__.py` | New |
| `src/pipeline/base.py` | New |
| `src/pipeline/transcription_stage.py` | New |
| `src/pipeline/response_stage.py` | New |
| `src/pipeline/tts_stage.py` | New |
| `src/pipeline/voice_pipeline.py` | New |
| `src/audio/playback_queue.py` | New |
| `src/llm/chat_orchestrator.py` | New |
| `src/bot/voice_session.py` | Modified (-400 lines) |
| `src/config.py` | Modified (add flags, then remove) |

---

### Phase 3: Per-User Audio (Medium Risk)

**Duration**: 1-2 sessions
**Risk**: Medium (timing-sensitive code)
**Impact**: Cleaner sink, testable wake detection
**Branch**: `refactor/phase3-sink-extraction`

**Prerequisites**: Phase 2 complete and merged to master

#### Step-by-Step Implementation

##### Step 3.1: Create Directory Structure

```bash
mkdir -p src/sink tests/sink
touch src/sink/__init__.py
```

##### Step 3.2: Create `sink/user_audio_state.py`

**CRITICAL**: This consolidates 4 scattered dicts into one dataclass (see `refactor-edge-cases.md` Section 3.5)

1. Create `UserAudioState` dataclass with:
   - `user_id: int`
   - `detector: Optional[WakeWordDetector]`
   - `pre_buffer: list[np.ndarray]`
   - `last_seen: float`
   - `silence_attempts: int`

2. Port buffer management from `voice_sink.py:510-518`:
   - `add_to_pre_buffer(audio)` - with FIFO eviction
   - `get_pre_buffer() -> np.ndarray`
   - `clear_pre_buffer()`
   - `MAX_PRE_BUFFER_SECONDS = 1.5`

3. Add timing methods:
   - `update_last_seen()`
   - `time_since_seen() -> float`

4. Add silence attempt tracking:
   - `increment_silence_attempts() -> int`
   - `reset_silence_attempts()`

5. Add `cleanup()` for resource release

6. Write tests

**Edge case test:**
```python
def test_pre_buffer_fifo_eviction():
    state = UserAudioState(user_id=123)
    # Add 3 seconds of audio (exceeds 1.5s limit)
    state.add_to_pre_buffer(large_chunk)
    # Should have trimmed to ~1.5s
    assert state.get_pre_buffer().shape[0] <= 24000  # 1.5s at 16kHz
```

##### Step 3.3: Create `sink/per_user_manager.py`

1. Create `PerUserAudioManager` class
2. Port user lifecycle from `voice_sink.py:152-200`:
   - `get_or_create(user_id) -> UserAudioState`
   - `get(user_id) -> Optional[UserAudioState]`
   - `ensure_detector(state) -> WakeWordDetector` (lazy init)
   - `cleanup_user(user_id)`
   - `cleanup_all()`

3. Port max detector limit (`_max_detectors = 10`)
4. Add iteration support (`__iter__`, `__len__`)
5. Write tests

##### Step 3.4: Create `sink/wake_processor.py`

**CRITICAL**: Preserve timing-sensitive patterns (see `refactor-edge-cases.md` Section 2.3)

1. Create `WakeWordProcessor` class with:
   - `PerUserAudioManager` dependency
   - Callback for wake word detection

2. Port detection logic from `voice_sink.py:521-533`:
   - `process_audio(user_id, audio) -> bool` (returns True if wake word)
   - Capture `wake_word_user_id` at detection time (NOT later)

3. Port silence padding from `voice_sink.py:315-380`:
   - `check_idle_audio_gap() -> Optional[int]` (returns user_id if detected)
   - Gap window: 250ms-2.5s
   - Max 5 attempts per user per gap
   - Feed 500ms silence (8000 samples)

4. Port flush logic from `voice_sink.py:456-465`:
   - `flush_on_gap(pre_gap_user_id) -> Optional[int]`
   - Try flush on all detectors after audio gap

5. Write tests with simulated timing

**Edge case tests:**
```python
def test_wake_word_user_captured_at_detection():
    # User A triggers wake word
    # User B sends packet before start_listening()
    # wake_word_user_id should still be User A

def test_silence_attempts_limited():
    # After 5 silence padding attempts, should stop trying

def test_gap_window_respected():
    # Gap < 250ms: no silence padding
    # Gap > 2.5s: no silence padding (stale)
    # Gap 250ms-2.5s: attempt silence padding
```

##### Step 3.5: Wire Up with Feature Flags

1. Add flags to `config.py`:
   ```python
   use_new_sink_manager: bool = False
   use_new_wake_processor: bool = False
   ```

2. Update `StreamingSink.__init__()` to create new components:
   ```python
   if settings.use_new_sink_manager:
       self._user_manager = PerUserAudioManager(...)
       self._wake_processor = WakeWordProcessor(self._user_manager, ...)
   ```

3. Add conditionals in `write()`:
   ```python
   if self.settings.use_new_sink_manager:
       state = self._user_manager.get_or_create(user_id)
       state.add_to_pre_buffer(audio_16k)
       if self._wake_processor.process_audio(user_id, audio_16k):
           self._on_wake_word()
   else:
       # old inline logic
   ```

##### Step 3.6: Integration Testing

**CRITICAL**: These are timing-sensitive - test carefully:

1. **Single user wake word** - normal detection
2. **Multi-user in channel** - correct user attribution
3. **Speaker isolation** - only wake user's audio buffered
4. **PTT pattern** - wake word with audio gap, then speech
5. **Flush detection** - wake word at end of utterance
6. **Rapid switching** - multiple users saying wake word

##### Step 3.7: Remove Old Code

1. Remove per-user dicts from `StreamingSink`:
   - `_per_user_detectors`
   - `_per_user_pre_buffers`
   - `_per_user_last_seen`
   - `_per_user_silence_attempts`

2. Remove inline detection logic from `write()`
3. Remove `check_idle_audio_gap()` inline logic
4. Remove feature flags

#### Verification Checklist

- [ ] `sink/user_audio_state.py` created with tests
- [ ] `sink/per_user_manager.py` created with tests
- [ ] `sink/wake_processor.py` created with tests
- [ ] Wake word user captured at detection time
- [ ] Pre-buffer FIFO eviction works
- [ ] Silence padding gap window (250ms-2.5s) preserved
- [ ] Max 5 silence attempts per user preserved
- [ ] Speaker isolation still works
- [ ] PTT timeout still works
- [ ] voice_sink.py reduced to ~200 lines

#### Files Changed

| File | Change |
|------|--------|
| `src/sink/__init__.py` | New |
| `src/sink/user_audio_state.py` | New |
| `src/sink/per_user_manager.py` | New |
| `src/sink/wake_processor.py` | New |
| `src/bot/voice_sink.py` | Modified (-200 lines), move to `src/sink/streaming_sink.py` |
| `src/config.py` | Modified (add flags, then remove) |

---

### Phase 4: Session Cleanup (Low Risk)

**Duration**: 1 session
**Risk**: Low
**Impact**: Final cleanup, testability
**Branch**: `refactor/phase4-session-cleanup`

**Prerequisites**: Phase 3 complete and merged to master

#### Step-by-Step Implementation

##### Step 4.1: Create Directory Structure

```bash
mkdir -p src/session tests/session
touch src/session/__init__.py
```

##### Step 4.2: Create `session/phrases.py`

1. Move phrase constants from `voice_session.py`:
   - `ACK_PHRASES` list
   - `THINKING_PHRASES` dict (generic and personalized)

2. Add helper functions:
   - `get_random_ack() -> str`
   - `get_thinking_phrase(speaker_name: Optional[str]) -> str`

3. Write tests

##### Step 4.3: Create `session/speaker_tracker.py`

1. Create `SpeakerTracker` class with:
   - `current_speaker_id: Optional[int]`
   - `current_speaker_name: Optional[str]`
   - `current_speaker_activity: Optional[str]`

2. Port speaker resolution from `voice_session.py:295-320`:
   - `set_speaker(user_id, voice_client)` - resolves name and activity
   - `clear_speaker()`

3. Port activity detection:
   - Get Discord presence/activity for current speaker
   - Used for game context in prompts

4. Write tests with mock VoiceClient

##### Step 4.4: Create `session/greeting_service.py`

**CRITICAL**: Preserve state recheck pattern (see `refactor-edge-cases.md` Section 3.3)

1. Create `GreetingService` class with:
   - LLM client dependency
   - TTS synthesizer dependency
   - Cooldown tracking dict

2. Port greeting logic from `voice_session.py:720-810`:
   - `async greet(member, current_state, play_audio_callback)`
   - Cooldown check (5 minutes per user)
   - State check BEFORE waiting for silence
   - State recheck AFTER waiting for silence
   - LLM greeting generation
   - TTS synthesis and playback

3. Port silence waiting (or accept callback for it)

4. Write tests

**Edge case test:**
```python
async def test_greeting_aborted_if_state_changes():
    # Start greeting
    # State changes to LISTENING during silence wait
    # Greeting should be aborted
```

##### Step 4.5: Create `bot/voice_client_adapter.py`

1. Create `VoiceClientAdapter` protocol/ABC:
   ```python
   class VoiceClientAdapter(Protocol):
       async def play(self, source: AudioSource) -> None: ...
       async def stop(self) -> None: ...
       def is_playing(self) -> bool: ...
       async def speak(self, speaking: bool) -> None: ...
       def start_recording(self, sink, callback) -> None: ...
       def stop_recording(self) -> None: ...
   ```

2. Create `DiscordVoiceClientAdapter` implementation:
   - Wraps real `discord.VoiceClient`
   - Handles websocket errors gracefully

3. Create `MockVoiceClientAdapter` for testing:
   - Records all calls
   - Simulates playback completion

4. Write tests

##### Step 4.6: Wire Up with Feature Flags

1. Add flags to `config.py`:
   ```python
   use_new_greeting_service: bool = False
   use_new_speaker_tracker: bool = False
   ```

2. Update `VoiceSession.__init__()` to accept new components
3. Add conditionals for greeting and speaker tracking

##### Step 4.7: Final Session Cleanup

1. Move `voice_session.py` from `src/bot/` to `src/session/`
2. Update imports throughout codebase
3. Remove all inline logic that's now in extracted modules:
   - Phrase selection → `phrases.py`
   - Speaker resolution → `speaker_tracker.py`
   - Greeting → `greeting_service.py`
   - State management → `state_machine.py` (Phase 1)
   - Pipeline → `voice_pipeline.py` (Phase 2)

4. Final `VoiceSession` should be thin orchestrator:
   - Component initialization
   - Event callback registration
   - Main loop coordination
   - Cleanup

##### Step 4.8: Verify Complete Testability

1. Create `tests/session/test_voice_session.py`
2. Test session with all mock dependencies:
   - Mock state machine
   - Mock pipeline
   - Mock sink
   - Mock voice client adapter
   - Mock greeting service
   - Mock speaker tracker

3. Verify no Discord connection needed for unit tests

#### Verification Checklist

- [ ] `session/phrases.py` created with tests
- [ ] `session/speaker_tracker.py` created with tests
- [ ] `session/greeting_service.py` created with tests
- [ ] `bot/voice_client_adapter.py` created with tests
- [ ] Greeting state recheck preserved
- [ ] Greeting cooldown preserved
- [ ] Activity detection works
- [ ] Full voice interaction works
- [ ] voice_session.py is ~150 lines
- [ ] VoiceSession fully unit testable (no Discord needed)

#### Files Changed

| File | Change |
|------|--------|
| `src/session/__init__.py` | New |
| `src/session/phrases.py` | New |
| `src/session/speaker_tracker.py` | New |
| `src/session/greeting_service.py` | New |
| `src/bot/voice_client_adapter.py` | New |
| `src/bot/voice_session.py` | Move to `src/session/voice_session.py`, reduce to ~150 lines |
| `src/bot/discord_bot.py` | Update imports |
| `src/main.py` | Update imports |
| `src/config.py` | Modified (add flags, then remove) |

---

### Post-Refactoring: Barge-in Feature

After Phase 4, implementing barge-in becomes trivial:

```python
# In session/voice_session.py (~10 lines of new code)

def _on_wake_word(self, user_id: Optional[int] = None):
    """Callback when wake word detected."""
    current_state = self._state_machine.state

    if current_state == SessionState.IDLE:
        # Normal activation
        self._speaker_tracker.set_speaker(user_id, self._voice_client)
        self._loop.call_soon_threadsafe(self._wake_event.set)

    elif current_state == SessionState.PROCESSING:
        # Barge-in: check if same user
        if user_id == self._speaker_tracker.current_speaker_id:
            logger.info(f"Barge-in from user {user_id}")
            asyncio.run_coroutine_threadsafe(
                self._handle_barge_in(user_id),
                self._loop,
            )

async def _handle_barge_in(self, user_id: int):
    """Handle barge-in interrupt from current speaker."""
    # Cancel current pipeline (stops TTS, clears queue)
    await self._pipeline.cancel()

    # Transition back to listening
    self._state_machine.transition_to(
        SessionState.LISTENING,
        trigger="barge_in",
        speaker_id=user_id,
    )

    # Signal wake event to continue loop
    self._wake_event.set()
```

**Why this is now easy:**
- `pipeline.cancel()` handles all cleanup (TTS tasks, queue, playback)
- `state_machine.transition_to()` validates and triggers hooks
- Sink already captures `wake_word_user_id` at detection time
- Speaker tracker already knows who's speaking

---

## Migration Strategy

### Parallel Development

Keep existing code working while building new modules:

```
# During Phase 1
groq_client.py (existing) ─── still works
tools/registry.py (new) ──── builds alongside
tools/handlers/* (new) ───── builds alongside

# Switchover
groq_client.py ← inject ToolRegistry
groq_client.py ← remove inline handlers
```

### Feature Flags

Use settings to toggle new vs old code paths. This enables:
- **Safe testing**: Toggle on, test, toggle off if broken
- **Gradual rollout**: Enable one feature at a time
- **Easy rollback**: Just flip the flag, no code revert needed

#### Feature Flag Configuration

```python
# config.py - Add to Settings class
class Settings(BaseSettings):
    # ... existing settings ...

    # Refactoring feature flags (Phase 1)
    use_new_state_machine: bool = False
    use_new_tool_registry: bool = False
    use_new_prompt_builder: bool = False

    # Refactoring feature flags (Phase 2)
    use_new_pipeline: bool = False
    use_new_playback_queue: bool = False

    # Refactoring feature flags (Phase 3)
    use_new_sink_manager: bool = False
    use_new_wake_processor: bool = False

    # Refactoring feature flags (Phase 4)
    use_new_greeting_service: bool = False
    use_new_speaker_tracker: bool = False
```

#### Feature Flag Usage Pattern

```python
# Pattern 1: Simple toggle (groq_client.py)
async def _execute_tool(self, tool_call) -> str:
    if self.settings.use_new_tool_registry:
        # New path: use registry
        handler = self._tool_registry.get(tool_call.function.name)
        if handler:
            result = await handler.execute(args)
            return result.output if result.success else f"Error: {result.error}"
        return f"Unknown tool: {tool_call.function.name}"
    else:
        # Old path: inline switch statement
        function_name = tool_call.function.name
        if function_name == "web_search":
            # ... existing code ...

# Pattern 2: Component injection (voice_session.py)
def __init__(self, ...):
    if self.settings.use_new_state_machine:
        self._state_machine = SessionStateMachine()
        self._register_state_hooks()
    else:
        self._state = "idle"  # Old string-based state

# Pattern 3: Conditional delegation (voice_sink.py)
def write(self, data, user):
    if self.settings.use_new_sink_manager:
        self._user_manager.process_audio(user_id, audio_16k)
    else:
        # Old inline per-user dict management
        if user_id not in self._per_user_detectors:
            # ...
```

#### Flag Activation Sequence

Enable flags one at a time, test between each:

```bash
# .env file - Phase 1 rollout
USE_NEW_TOOL_REGISTRY=true      # Enable first, test
USE_NEW_PROMPT_BUILDER=true     # Enable second, test
USE_NEW_STATE_MACHINE=true      # Enable third, test

# Once all Phase 1 flags work, remove old code and flags
# Then proceed to Phase 2 flags
```

#### Flag Removal (End of Each Phase)

After a phase is stable with all flags enabled:

1. Remove the feature flag checks
2. Remove the old code paths
3. Remove the flag from Settings
4. Commit as "Remove Phase N feature flags, old code paths"

### Incremental Verification

After each extraction:

1. **Unit test the new module** (isolated)
2. **Toggle flag on** in .env
3. **Run bot manually** - voice interaction test
4. **Verify logs** show new code path being used
5. **Toggle flag off** - verify old path still works
6. **Commit** with clear message

### Rollback Procedure

If something breaks after enabling a flag:

```bash
# 1. Immediately disable the flag
# .env
USE_NEW_TOOL_REGISTRY=false

# 2. Restart the bot
# Bot now uses old code path

# 3. Debug at leisure
# Fix the new module, re-enable flag when ready
```

### Edge Case: Flag-Dependent Tests

Some tests should run both paths:

```python
@pytest.mark.parametrize("use_new_registry", [True, False])
async def test_tool_execution(use_new_registry, settings):
    settings.use_new_tool_registry = use_new_registry
    # ... test that works with either path ...
```

---

## Testing Strategy

### Unit Tests (New)

Each extracted module gets unit tests:

```python
# tests/tools/test_search_handler.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.handlers.search import WebSearchHandler
from src.tools.base import ToolResult

@pytest.fixture
def mock_search_service():
    service = MagicMock()
    service.search = AsyncMock(return_value=[
        {"title": "Result 1", "snippet": "..."},
    ])
    return service

@pytest.fixture
def handler(mock_search_service):
    return WebSearchHandler(mock_search_service, max_searches=3)

class TestWebSearchHandler:
    async def test_successful_search(self, handler):
        result = await handler.execute({"query": "test query"})

        assert result.success is True
        assert "Result 1" in result.output

    async def test_empty_query_fails(self, handler):
        result = await handler.execute({"query": ""})

        assert result.success is False
        assert "No query" in result.error

    async def test_search_limit_enforced(self, handler):
        # Use up limit
        for _ in range(3):
            await handler.execute({"query": "test"})

        # Fourth should fail
        result = await handler.execute({"query": "test"})

        assert result.success is False
        assert "limit" in result.error.lower()
```

### Integration Tests

Test module interactions:

```python
# tests/pipeline/test_voice_pipeline.py
class TestVoicePipeline:
    async def test_full_pipeline(self, pipeline, mock_audio):
        context = PipelineContext(speaker_name="Test User")

        await pipeline.process(mock_audio, context)

        # Verify all stages were called
        pipeline._transcription.process.assert_called_once()
        pipeline._response.generate_stream.assert_called_once()
        pipeline._tts.process_stream.assert_called_once()

    async def test_cancellation(self, pipeline, mock_audio):
        # Start processing in background
        task = asyncio.create_task(
            pipeline.process(mock_audio, PipelineContext())
        )

        # Cancel mid-processing
        await asyncio.sleep(0.1)
        await pipeline.cancel()

        # Should complete without error
        await task

        assert pipeline._cancelled is True
```

### Test Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| core/state_machine.py | 95% |
| tools/* | 90% |
| pipeline/* | 85% |
| sink/* | 80% |
| session/* | 80% |

---

## Risk Assessment

> **IMPORTANT**: Before starting any phase, read [refactor-edge-cases.md](refactor-edge-cases.md) which documents 25+ edge cases that MUST be preserved.

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Async race conditions in pipeline | Medium | High | Careful task tracking, extensive testing |
| Wake word timing regression | Medium | Medium | Preserve exact timing logic, A/B testing |
| State machine edge cases | Low | Medium | Exhaustive transition testing |
| Performance regression | Low | Low | Profile before/after |
| Thread safety violations | Medium | High | Use `call_soon_threadsafe()`, test callbacks |
| Lost audio ordering | Medium | High | Preserve sentence numbering pattern |

### Critical Patterns to Preserve

These patterns are documented in detail in `refactor-edge-cases.md`:

| Pattern | Location | Why It Matters |
|---------|----------|----------------|
| `call_soon_threadsafe()` | Sink callbacks | Thread safety |
| Wake word user capture at detection | `voice_sink.py:526` | Correct speaker isolation |
| PTT timeout commitment | `voice_sink.py:264-281` | No flip-flop |
| Sentence ordering with `pending` dict | `voice_session.py:881-886` | Correct playback order |
| Early flag setting | `voice_session.py:586` | No duplicate acks |
| Tool counter reset per-request | `groq_client.py:295-296` | Search limits work |
| Cleanup order (consumers before producers) | `voice_session.py:341-398` | Clean shutdown |
| Finally block sentinel | `voice_session.py:1134-1145` | No stuck tasks |

### Process Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | Medium | Medium | Strict phase boundaries |
| Breaking changes mid-phase | Low | High | Feature flags, parallel code paths |
| Incomplete extraction | Medium | Low | Clear module boundaries, review checklist |
| Edge case regression | Medium | High | Write tests for edge cases FIRST |

### Rollback Plan

Each phase maintains ability to rollback:

1. **Phase 1**: Remove tool registry injection, restore inline handlers
2. **Phase 2**: Remove pipeline injection, restore inline processing
3. **Phase 3**: Restore per-user dicts in sink
4. **Phase 4**: Restore inline session logic

**Quick rollback via feature flags:**
```bash
# Disable all new code, use old paths
USE_NEW_TOOL_REGISTRY=false
USE_NEW_PIPELINE=false
USE_NEW_SINK_MANAGER=false
# Restart bot - immediately using old code
```

---

## Success Criteria

### Quantitative

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| voice_session.py lines | 1,212 | ~150 | <200 |
| groq_client.py lines | 997 | ~180 | <250 |
| voice_sink.py lines | 604 | ~150 | <200 |
| Max method length | 254 | ~50 | <60 |
| Number of modules | 3 god classes | 25+ focused | 20+ |
| Test coverage | 0% | >80% | >75% |

### Qualitative

- [ ] **Understandability**: New developer can understand any module in <10 minutes
- [ ] **Modifiability**: Adding new tool requires only new handler file
- [ ] **Testability**: All modules testable without Discord connection
- [ ] **Maintainability**: Bug fixes isolated to single module

### Feature Enablement

After refactoring, these features become straightforward:

- [ ] **Barge-in**: Cancel pipeline + state transition (~10 lines)
- [ ] **Multiple wake words**: Configure in WakeProcessor
- [ ] **Alternative TTS**: Swap TTSStage implementation
- [ ] **Conversation memory**: Add to PipelineContext
- [ ] **Rate limiting**: Add to ToolExecutor

---

## Appendix: File-by-File Changes

### Files to Create (25 new files)

```
src/core/__init__.py
src/core/state_machine.py
src/core/events.py
src/core/types.py

src/pipeline/__init__.py (update existing)
src/pipeline/base.py
src/pipeline/voice_pipeline.py
src/pipeline/transcription_stage.py
src/pipeline/response_stage.py
src/pipeline/tts_stage.py

src/audio/buffer_manager.py
src/audio/playback_queue.py

src/tools/__init__.py
src/tools/registry.py
src/tools/base.py
src/tools/executor.py
src/tools/definitions.py
src/tools/handlers/__init__.py
src/tools/handlers/search.py
src/tools/handlers/image.py
src/tools/handlers/discord.py
src/tools/handlers/research.py

src/llm/chat_orchestrator.py
src/llm/prompt_builder.py
src/llm/conversation.py

src/session/__init__.py
src/session/voice_session.py (refactored)
src/session/speaker_tracker.py
src/session/greeting_service.py
src/session/phrases.py

src/sink/__init__.py
src/sink/streaming_sink.py (refactored)
src/sink/user_audio_state.py
src/sink/per_user_manager.py
src/sink/wake_processor.py

src/bot/voice_client_adapter.py
```

### Files to Modify

```
src/bot/voice_session.py → src/session/voice_session.py (move + refactor)
src/bot/voice_sink.py → src/sink/streaming_sink.py (move + refactor)
src/llm/groq_client.py (major refactor)
src/llm/llm_factory.py (minor updates)
src/main.py (minor updates for new paths)
```

### Files Unchanged

```
src/bot/discord_bot.py
src/bot/discord_service.py
src/bot/speaker_manager.py
src/bot/voice_source.py
src/llm/base.py
src/llm/claude_client.py
src/detection/*
src/tts/*
src/stt/*
src/search/*
src/image/*
src/audio/resampler.py
src/config.py
```

---

## Next Steps

1. **Review this document** with stakeholders
2. **Prioritize phases** based on pain points
3. **Begin Phase 1** with state machine extraction
4. **Set up test infrastructure** before writing new code
5. **Iterate** with regular verification checkpoints
