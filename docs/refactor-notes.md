# Architecture Refactoring Progress

**Started**: 2026-01-26
**Current Phase**: 3 - Per-User Audio
**Branch**: refactor/phase1-core-abstractions (all phases on same branch)

## Workflow Decision

**Approach**: Complete all 4 phases on a single branch before merging to master.
- Feature flags remain enabled throughout all phases
- Old code removal deferred until all phases complete
- Single merge to master after full refactor is verified

## Quick Status

| Phase | Status | Tested | Notes |
|-------|--------|--------|-------|
| 1. Core Abstractions | COMPLETE | Yes | All tools verified working |
| 2. Pipeline Extraction | COMPLETE | Yes | 458 tests passing |
| 3. Per-User Audio | COMPLETE | Yes | 531 tests passing |
| 4. Session Cleanup | Not Started | - | - |
| Final Cleanup | Pending | - | Remove old code, merge to master |

## Phase 1: Core Abstractions

### Completed Steps
- [x] Step 1.0: Create feature branch
- [x] Step 1.1: Create directories
- [x] Step 1.2: core/state_machine.py (28 tests)
- [x] Step 1.3: tools/base.py, tools/registry.py (21 tests)
- [x] Step 1.4.1: tools/handlers/search.py (28 tests)
- [x] Step 1.4.2: tools/handlers/discord.py (36 tests)
- [x] Step 1.4.3: tools/handlers/image.py (28 tests)
- [x] Step 1.4.4: tools/handlers/research.py (23 tests)
- [x] Step 1.5: tools/executor.py (23 tests)
- [x] Step 1.6: llm/system_prompt_builder.py (38 tests)
- [x] Step 1.7: tools/definitions.py
- [x] Step 1.8: Wire up with feature flags
- [x] Step 1.9: Integration testing (14 tests)
- [x] Step 1.10: Production testing verified
- [ ] Step 1.11: Remove old code (DEFERRED - after all phases complete)

### Notes
- Started refactoring on 2026-01-26
- Following plan in docs/architecture-refactor-plan.md
- Edge cases documented in docs/refactor-edge-cases.md
- **Total tests: 239 passing**

### Feature Flags Wired Up (Step 1.8)
Added to `src/config.py`:
- `USE_NEW_TOOL_REGISTRY` - enables modular tool system
- `USE_NEW_PROMPT_BUILDER` - enables new prompt builder

Modified `src/llm/groq_client.py`:
- Added `tool_registry` parameter to `__init__`
- Creates `ToolExecutor` when feature flag enabled
- Creates `SystemPromptBuilder` when feature flag enabled
- `_execute_tool()` routes to executor when enabled
- `_build_system_prompt()` routes to builder when enabled
- `get_definitions()` used for available tools when enabled
- `reset_counters()` called at request start when enabled

### Integration Testing (Step 1.9)
Created `tests/llm/test_groq_feature_flags.py` with 14 tests:
- Feature flag disabled tests (3 tests)
- Feature flag enabled tests (4 tests)
- Tool execution routing tests (4 tests)
- Available tools routing tests (2 tests)
- Counter reset tests (1 test)

### Production Testing (COMPLETE)

**Date**: 2026-01-26
**Result**: All tools verified working

Tested tools:
- [x] web_search - search queries working
- [x] generate_image - image generation working
- [x] edit_image - image editing working
- [x] deep_research - Claude delegation working
- [x] tag_user - Discord mentions working
- [x] post_table - table formatting working

**Issues Found & Fixed**:
1. `llm_factory.py` wasn't passing `tool_registry` to GroqClient - fixed
2. `.env` inline comments broke boolean parsing - user fixed

### Files Created
- [x] src/core/__init__.py
- [x] src/core/state_machine.py (~180 lines)
- [x] src/tools/__init__.py
- [x] src/tools/base.py (~80 lines)
- [x] src/tools/registry.py (~120 lines)
- [x] src/tools/executor.py (~150 lines)
- [x] src/tools/definitions.py (~250 lines)
- [x] src/tools/handlers/__init__.py
- [x] src/tools/handlers/search.py (~200 lines)
- [x] src/tools/handlers/image.py (~200 lines)
- [x] src/tools/handlers/discord.py (~300 lines)
- [x] src/tools/handlers/research.py (~200 lines)
- [x] src/llm/system_prompt_builder.py (~280 lines)

### Tests Created
- [x] tests/core/__init__.py
- [x] tests/core/test_state_machine.py (28 tests)
- [x] tests/tools/__init__.py
- [x] tests/tools/test_registry.py (21 tests)
- [x] tests/tools/test_search_handler.py (28 tests)
- [x] tests/tools/test_image_handler.py (28 tests)
- [x] tests/tools/test_discord_handler.py (36 tests)
- [x] tests/tools/test_research_handler.py (23 tests)
- [x] tests/tools/test_executor.py (23 tests)
- [x] tests/llm/__init__.py
- [x] tests/llm/test_system_prompt_builder.py (38 tests)
- [x] tests/llm/test_groq_feature_flags.py (14 tests)

## Phase 2: Pipeline Extraction

### Completed Steps
- [x] Step 2.1: Create directory structure (using existing src/pipeline/)
- [x] Step 2.2: src/audio/playback_queue.py - OrderedPlaybackQueue (~190 lines)
- [x] Step 2.3: src/pipeline/base.py - PipelineStage ABC, PipelineContext (~60 lines)
- [x] Step 2.4: src/pipeline/transcription_stage.py - STT + wake word stripping (~226 lines)
- [x] Step 2.5: src/llm/chat_orchestrator.py - Tool loop extraction (~424 lines, 22 tests)
- [x] Step 2.6: src/pipeline/response_stage.py - ChatOrchestrator wrapper (~144 lines, 14 tests)
- [x] Step 2.7: src/pipeline/tts_stage.py - Parallel TTS + ordered queue (~242 lines, 18 tests)
- [x] Step 2.8: src/pipeline/voice_pipeline.py - Stage composition (~210 lines, 15 tests)
- [x] Step 2.9: Wire up feature flags
- [x] Step 2.10: Integration testing (458 tests passing)
- [ ] Step 2.11: Remove old code (DEFERRED - after all phases complete)

### Notes
- Extracted pipeline patterns from voice_session.py (~400 lines moved)
- Extracted tool loop from groq_client.py (~300 lines to ChatOrchestrator)
- **Total new tests: 69** (bringing total to 458)

### Feature Flags Wired Up (Step 2.9)
Added to `src/config.py`:
- `USE_NEW_PIPELINE` - enables modular VoicePipeline
- `USE_NEW_CHAT_ORCHESTRATOR` - enables ChatOrchestrator

Modified `src/llm/groq_client.py`:
- Added `chat_orchestrator` property to expose ChatOrchestrator
- Added `get_system_prompt()` method for ResponseStage
- Creates ChatOrchestrator when `USE_NEW_CHAT_ORCHESTRATOR` enabled
- `chat_stream()` delegates to orchestrator when enabled

Modified `src/bot/voice_session.py`:
- Added `_voice_pipeline` attribute
- Creates VoicePipeline in `_init_components()` when flag enabled
- Added `_process_speech_with_pipeline()` method
- `_process_speech()` routes to pipeline when enabled

### Critical Patterns Preserved
- Sentence ordering via OrderedPlaybackQueue pending dict
- Parallel TTS synthesis (tasks spawned per sentence)
- Thinking phrase runs in parallel with LLM
- 150ms inter-sentence pause
- TTS error recovery (queue empty audio on failure)
- Search limit enforcement (3 max)
- Action success tracking for fallback
- Pre-tool content yielding

### Files Created
- [x] src/audio/playback_queue.py (~190 lines)
- [x] src/pipeline/base.py (~60 lines)
- [x] src/pipeline/transcription_stage.py (~226 lines)
- [x] src/pipeline/response_stage.py (~144 lines)
- [x] src/pipeline/tts_stage.py (~242 lines)
- [x] src/pipeline/voice_pipeline.py (~210 lines)
- [x] src/llm/chat_orchestrator.py (~424 lines)

### Tests Created
- [x] tests/audio/test_playback_queue.py (19 tests)
- [x] tests/pipeline/test_transcription_stage.py (17 tests)
- [x] tests/pipeline/test_response_stage.py (14 tests)
- [x] tests/pipeline/test_tts_stage.py (18 tests)
- [x] tests/pipeline/test_voice_pipeline.py (15 tests)
- [x] tests/llm/test_chat_orchestrator.py (22 tests)

### Production Testing
**Status**: Ready for testing
**To test**, set these in `.env`:
```
USE_NEW_PIPELINE=true
USE_NEW_CHAT_ORCHESTRATOR=true
```

Then restart the bot and test:
1. Normal conversation (wake word -> question -> response)
2. Long response (verify sentences play in order)
3. Tool use (web search, image generation)
4. Thinking phrase plays before response
5. Multiple sentences stream correctly

## Phase 3: Per-User Audio

### Completed Steps
- [x] Step 3.1: Create directory structure (src/sink/, tests/sink/)
- [x] Step 3.2: src/sink/user_audio_state.py - Per-user state dataclass (~130 lines, 30 tests)
- [x] Step 3.3: src/sink/per_user_manager.py - Lifecycle management (~120 lines, 22 tests)
- [x] Step 3.4: src/sink/wake_processor.py - Wake detection logic (~150 lines, 21 tests)
- [x] Step 3.5: Add feature flags to config.py
- [x] Step 3.6: Wire up conditionals in voice_sink.py
- [x] Step 3.7: Run all tests (531 tests passing)
- [ ] Step 3.8: Remove old code (DEFERRED - after all phases complete)

### Notes
- Consolidated 4 scattered per-user dictionaries into focused modules
- **Total new tests: 73** (bringing total to 531)

### Feature Flags Wired Up (Step 3.5)
Added to `src/config.py`:
- `USE_NEW_SINK_MANAGER` - enables PerUserAudioManager
- `USE_NEW_WAKE_PROCESSOR` - enables WakeWordProcessor

Modified `src/bot/voice_sink.py`:
- Creates PerUserAudioManager and WakeWordProcessor when flags enabled
- `write()` routes to new components for pre-buffer and wake detection
- `check_idle_audio_gap()` routes to WakeWordProcessor when enabled
- `state.setter` uses processor for idle transition cleanup
- `start_listening()` gets pre-buffer from new manager
- `remove_user()` delegates to manager when enabled
- `cleanup()` calls manager.cleanup_all() when enabled
- Speaker isolation uses processor's wake_word_user_id

### Critical Edge Cases Preserved
| Edge Case | Implementation |
|-----------|---------------|
| Wake word user capture at detection time | WakeWordProcessor.process_audio() sets wake_word_user_id |
| Gap window 250ms-2.5s | wake_processor.py constants: MIN_GAP_SECONDS, MAX_GAP_SECONDS |
| Max 5 silence attempts | wake_processor.py constant: MAX_SILENCE_ATTEMPTS |
| Pre-buffer FIFO eviction (1.5s) | UserAudioState.add_to_pre_buffer() with MAX_PRE_BUFFER_SECONDS |
| 8000 samples (500ms) silence chunk | wake_processor.py constant: SILENCE_SAMPLES |

### Files Created
- [x] src/sink/__init__.py
- [x] src/sink/user_audio_state.py (~130 lines)
- [x] src/sink/per_user_manager.py (~120 lines)
- [x] src/sink/wake_processor.py (~150 lines)

### Tests Created
- [x] tests/sink/__init__.py
- [x] tests/sink/test_user_audio_state.py (30 tests)
- [x] tests/sink/test_per_user_manager.py (22 tests)
- [x] tests/sink/test_wake_processor.py (21 tests)

### Production Testing (COMPLETE)

**Date**: 2026-01-26
**Result**: All features verified working

Tested scenarios:
- [x] Single user wake word detection
- [x] Pre-buffer capture (~0.88s captured)
- [x] Ack playback completes without interruption
- [x] PTT pattern (wake word with audio gap)
- [x] Flush detection from gap

**Issues Found & Fixed**:
1. Ack playback cut off ("brief chirp") - removed `voice_client.stop()` on cancel
2. Detector flush only worked for old code path - added Phase 3 flush logic
3. Ack played while user speaking - added 300ms initial delay before silence check

## Phase 4: Session Cleanup
[Will be filled when Phase 4 starts]

---

## Final Cleanup: Identifying Old Code

When all phases are complete, use these patterns to find old code to remove:

### Phase 1 (Tools/Prompts)
- Search: `if self._tool_executor` in groq_client.py
- Old code: Inline tool execution after the conditional
- Search: `if self._prompt_builder` in groq_client.py
- Old code: Inline prompt building after the conditional

### Phase 2 (Pipeline/Orchestrator)
- Search: `if self._chat_orchestrator` in groq_client.py
- Old code: ~300 lines of inline tool loop after `return`
- Search: `if self._voice_pipeline` in voice_session.py
- Old code: ~150 lines of inline processing after `return`

### Phase 3 (Per-User Audio)
- Search: `if self._settings.use_new_sink_manager` in voice_sink.py
- Old code: Inline per-user dict management in `else` branches
- Old dicts to remove: `_per_user_detectors`, `_per_user_pre_buffers`, `_per_user_last_seen`, `_per_user_silence_attempts`
- Old method to remove: `_get_or_create_detector()`

### Cleanup Steps
1. Remove feature flag conditionals (make new code the default)
2. Delete old code paths after the `return` statements
3. Remove flag definitions from `src/config.py`
4. Remove unused imports
5. Run tests to verify

---

## Issues & Decisions Log

### 2026-01-26 - Refactoring Started
**Context**: Beginning major architecture refactoring of 3 god classes
**Decision**: Use 4-phase approach with feature flags
**Rationale**: Allows safe incremental migration with easy rollback

### 2026-01-26 - Single Branch Workflow
**Context**: Phase 1 production testing complete, deciding merge strategy
**Decision**: Keep all phases on same branch, merge once after Phase 4
**Rationale**:
- Feature flags provide safety net throughout
- Old code removal deferred to final cleanup
- Single merge reduces integration risk
- Easier to test full refactor as a unit

### 2026-01-26 - Phase 2 Pipeline Design
**Context**: Deciding how to wire VoicePipeline with existing components
**Decision**:
- VoicePipeline requires ChatOrchestrator from GroqClient
- `USE_NEW_PIPELINE` implicitly requires `USE_NEW_CHAT_ORCHESTRATOR`
- Exposed `chat_orchestrator` property and `get_system_prompt()` method on GroqClient
**Rationale**:
- Avoids duplicating orchestrator logic
- ResponseStage wraps existing ChatOrchestrator
- System prompt comes from LLM client for consistency

---

## Resume Instructions

If resuming work after a break:

1. Check current branch: `git branch`
2. Read this file to see current status
3. Read the plan: `docs/architecture-refactor-plan.md`
4. Find the next unchecked step
5. Continue implementation
