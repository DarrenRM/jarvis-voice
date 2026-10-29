# Refactoring Edge Cases Analysis

**Created**: 2026-01-26
**Purpose**: Document subtle behaviors, edge cases, and timing-sensitive logic that MUST be preserved during refactoring.

---

## Table of Contents

1. [Thread Safety & Cross-Thread Communication](#1-thread-safety--cross-thread-communication)
2. [Timing-Sensitive Logic](#2-timing-sensitive-logic)
3. [State Coupling & Flag Dependencies](#3-state-coupling--flag-dependencies)
4. [Race Condition Mitigations (Already Implemented)](#4-race-condition-mitigations-already-implemented)
5. [Error Recovery Paths](#5-error-recovery-paths)
6. [Cleanup Sequences](#6-cleanup-sequences)
7. [Audio Pipeline Edge Cases](#7-audio-pipeline-edge-cases)
8. [Tool Execution Edge Cases](#8-tool-execution-edge-cases)
9. [Speaker Isolation Edge Cases](#9-speaker-isolation-edge-cases)
10. [LLM Streaming Edge Cases](#10-llm-streaming-edge-cases)

---

## 1. Thread Safety & Cross-Thread Communication

### 1.1 Sink Callbacks Run in Discord Thread

**Location**: `voice_sink.py:524-533`, `voice_session.py:226-233, 282-294`

**Current Behavior**:
- `StreamingSink.write()` is called by Discord's audio thread (NOT asyncio)
- Callbacks `_on_wake_word()` and `_on_speech_end()` are invoked from this thread
- Session uses `self._loop.call_soon_threadsafe()` to safely signal asyncio events

**Code**:
```python
# voice_session.py:229-231
def _on_wake_word(self):
    if self._state == "idle" and self._loop:
        self._loop.call_soon_threadsafe(self._wake_event.set)
```

**CRITICAL**: Any refactoring MUST preserve:
1. Callbacks are sync functions (not async)
2. Use `call_soon_threadsafe()` to communicate with asyncio loop
3. State checks (`self._state == "idle"`) happen atomically in callback
4. `self._loop` must be set before callbacks can fire

**Regression Risk**: If pipeline stages register their own callbacks, they must also use thread-safe signaling.

---

### 1.2 Pending Audio Assignment from Sink Thread

**Location**: `voice_session.py:287`, `voice_sink.py:574-576`

**Current Behavior**:
- `_on_speech_end()` sets `self._pending_audio = audio` from Discord thread
- Main loop reads `self._pending_audio` from asyncio thread
- No lock used - relies on Python's GIL for atomic reference assignment

**Code**:
```python
# voice_session.py:287
self._pending_audio = audio  # Set from Discord thread

# voice_session.py:514
audio = self._pending_audio  # Read from asyncio thread
```

**CRITICAL**: This works because:
1. Reference assignment is atomic under GIL
2. Reading happens AFTER event is set (memory barrier via `call_soon_threadsafe`)
3. Only one writer (sink thread) and one reader (session loop)

**Regression Risk**: If buffer management is extracted, maintain single-writer-single-reader pattern.

---

### 1.3 Speaker Resolution from Sink Thread

**Location**: `voice_session.py:288-290`

**Current Behavior**:
- `_resolve_current_speaker()` is called FROM the Discord thread inside `_on_speech_end()`
- This accesses `self.voice_client.guild` which is a Discord object

**Code**:
```python
def _on_speech_end(self, audio: np.ndarray, user_id: Optional[int] = None):
    if self._state == "listening" and self._loop:
        self._pending_audio = audio
        self._resolve_current_speaker(user_id)  # <-- Called from Discord thread!
```

**CRITICAL**: Discord objects may not be fully thread-safe. Current code works because:
1. Guild lookup is read-only
2. SpeakerManager does simple dict lookups

**Regression Risk**: If SpeakerTracker is extracted, ensure it doesn't do Discord API calls from sink thread.

---

## 2. Timing-Sensitive Logic

### 2.1 Dynamic Deadline Extension

**Location**: `voice_session.py:470-475, 650-655`

**Current Behavior**:
- Initial listening timeout is 10 seconds
- Deadline extends while `sink.is_speech_active()` returns True
- Maximum extension capped at 30 seconds (`settings.max_question_duration`)

**Code**:
```python
# voice_session.py:470-475
if self.sink.is_speech_active():
    new_deadline = time.perf_counter() + initial_timeout
    if new_deadline > deadline and new_deadline <= final_deadline:
        deadline = new_deadline
```

**CRITICAL**: The extension logic must:
1. Only extend if speech is STILL active (VAD says so)
2. Never exceed `final_deadline` (30s safety cap)
3. Poll at 200ms intervals (not too fast, not too slow)

**Regression Risk**: If state machine handles timeouts, preserve dynamic extension logic.

---

### 2.2 PTT (Push-to-Talk) Audio Timeout

**Location**: `voice_sink.py:257-283`, `voice_session.py:477-486`

**Current Behavior**:
- Tracks `_last_audio_time` when packets arrive
- If no packets for 500ms during listening, assumes PTT released
- `_audio_timeout_detected` flag prevents flip-flopping

**Code**:
```python
# voice_sink.py:264-265, 281
if self._audio_timeout_detected:
    return True  # Commit to decision once made
# ...
self._audio_timeout_detected = True  # Commit to this decision
```

**CRITICAL**:
1. 500ms threshold balances responsiveness vs false positives
2. Once timeout detected, COMMIT to it (no flip-flopping)
3. Only check during `listening` state with `_speech_started`

**Regression Risk**: Audio buffer manager must preserve commitment pattern.

---

### 2.3 Wake Word Silence Padding

**Location**: `voice_sink.py:315-380`

**Current Behavior**:
- When idle and audio gap detected (250ms-2.5s), feed silence to wake detector
- This completes pending detections (OpenWakeWord is a streaming model)
- Limited to 5 attempts per user per gap

**Code**:
```python
# voice_sink.py:360-364
attempts = self._per_user_silence_attempts.get(user_id, 0)
if attempts >= 5:
    continue  # Give up after 5 attempts for this user
```

**CRITICAL**:
1. Gap window is 250ms-2.5s (too short = false positives, too long = stale)
2. Feed 500ms of silence (8000 samples at 16kHz)
3. Max 5 attempts prevents infinite loops
4. Reset attempts when real audio arrives

**Regression Risk**: WakeWordProcessor must preserve attempt limiting and gap windowing.

---

### 2.4 Silence Wait with Debounce Modes

**Location**: `voice_session.py:533-565`

**Current Behavior**:
- Two modes: `use_debounce=True` (3s after speech ends) or `use_debounce=False` (immediate)
- Debounce mode used for response playback (wait for conversation to settle)
- Immediate mode used for ack playback (quick check)

**Code**:
```python
if use_debounce:
    # Full channel silence (3s debounce) - for responses
    if not self.sink.is_anyone_speaking():
        return True
else:
    # Immediate check (no debounce) - for ack
    if not self.sink.is_anyone_actively_speaking():
        return True
```

**CRITICAL**: Different use cases need different timing:
- Ack: Quick check, play ASAP
- Response: Wait for conversation to settle

**Regression Risk**: Pipeline stages must use correct silence mode for their purpose.

---

### 2.5 Ack Played Flag Race Prevention

**Location**: `voice_session.py:586, 619-627`

**Current Behavior**:
- `_ack_played` flag set EARLY (before TTS synthesis completes)
- Prevents duplicate ack if fast transcription finishes first
- `_handle_wake_word_only()` checks this flag

**Code**:
```python
# voice_session.py:586
self._ack_played = True  # Mark early - prevents race with fast transcription

# voice_session.py:619-620
if not self._ack_played:
    # Play ack
```

**CRITICAL**: Set flag BEFORE async TTS call, not after, to prevent race.

**Regression Risk**: If ack logic moves to WakeHandler, preserve early flag setting.

---

### 2.6 Thinking Phrase Parallel Execution

**Location**: `voice_session.py:1055-1064, 889-899`

**Current Behavior**:
- Thinking phrase TTS runs as background task
- LLM request starts IMMEDIATELY (parallel with thinking phrase)
- Playback consumer waits for thinking task before first sentence

**Code**:
```python
# voice_session.py:1062
self._thinking_task = asyncio.create_task(self._play_thinking_phrase(phrase))

# voice_session.py:889-899
if not waited_for_thinking and self._thinking_task is not None:
    await self._thinking_task
    waited_for_thinking = True
```

**CRITICAL**:
1. LLM must start BEFORE thinking phrase completes (parallel)
2. Playback must wait for thinking BEFORE first response sentence
3. Only wait once (`waited_for_thinking` flag)

**Regression Risk**: Pipeline must preserve parallel thinking + LLM pattern.

---

## 3. State Coupling & Flag Dependencies

### 3.1 Session State + Sink State Coupling

**Location**: `voice_session.py:407-410, 437`

**Current Behavior**:
- Session maintains `self._state` (idle/listening/processing)
- Sink maintains `self._state` (idle/listening)
- Both must be synchronized on transitions

**Code**:
```python
# voice_session.py:407-410
self.state = "idle"
self._wake_event.clear()
self._speech_end_event.clear()
self.sink.state = "idle"  # <-- Must sync!

# voice_session.py:437
self.sink.start_listening()  # Sets sink state
```

**CRITICAL**: Session and sink states must stay synchronized:
- `session.idle` ↔ `sink.idle`
- `session.listening` ↔ `sink.listening`
- `session.processing` → sink stays in last state (not changed)

**Regression Risk**: State machine must control both session AND sink state.

---

### 3.2 Speech Started + Buffer Non-Empty

**Location**: `voice_sink.py:565-580`

**Current Behavior**:
- `_speech_started` flag tracks if VAD detected speech
- Speech end only fires if `_speech_started` was True AND buffer has audio
- Prevents firing on pure silence

**Code**:
```python
elif endpoint == "end" and self._speech_started:
    # Only fire callback if we actually had speech
    self._speech_started = False
    if self._on_speech_end:
        audio = self.get_audio_buffer()
        self._on_speech_end(audio, self._last_user_id)
```

**CRITICAL**: Guard against false speech-end events.

---

### 3.3 Greeting Cooldown + State Recheck

**Location**: `voice_session.py:740-764`

**Current Behavior**:
- Greeting checks state BEFORE waiting for silence
- Greeting checks state AGAIN after waiting (state may have changed)
- Cooldown prevents re-greeting same user within 5 minutes

**Code**:
```python
# voice_session.py:736-737
if self.state != "idle":
    return  # Check 1

# ... wait for silence (may take seconds) ...

# voice_session.py:761-763
if self.state != "idle":
    logger.info("[GREETING] Skipping - bot became busy during wait")
    return  # Check 2
```

**CRITICAL**: State can change during async waits - always recheck.

**Regression Risk**: GreetingService must handle async state changes.

---

### 3.4 Tool Call Counter Reset Per Request

**Location**: `groq_client.py:295-296`

**Current Behavior**:
- `_tool_calls_this_request` and `_search_calls_this_request` reset at start of each `chat_stream()`
- Counters persist across tool loop iterations within same request
- Reset ensures fresh limits for next user question

**Code**:
```python
# groq_client.py:295-296
self._tool_calls_this_request = []
self._search_calls_this_request = 0
```

**CRITICAL**: Counters must reset per-request, not per-tool-call.

**Regression Risk**: ToolExecutor must reset counters at request start.

---

### 3.5 Wake Word User ID Capture Timing

**Location**: `voice_sink.py:116, 208, 461, 526, 537-546`

**Current Behavior**:
- `_wake_word_user_id` captured at detection time (NOT at `start_listening()` time)
- This prevents race condition where another user's packet arrives between detection and start
- Speaker isolation uses this ID, not `_last_user_id`

**Code**:
```python
# voice_sink.py:526 - Detection time capture
self._wake_word_user_id = user_id

# voice_sink.py:537-539 - Isolation uses captured ID
if self._speaker_isolation_enabled and self._wake_word_user_id is not None and user_id != self._wake_word_user_id:
    # Filter this audio
```

**CRITICAL**: Capture wake word user ID at DETECTION time, not later.

**Regression Risk**: WakeWordProcessor must capture and expose user ID at detection.

---

## 4. Race Condition Mitigations (Already Implemented)

### 4.1 Wake Word User vs Last User Race

**Problem**: Between wake word detection and `start_listening()`, another user's packet might arrive and update `_last_user_id`.

**Solution** (implemented): Capture `_wake_word_user_id` at detection time, use for isolation.

**Location**: `voice_sink.py:116, 206-221, 526`

**MUST PRESERVE**: Detection-time capture of user ID.

---

### 4.2 Ack vs Fast Transcription Race

**Problem**: If STT is very fast, transcription might complete before ack finishes synthesizing.

**Solution** (implemented): Set `_ack_played = True` BEFORE starting TTS, not after.

**Location**: `voice_session.py:586`

**MUST PRESERVE**: Early flag setting.

---

### 4.3 Multiple Tool Calls in Same Response

**Problem**: Model might request multiple tool calls; executing them sequentially could cause issues.

**Solution** (implemented): `parallel_tool_calls=False` in API request.

**Location**: `groq_client.py:338`

**Code**:
```python
api_params["parallel_tool_calls"] = False
```

**MUST PRESERVE**: Sequential tool execution.

---

### 4.4 Audio Queue Ordering Race

**Problem**: Parallel TTS synthesis might complete out of order (sentence 3 before sentence 2).

**Solution** (implemented): Queue stores `(sentence_num, audio)` tuples; consumer plays in order using `pending` dict.

**Location**: `voice_session.py:861-912`

**Code**:
```python
# voice_session.py:881-886
sentence_num, audio = item
pending[sentence_num] = audio
while next_to_play in pending:
    audio = pending.pop(next_to_play)
    # play...
    next_to_play += 1
```

**MUST PRESERVE**: Sentence numbering and ordered playback.

---

### 4.5 Playback Task Cleanup on Error

**Problem**: If LLM streaming errors, playback task might hang waiting for `None` sentinel.

**Solution** (implemented): `finally` block ensures `None` is always queued.

**Location**: `voice_session.py:1134-1145`

**Code**:
```python
finally:
    # Wait for TTS tasks
    if self._tts_tasks:
        await asyncio.gather(*self._tts_tasks, return_exceptions=True)
    # Signal end
    await self._audio_queue.put(None)
    if self._playback_task:
        await self._playback_task
```

**MUST PRESERVE**: Cleanup in finally block, sentinel always sent.

---

## 5. Error Recovery Paths

### 5.1 Groq Output Parsing Failure

**Location**: `groq_client.py:486-494`

**Current Behavior**:
- Groq sometimes outputs thinking text instead of proper tool call
- Error contains "output_parse_failed" or "Parsing failed"
- If an action tool (image, tag) succeeded before error, use that result
- Otherwise, give generic fallback

**Code**:
```python
if "output_parse_failed" in error_str:
    if last_action_success:
        yield last_action_success
    else:
        yield "I couldn't find specific information..."
```

**CRITICAL**: Check for `last_action_success` before using generic fallback.

---

### 5.2 Search Limit Reached Recovery

**Location**: `groq_client.py:309-326, 343-391`

**Current Behavior**:
- After 3 search calls, disable search tools but keep action tools
- Inject system message forcing model to answer with gathered info
- If model still tries to search, extract tool results and summarize

**Code**:
```python
# voice_session.py:315-318
if search_limit_reached:
    available_tools = [t for t in TOOLS if t["function"]["name"] in ACTION_TOOLS]
```

**CRITICAL**: Don't just fail - force model to use gathered info.

---

### 5.3 Tool Execution Error Recovery

**Location**: `groq_client.py:745-747`

**Current Behavior**:
- Tool execution wrapped in try/except
- On error, return `"Tool error: {str(e)}"` to model
- Model can then apologize or retry differently

**CRITICAL**: Never let tool error crash the request - return error message to model.

---

### 5.4 Deep Research Timeout

**Location**: `groq_client.py:718-727`

**Current Behavior**:
- Deep research (Claude delegation) has configurable timeout
- 90s if Claude needs to search, 45s if just synthesizing
- On timeout, return user-friendly message

**Code**:
```python
timeout_secs = 90 if use_tools else 45
try:
    response = await asyncio.wait_for(
        self._claude_service.chat(prompt, use_tools=use_tools),
        timeout=timeout_secs
    )
except asyncio.TimeoutError:
    return "Deep research took too long. Try a simpler question."
```

**CRITICAL**: Don't let deep research hang forever.

---

### 5.5 TTS Synthesis Error Recovery

**Location**: `voice_session.py:950-953`

**Current Behavior**:
- If TTS synthesis fails for a sentence, queue empty audio
- This prevents blocking the ordered playback (which waits for sentence N)

**Code**:
```python
except Exception as e:
    logger.error(f"[TTS] Error synthesizing sentence {sentence_num}: {e}")
    # Queue empty audio to not block ordering
    await self._audio_queue.put((sentence_num, np.array([], dtype=np.float32)))
```

**CRITICAL**: Always queue something (even empty) to maintain ordering.

---

## 6. Cleanup Sequences

### 6.1 Session Stop Order

**Location**: `voice_session.py:341-398`

**Current Order**:
1. Set `_running = False`
2. Cancel thinking task (await completion)
3. Cancel playback task (await completion)
4. Cancel all TTS tasks (gather with return_exceptions)
5. Stop Discord recording
6. Stop Discord playback
7. Clear speaking state (`speak(False)`)
8. Reset image chat context

**CRITICAL**: Order matters:
- Cancel consumers before producers
- Await cancelled tasks (don't abandon)
- Discord operations may fail (socket dead) - catch and continue

---

### 6.2 Playback Consumer Cleanup

**Location**: `voice_session.py:870-878, 915-917`

**Current Behavior**:
- `None` sentinel signals end of stream
- Consumer plays remaining buffered audio in order before exiting
- CancelledError handled gracefully

**Code**:
```python
if item is None:
    # Play any remaining buffered audio in order
    while next_to_play in pending:
        audio = pending.pop(next_to_play)
        if len(audio) > 0:
            await self._play_audio(audio)
        next_to_play += 1
    break
```

**CRITICAL**: Don't lose queued audio on clean shutdown.

---

### 6.3 Sink State Reset on Idle

**Location**: `voice_sink.py:142-150`

**Current Behavior**:
- When sink transitions to idle:
  - Clear audio buffer
  - Clear wake word user ID
  - Reset ALL per-user wake detectors
  - Clear silence attempt counters

**CRITICAL**: Full reset prevents stale state from previous session.

---

### 6.4 Discord Speaking State Clear

**Location**: `voice_session.py:331-336, 388-393`

**Current Behavior**:
- Clear speaking state on start (`speak(False)`)
- Clear speaking state on stop (`speak(False)`)
- Fixes "stuck green ring" visual bug

**Code**:
```python
try:
    await self.voice_client.ws.speak(False)
except Exception:
    pass  # May fail if websocket not ready
```

**CRITICAL**: Always try to clear, but don't fail if websocket is dead.

---

## 7. Audio Pipeline Edge Cases

### 7.1 Pre-Buffer Transfer on Wake

**Location**: `voice_sink.py:202-226`

**Current Behavior**:
- When wake word detected, transfer ONLY that user's pre-buffer
- Pre-buffer captures question start (e.g., "what's the weather" from "hey jarvis what's the weather")
- Clear ALL pre-buffers after transfer (prevent stale audio)

**Code**:
```python
if wake_user and wake_user in self._per_user_pre_buffers:
    pre_buffer = self._per_user_pre_buffers[wake_user]
    self._audio_buffer = pre_buffer.copy()
# Clear all pre-buffers
self._per_user_pre_buffers.clear()
```

**CRITICAL**: Only transfer wake user's buffer, clear all others.

---

### 7.2 Pre-Buffer Size Limiting

**Location**: `voice_sink.py:510-518`

**Current Behavior**:
- Max 1.5 seconds of pre-buffer per user
- FIFO eviction when limit exceeded
- Prevents memory growth with many users

**Code**:
```python
self._pre_buffer_max_samples = int(SAMPLE_RATE_PROCESSING * 1.5)  # 1.5s
# ...
while total > self._pre_buffer_max_samples and user_buffer:
    removed = user_buffer.pop(0)
    total -= len(removed)
```

**CRITICAL**: FIFO eviction, per-user limits.

---

### 7.3 Audio Gap Detection and Buffer Clear

**Location**: `voice_sink.py:443-454`

**Current Behavior**:
- Large gap (>1s) in idle state clears all pre-buffers
- Prevents stale audio from before PTT press
- Also flushes wake detectors on gap

**Code**:
```python
if gap > 1.0:
    self._per_user_pre_buffers.clear()
```

**CRITICAL**: Clear stale audio on large gaps.

---

### 7.4 Sentence Boundary Detection (Newline + Punctuation)

**Location**: `voice_session.py:1096-1122`

**Current Behavior**:
- Primary: Split on newlines (Claude outputs one sentence per line)
- Secondary: Also queue on sentence-ending punctuation (.!?)
- Handles "I'll look that up." before long tool call

**Code**:
```python
# Primary: newlines
while '\n' in buffer:
    line, buffer = buffer.split('\n', 1)
    # queue line...

# Secondary: punctuation
stripped = buffer.strip()
if stripped and stripped[-1] in '.!?':
    # queue stripped, clear buffer
```

**CRITICAL**: Both splitting mechanisms needed for different scenarios.

---

### 7.5 Inter-Sentence Pause

**Location**: `voice_session.py:850`

**Current Behavior**:
- 150ms pause after each sentence playback
- Creates natural speech rhythm

**Code**:
```python
await asyncio.sleep(0.15)  # 150ms gap for natural sentence pacing
```

**CRITICAL**: Small pause prevents sentences running together.

---

## 8. Tool Execution Edge Cases

### 8.1 Tool Name Aliasing (Hallucination Fix)

**Location**: `groq_client.py:184-188, 426, 626`

**Current Behavior**:
- Model sometimes hallucinates tool names like "web_fetch_url"
- Aliases mapped to real names before execution

**Code**:
```python
TOOL_ALIASES = {
    "web_fetch_url": "fetch_url",
    "web_fetch": "fetch_url",
}
# ...
tool_name = TOOL_ALIASES.get(tool_name, tool_name)
```

**CRITICAL**: Normalize before execution AND tracking.

---

### 8.2 Image Edit Without Previous Image

**Location**: `groq_client.py:656-657`

**Current Behavior**:
- Check `_image_service.has_chat()` before allowing edit
- Return helpful error if no previous image

**Code**:
```python
if not self._image_service.has_chat():
    return "No previous image to edit. Ask the user what image they'd like you to generate first."
```

**CRITICAL**: Graceful error, not crash.

---

### 8.3 Action Tool Success Tracking

**Location**: `groq_client.py:457-461`

**Current Behavior**:
- Track successful action tool results (starts with "Done!")
- Used for fallback if model output parsing fails later

**Code**:
```python
if tool_name in ACTION_TOOLS and result.startswith("Done!"):
    success_msg = result.split(".")[0] + "."
    last_action_success = success_msg
```

**CRITICAL**: Preserve successful results for error recovery.

---

### 8.4 Deep Research Tool Notification

**Location**: `groq_client.py:429-430`

**Current Behavior**:
- Yield notification BEFORE deep research execution
- User hears "This one needs some deeper digging" while waiting

**Code**:
```python
if tool_name == "deep_research":
    yield "[thoughtful] This one needs some deeper digging, give me a moment.\n"
```

**CRITICAL**: User feedback before long operation.

---

### 8.5 Post-Tool Content Yield

**Location**: `groq_client.py:401-404`

**Current Behavior**:
- If model says something BEFORE tool call, yield it
- Ensures "I'll look that up" gets spoken before search

**Code**:
```python
if response_message.content:
    yield response_message.content
    if not response_message.content.endswith('\n'):
        yield '\n'
```

**CRITICAL**: Pre-tool content must be yielded.

---

## 9. Speaker Isolation Edge Cases

### 9.1 Isolation Based on Wake Word User

**Location**: `voice_sink.py:537-546`

**Current Behavior**:
- Use `_wake_word_user_id` (captured at detection), not `_last_user_id`
- Filter audio from other users during listening
- Log filtered users once per session

**Code**:
```python
if self._speaker_isolation_enabled and self._wake_word_user_id is not None and user_id != self._wake_word_user_id:
    if user_id not in self._filtered_users:
        logger.info(f"[SINK ISOLATION] Filtering audio from user {user_id}")
        self._filtered_users.add(user_id)
    return  # Don't buffer
```

**CRITICAL**: Use captured wake word user, not last user.

---

### 9.2 PTT Timeout Still Works with Isolation

**Location**: `voice_sink.py:469-473`

**Current Behavior**:
- `_last_audio_time` only updates for wake word user during listening
- Other users' audio doesn't reset PTT timeout
- Prevents other users from keeping session alive

**Code**:
```python
if self._state == "idle":
    self._last_audio_time = now
elif not self._speaker_isolation_enabled or self._last_user_id is None or user_id == self._last_user_id:
    self._last_audio_time = now
# else: wrong user - don't update (will trigger PTT timeout)
```

**CRITICAL**: Only wake user's audio extends timeout.

---

### 9.3 Speaking Detector Still Processes All Users

**Location**: `voice_sink.py:545-546`

**Current Behavior**:
- Even when filtering audio from non-wake users, still process through speaking detector
- This maintains awareness of who's talking in channel
- Comment: "Still process through speaking detector for awareness"

**CRITICAL**: Isolation affects buffering, not awareness.

---

## 10. LLM Streaming Edge Cases

### 10.1 Conversation History Truncation

**Location**: `groq_client.py:826-831`

**Current Behavior**:
- Only include last 6 messages (3 exchanges) in context
- Prevents token limit issues with long conversations

**Code**:
```python
history_text = "\n".join(
    f"{msg['role'].title()}: {msg['content']}"
    for msg in self._conversation_history[-6:]
)
```

**CRITICAL**: Truncate to prevent token overflow.

---

### 10.2 Tool Choice None on Final Response

**Location**: `groq_client.py:599`

**Current Behavior**:
- After tool execution, final response uses `tool_choice="none"`
- Prevents "model called a tool" error on streaming response

**Code**:
```python
stream = await self._client.chat.completions.create(
    # ...
    tool_choice="none",  # Explicitly disable
)
```

**CRITICAL**: Disable tools for final streaming response.

---

### 10.3 System Message Injection for Forced Answer

**Location**: `groq_client.py:321-326`

**Current Behavior**:
- When search limit reached, inject system message
- Only inject once (check for duplicate)
- Message forces model to answer or use action tool

**Code**:
```python
force_answer_msg = "You have gathered enough search information..."
if not any(m.get("content") == force_answer_msg for m in messages):
    messages.append({"role": "system", "content": force_answer_msg})
```

**CRITICAL**: Don't duplicate injection, check first.

---

### 10.4 Streaming vs Non-Streaming for Tool Detection

**Location**: `groq_client.py:333`

**Current Behavior**:
- Use `stream=False` for tool detection turns
- Use `stream=True` only for final response
- Tool calls can't be detected in streaming mode

**Code**:
```python
api_params["stream"] = False  # Non-streaming for tool detection
```

**CRITICAL**: Tool detection requires non-streaming.

---

### 10.5 Graceful Failure Message Construction

**Location**: `groq_client.py:529-536`

**Current Behavior**:
- Short message for voice output
- Uses emotion tags
- Doesn't list all searches (too verbose for voice)

**Code**:
```python
def _build_graceful_failure(self) -> str:
    # Keep it SHORT for voice
    return "[surprised] I searched but couldn't find a good answer. Could you try asking differently?"
```

**CRITICAL**: Keep failure message short for TTS.

---

## Summary: High-Risk Refactoring Areas

### Must Preserve Exactly

1. **Thread-safe callbacks** (`call_soon_threadsafe`)
2. **Wake word user capture timing** (at detection, not start_listening)
3. **Dynamic deadline extension** (poll + extend pattern)
4. **PTT timeout commitment** (once detected, stay detected)
5. **Sentence ordering** (number + pending dict pattern)
6. **Cleanup order** (cancel consumers before producers)
7. **Early flag setting** (`_ack_played` before TTS)
8. **Tool counter reset** (per-request, not per-call)

### Test Coverage Recommendations

| Edge Case | Test Type |
|-----------|-----------|
| Thread-safe callbacks | Integration test with mock sink |
| Deadline extension | Unit test with mock VAD |
| PTT timeout | Unit test with simulated packet timing |
| Sentence ordering | Unit test with out-of-order completion |
| Cleanup sequence | Unit test with cancellation |
| Tool limits | Unit test with mock API |
| Speaker isolation | Integration test with multi-user packets |

### Pre-Refactoring Checklist

- [ ] Read this document before each phase
- [ ] Write tests for timing-sensitive code FIRST
- [ ] Verify thread-safety of any new callbacks
- [ ] Test state transitions with edge cases
- [ ] Verify cleanup works on error paths
- [ ] Test with multiple simulated users
- [ ] Profile for performance regression
