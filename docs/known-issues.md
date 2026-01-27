# Known Issues

Tracking issues discovered during testing for future fixes.

---

## Issue #1: LLM Thinking Token Leak + TTS Rate Limiting

**Date:** 2026-01-26
**Severity:** Medium
**Status:** Open

### Symptoms

1. TTS receives dozens of garbage fragments like `"……......"`, `"???..."`, `"…We…..."`
2. Inworld TTS hits 429 rate limit (100 req/s exceeded)
3. Log shows model's internal reasoning: `"The assistant's last response is garbled nonsense"`, `"Need to correct"`, `"We should respond with brief answer"`

### Root Cause

Two compounding issues:

1. **Groq model leaked chain-of-thought reasoning** into the output stream instead of just the response. This violates the system prompt rule "Do NOT output thinking or planning text."

2. **Sentence detector triggers too eagerly** - In [voice_session.py:1202-1213](../src/bot/voice_session.py), any buffer ending with `.` gets sent to TTS immediately. Garbage content full of `...` and `…` triggers this repeatedly.

### Reproduction

- Occurred after successful image generation tool call
- Model: Groq (GPT-OSS or Llama 3.3)
- User request: "generate a brand new image based off of the last image"

### Proposed Fixes

1. **Defensive TTS filtering** - Before sending to TTS:
   - Minimum 10 characters per sentence
   - Must contain at least one alphabetic character
   - Filter pure punctuation/ellipsis strings

2. **Debounce sentence detection** - Don't trigger on every `.`, wait for meaningful content

3. **Model-side** - Investigate why Groq model leaks thinking tokens (may be model-specific bug)

### Log Excerpt

```
2026-01-26 21:55:28 | INFO | src.tts.synthesizer | [TTS] Sending to Inworld: text=[laugh0] O…………......
2026-01-26 21:55:28 | INFO | src.tts.synthesizer | [TTS] Sending to Inworld: text=……......
2026-01-26 21:55:28 | INFO | src.tts.synthesizer | [TTS] Sending to Inworld: text=The assistant's last response is garbled nonsense....
2026-01-26 21:55:29 | ERROR | src.tts.synthesizer | Inworld API error 429: rate limit exceeded
```

---

## Issue #2: Wake Word Detection Spam After Speech End

**Date:** 2026-01-26
**Severity:** Medium
**Status:** In Progress (workaround committed)

### Symptoms

1. After user finishes speaking and session enters `processing` state, wake word detector fires 20-30+ times in rapid succession
2. All detections are from the same user who originally triggered the wake word
3. Session correctly rejects these (`NOT setting event - state=processing`) but logs are noisy
4. Score progression shows classic activation buildup: 0.75 -> 0.90 -> 0.99 -> 1.00

### Root Cause

**Unclear - multiple theories:**

1. **OpenWakeWord reset() incomplete** - `model.reset()` may not fully clear internal mel spectrogram buffers. When new audio arrives after reset, residual activation patterns cause false positives.

2. **Audio contains wake word** - The post-gap audio actually contains "hey jarvis" (user said it again, or captured buffer is being replayed somehow).

3. **State transition race** - `stop_listening()` -> `on_idle_transition()` -> `reset_all_detectors()` happens, but audio packets already in-flight get processed before reset takes effect.

### Reproduction

1. Have two people in voice channel
2. Person A says "Hey Jarvis, [question]"
3. Person B talks while Person A is speaking
4. After speech ends, observe wake word spam in logs

### Workaround (Committed)

Added 5-second refractory period after wake word detection. Suppresses further detections but doesn't fix root cause.

```python
self._wake_word_refractory_until: Optional[float] = None
self._wake_word_refractory_seconds = 5.0
```

### Proposed Root Cause Fixes

1. Investigate OpenWakeWord's `reset()` behavior - does it clear feature extractor state?
2. Add logging to confirm whether post-gap audio actually contains wake word
3. Consider blocking wake word detection during `processing` state at sink level (not just session level)

### Log Excerpt

```
22:16:41 | [SINK] stop_listening() called
22:16:41 | [SINK] Audio resumed after 0.8s gap
22:16:41 | Wake word detected: hey_jarvis (score: 0.75, inference: 0.0ms)
22:16:41 | [SINK] Wake word detected! (packet lag: 1ms, user: 100000000000000003)
22:16:41 | [CALLBACK] _on_wake_word called, state=processing
22:16:41 | [CALLBACK] NOT setting event - state=processing
... (repeats 20+ times with scores increasing to 1.00)
```

---

## Issue #3: Multi-Speaker VAD Truncation

**Date:** 2026-01-26
**Severity:** High
**Status:** In Progress (workaround committed)

### Symptoms

1. User A says "Hey Jarvis, generate an image of a real [something]"
2. User B talks simultaneously
3. User A's speech is truncated at "a real" - rest of sentence lost
4. VAD reports speech ended at 2.10s even though user wasn't done

### Root Cause

**Theory: Discord packet delivery unreliability**

When multiple users talk simultaneously:
- Discord may not reliably deliver audio packets from all users
- User A's packets may get delayed/dropped while User B's arrive
- VAD sees gap in User A's audio -> interprets as silence -> ends speech

Speaker isolation IS working (User B's audio filtered from VAD), but the underlying packet delivery issue causes User A's stream to have gaps.

### Reproduction

1. Two people in voice channel
2. Person A triggers wake word and starts speaking
3. Person B talks over Person A
4. Person A's speech gets cut off mid-sentence

### Workaround (Committed)

Added 500ms multi-speaker silence extension. When VAD detects speech end but others are speaking, require extra confirmation time before committing.

```python
self._multi_speaker_silence_extension_ms = 500
if others_speaking:
    # Wait 500ms before committing to speech end
```

### Proposed Root Cause Fixes

1. Investigate Discord.py's audio packet delivery behavior with concurrent speakers
2. Consider increasing base `min_silence_ms` when multiple users detected
3. Add packet loss detection/logging to confirm theory

---

## Issue #4: 66.5s Buffer Anomaly

**Date:** 2026-01-26
**Severity:** High
**Status:** Open

### Symptoms

1. VAD detects speech ended at 1.30s
2. But audio buffer grows to 66.5 seconds (!)
3. PTT timeout eventually fires with massive buffer
4. Suggests audio kept being buffered long after VAD said speech ended

### Root Cause

**Unknown** - State management issue between VAD endpoint detection and buffer handling.

VAD's `detect_endpoint()` returns "end" at 1.30s, but:
- `_speech_started` doesn't get reset properly?
- Buffer keeps accumulating in listening state?
- Or something else prevents the speech_end callback from firing?

### Reproduction

Observed during multi-speaker testing. Exact trigger unclear.

### Log Excerpt

```
22:16:41 | VAD: Speech ENDED at 1.30s
22:16:41 | [SINK] Audio stopped for 0.57s (PTT released?), buffer=66.5s
22:16:41 | [TIMING] Audio captured: 66.50s (1064000 samples)
```

### Proposed Fixes

1. Add state logging to trace why buffer keeps growing after VAD end
2. Check if `_speech_started` flag is being properly managed
3. Investigate interaction between VAD endpoint and PTT timeout paths

---

## Issue #5: PTT Timeout Cuts Off Speech When Multiple Users Talking

**Date:** 2026-01-27
**Severity:** High
**Status:** Open

### Symptoms

1. User A says "Hey Jarvis" + question while User B is also talking
2. User A pauses briefly (0.5-0.6s) mid-sentence
3. System interprets pause as end of speech (PTT released)
4. Transcription comes back garbled/truncated (e.g., "Hey, Jar." instead of "Hey, Jarvis, do something")
5. Bot responds incorrectly or enters wake-word-only follow-up flow

### Root Cause

**Three compounding issues:**

1. **Aggressive PTT timeout (0.5s)** - In [voice_sink.py:99](../src/bot/voice_sink.py), `_silence_timeout = 0.5` means any 0.5s pause triggers speech end.

2. **Speaker isolation exacerbates the issue** - When speaker isolation is active ([voice_sink.py:432-434](../src/bot/voice_sink.py)), only audio from the locked user updates `_last_audio_time`. Even though User B is still talking, their audio doesn't prevent the timeout for User A.

3. **Wake word pattern mismatch** - The transcription "Hey, Jar." doesn't match the wake word pattern `^(hey[,\s]*jarvis|a[,\s]*jarvis)`, so it's passed to the LLM as a query instead of being recognized as wake-word-only.

### Reproduction

1. Two people in voice channel
2. Person A (Pat) says "Hey Jarvis" + something
3. Person B talks simultaneously (gets filtered by speaker isolation)
4. Person A pauses briefly (natural speech gap or waiting for acknowledgment)
5. System triggers `[SPEECH_END] PTT released / no audio packets` after ~0.5s

### Proposed Fixes

1. **Increase PTT timeout when speaker isolation is active** - Extend from 0.5s to 0.75-1.0s when other users are detected speaking
2. **Add truncated wake word variants to pattern** - Include `"hey[,\s]*jar\.?"` to catch garbled transcriptions
3. **Don't trigger PTT timeout while others are speaking** - If `speaking_detector` shows others active, extend the timeout

### Log Excerpt

```
00:00:15 | Wake word detected: hey_jarvis (score: 0.97)
00:00:15 | [SINK ISOLATION] Filtering audio from user 100000000000000001 (locked to 100000000000000003)
00:00:16 | VAD: Speech STARTED at 1.30s
00:00:17 | [SINK] Audio stopped for 0.58s (PTT released?), buffer=3.1s
00:00:17 | [SPEECH_END] PTT released / no audio packets (buffer=3.14s)
00:00:21 | [GROQ] Transcribed: Hey, Jar.
00:00:21 | [TRANSCRIPTION] Raw: 'Hey, Jar.'
00:00:21 | [TRANSCRIPTION] Processed: 'Hey, Jar.'  # Should have been wake_word_only!
```

---

## Template for New Issues

```markdown
## Issue #N: Title

**Date:** YYYY-MM-DD
**Severity:** Critical/High/Medium/Low
**Status:** Open/In Progress/Fixed

### Symptoms

What the user/developer observes.

### Root Cause

Technical explanation of why it happens.

### Reproduction

Steps or conditions to trigger the issue.

### Proposed Fixes

Potential solutions.

### Log Excerpt

Relevant log snippets.
```
