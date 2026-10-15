"""Detectors: turn a LiveTurn into findings worth telling the owner about.

Thresholds come from e2e runs (scripts/e2e): good turns start audio ~0.6-0.9s
after the question ends and the answer ~3-4s after; normal sentence gaps are
~0.3s. The defaults flag clearly bad turns only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import turn as tr
from .text_checks import (
    WEAK_GIVE_UP_PHRASES,
    give_up_phrases,
    spoken_artifacts,
    strip_tags,
    text_artifacts,
    unclear_phrases,
    word_error_rate,
)


@dataclass(frozen=True)
class Thresholds:
    first_audio_s: float = 3.0  # first sound (affirmation) after speech end
    answer_s: float = 8.0  # first answer sentence, no web lookup
    answer_web_s: float = 20.0  # first answer sentence after web_search/web_extract
    gap_s: float = 2.5  # silence between two answer sentences
    hang_s: float = 30.0  # no progress at all
    tts_wer: float = 0.4  # answer text vs Whisper of its audio (MONITOR_RETRANSCRIBE)
    # The quiet the bot waits for before a question counts as over (MIN_SILENCE_MS). The
    # limits above are measured from the end of it, not from the speaker's last word.
    pause_s: float = 0.0

    @classmethod
    def from_settings(cls, settings) -> Thresholds:
        return cls(
            first_audio_s=settings.monitor_slow_first_audio_seconds,
            answer_s=settings.monitor_slow_answer_seconds,
            answer_web_s=settings.monitor_slow_answer_web_seconds,
            gap_s=settings.monitor_long_gap_seconds,
            hang_s=settings.monitor_hang_seconds,
            pause_s=settings.min_silence_ms / 1000,
        )


@dataclass
class Finding:
    detector: str  # stable id, e.g. "error_reply"
    summary: str  # short title
    detail: str  # plain-English explanation
    cause: str = ""  # what makes two findings "the same problem" (dedupe)

    @property
    def key(self) -> str:
        return f"{self.detector}:{normalize_cause(self.cause)}"


def normalize_cause(text: str) -> str:
    """Collapse numbers, links, quoted text and ids so recurring causes match."""
    text = re.sub(r"https?://\S+", "<url>", text.lower())
    text = re.sub(r"(['\"]).*?\1", "<q>", text)
    text = re.sub(r"\d+(\.\d+)?", "#", text)
    text = re.sub(r"[^a-z#<>:_ ]+", " ", text)
    return " ".join(text.split())[:80]


# Warnings/errors that matter on their own: (detector, summary, pattern, explanation)
LOG_RULES = (
    ("tts_repeat", "TTS repeat guard fired",
     re.compile(r"possible repeat"),
     "Inworld returned a clip far longer than its text (probably a repeated sentence); "
     "the bot synthesized it again."),
    ("thinking_phrase_failed", "Filler phrase failed",
     re.compile(r"Dynamic thinking phrase failed|Opening phrase failed|"
                r"Failed to play thinking phrase|Failed to play acknowledgment"),
     "A filler phrase (affirmation/thinking/ack) failed, so there was dead air."),
    ("tts_error", "TTS failed",
     re.compile(r"Inworld API error|Inworld TTS (stream )?error|TTS error for sentence|"
                r"Error synthesizing sentence"),
     "Inworld failed on part of the reply, so some of it was never spoken."),
    ("stt_error", "Speech-to-text failed",
     re.compile(r"Groq transcription error"),
     "Whisper failed, so the question was dropped."),
    ("playback_error", "Playback error",
     re.compile(r"Playback error|\[PLAYBACK\] Error"),
     "Discord playback reported an error."),
    ("cut_off", "Question was cut off",
     re.compile(r"\[CARRY-ON\] .*kept talking"),
     "The speaker paused long enough that the bot took the question as finished, then kept "
     "talking; the bot started over with everything they said."),
    ("talked_during_answer", "Speaker talked during the answer",
     re.compile(r"\[CARRY-ON\] .*talked during the answer"),
     "The speaker talked while the bot was answering; it was kept for their follow-up "
     "(or it was the bot's own voice coming back through their mic)."),
)
# Explain error/graceful-failure replies rather than being posted separately
_LLM_EVENTS = re.compile(r"LLM API error|LLM stream ended without a result|LLM returned no text|"
                         r"Max tool turns|max_completion_tokens|Failed to parse tool arguments")
# Covered by the tool detectors
_TOOL_EVENTS = re.compile(r"Tool execution error|web_search failed|web_extract failed|Unknown tool")


def _first_event(turn: tr.LiveTurn, pattern: re.Pattern) -> tr.LogEvent | None:
    return next((e for e in turn.events if pattern.search(e.message)), None)


def detect(turn: tr.LiveTurn, th: Thresholds = Thresholds()) -> list[Finding]:
    """All findings for one turn, most important first."""
    if turn.outcome == tr.CANCELLED:
        return []
    if turn.outcome == tr.LATE_REPLY:
        return [Finding("late_reply", "Reply came after the window closed", turn.note)]
    if turn.outcome == tr.HUNG:
        return [Finding(
            "hang", "Bot went silent",
            f"No audio and no progress for {turn.idle_s:.0f}s ({turn.last_step}). "
            "The turn may still finish; a follow-up post will say how if it does.",
            cause=turn.last_step,
        )]

    findings: list[Finding] = []
    llm_event = _first_event(turn, _LLM_EVENTS)
    llm_note = f' Logged: "{llm_event.message[:200]}"' if llm_event else ""

    if turn.outcome == tr.CRASHED:
        findings.append(Finding("crash", "Turn crashed",
                                f"Processing the question raised {turn.error}.", cause=turn.error))
    elif turn.outcome == tr.ERROR_REPLY_OUTCOME:
        findings.append(Finding(
            "error_reply", "Error reply spoken",
            "The LLM call failed and the bot said its error reply." + llm_note,
            cause=llm_event.message if llm_event else "",
        ))
    elif turn.outcome == tr.GRACEFUL_FAILURE:
        findings.append(Finding(
            "graceful_failure", "No usable answer",
            "The model gave no usable answer, so the bot said \"I couldn't find a good answer\"."
            + llm_note,
            cause=llm_event.message if llm_event else "",
        ))
    elif turn.outcome == tr.NO_REPLY:
        findings.append(Finding("no_reply", "Question got no reply",
                                "The question was transcribed but the bot said nothing back."))
    elif turn.outcome == tr.ANSWERED:
        findings += _answer_findings(turn, th)

    if turn.outcome not in (tr.WAKE_WORD_ONLY, tr.EMPTY, tr.NOT_ADDRESSED):
        findings += _tool_findings(turn)
        if not turn.hang_reported:
            findings += _timing_findings(turn, th)
    findings += _log_findings(turn, skip={f.detector for f in findings})
    return findings


def _answer_findings(turn: tr.LiveTurn, th: Thresholds) -> list[Finding]:
    out = []
    reply = turn.reply
    strong = [p for p in give_up_phrases(reply) if p not in WEAK_GIVE_UP_PHRASES]
    weak = [p for p in give_up_phrases(reply) if p in WEAK_GIVE_UP_PHRASES]
    if strong or (weak and turn.used_web):
        phrases = strong or weak
        out.append(Finding(
            "give_up", "Bot gave up on the answer",
            f"The answer admits failure ({', '.join(repr(p) for p in phrases)})"
            + (" after a web lookup." if turn.used_web else "."),
            cause=" ".join(t.name for t in turn.tools),
        ))
    unclear = unclear_phrases(reply)
    if unclear:
        out.append(Finding(
            "unclear_question", "Bot didn't understand the question",
            f"The bot asked to hear it again ({', '.join(repr(p) for p in unclear)}); "
            "Whisper may have misheard it.",
        ))
    artifacts = text_artifacts(reply)
    if artifacts:
        out.append(Finding("formatting", "Formatting in spoken answer",
                           f"The text sent to TTS contained {', '.join(artifacts)}.",
                           cause=" ".join(artifacts)))
    if turn.spoken is not None:
        said = strip_tags(reply)
        wer = word_error_rate(said, turn.spoken)
        if wer > th.tts_wer:
            out.append(Finding(
                "tts_mismatch", "Spoken audio doesn't match the text",
                f'Whisper heard the answer audio as "{turn.spoken[:200]}" '
                f"(word error rate {wer:.2f} against the text).",
            ))
        spoken = spoken_artifacts(turn.spoken, reply)
        if spoken:
            out.append(Finding("spoken_artifacts", "Tags or links read aloud",
                               f"The audio contained {', '.join(spoken)}.",
                               cause=" ".join(spoken)))
    return out


def _tool_findings(turn: tr.LiveTurn) -> list[Finding]:
    out = []
    for i, use in enumerate(turn.tools):
        recovered = any(t.name == use.name and t.ok and not t.no_results
                        for t in turn.tools[i + 1:])
        if not use.ok and not recovered:
            timed_out = "timed out" in use.error.lower()
            out.append(Finding(
                "tool_timeout" if timed_out else "tool_error",
                f"{use.name} {'timed out' if timed_out else 'failed'}",
                f"{use.name}({use.args}) returned an error: \"{use.error}\"",
                cause=f"{use.name} {use.error}",
            ))
    searches = [t for t in turn.tools if t.name == "web_search"]
    if searches and all(t.no_results for t in searches) and not any(
            t.name == "web_extract" and t.ok for t in turn.tools):
        queries = ", ".join(t.args for t in searches)
        out.append(Finding(
            "web_no_results", "Web search found nothing",
            f"Every web_search returned no results ({queries}). "
            "Often a misheard question or a search backend problem.",
        ))
    return out


def _timing_findings(turn: tr.LiveTurn, th: Thresholds) -> list[Finding]:
    out = []
    first, answer = turn.first_audio_s, turn.answer_audio_s
    pause = th.pause_s
    # Times below count from the end of the pause that ends a question
    note = f" ({pause:g}s of it is the pause that ends a question)" if pause else ""
    # When the affirmation was skipped (someone still talking) the answer is the
    # first sound, and only the answer threshold applies
    opening_played = bool(turn.clips) and not turn.clips[0].answer
    if opening_played and first is not None and first - pause > th.first_audio_s:
        out.append(Finding(
            "slow_first_audio", "Slow to respond",
            f"The first sound came {first:.1f}s after they stopped talking{note} "
            f"(normally under 1s after that; limit {th.first_audio_s:g}s).",
        ))
    limit = th.answer_web_s if turn.used_web else th.answer_s
    if answer is not None and answer - pause > limit:
        out.append(Finding(
            "slow_answer", "Slow answer",
            f"The answer started {answer:.1f}s after they stopped talking{note} "
            f"(limit {limit:g}s{' with a web lookup' if turn.used_web else ''}). {breakdown(turn)}",
            cause="web" if turn.used_web else "",
        ))
    replied = turn.outcome in (tr.ANSWERED, tr.ERROR_REPLY_OUTCOME, tr.GRACEFUL_FAILURE)
    if answer is None and replied and strip_tags(turn.reply):
        out.append(Finding(
            "no_answer_audio", "Answer was never played",
            "The bot had a reply but none of it was played (TTS or playback failed).",
        ))
    long_gaps = [g for g in turn.answer_gaps_s if g > th.gap_s]
    if long_gaps:
        out.append(Finding(
            "long_gap", "Long pause mid-answer",
            f"Silence of {max(long_gaps):.1f}s between answer sentences "
            f"(normally ~0.3s; limit {th.gap_s:g}s). All gaps: {turn.answer_gaps_s}.",
        ))
    return out


def _log_findings(turn: tr.LiveTurn, skip: set[str]) -> list[Finding]:
    out = []
    matched: set[int] = set()
    for detector, summary, pattern, explanation in LOG_RULES:
        hits = [i for i, e in enumerate(turn.events) if pattern.search(e.message)]
        matched.update(hits)
        if hits and detector not in skip:
            first = turn.events[hits[0]]
            count = f" ({len(hits)} times this turn)" if len(hits) > 1 else ""
            out.append(Finding(detector, summary,
                               f'{explanation}{count} Logged: "{first.message[:200]}"',
                               cause=first.message))
    for i, e in enumerate(turn.events):
        if i in matched or e.level not in ("ERROR", "CRITICAL") or not e.logger.startswith("src"):
            continue
        if _LLM_EVENTS.search(e.message) or _TOOL_EVENTS.search(e.message):
            continue
        out.append(Finding("unexpected_error", "Error logged",
                           f'{e.logger} logged an error: "{e.message[:200]}"',
                           cause=f"{e.logger} {e.message}"))
        break  # one is enough to start looking
    return out


def breakdown(turn: tr.LiveTurn) -> str:
    """Where the time went, for slow-turn messages."""
    parts = []
    if turn.stt_s is not None:
        parts.append(f"STT done at {turn.stt_s:.1f}s")
    if turn.llm_calls:
        firsts = [c.get("first_text_s") for c in turn.llm_calls]
        totals = [c.get("total_s") for c in turn.llm_calls]
        parts.append(f"{len(turn.llm_calls)} LLM call(s), total "
                     + "+".join(f"{t:.1f}s" for t in totals if t is not None)
                     + (f" (first text {firsts[-1]:.1f}s)" if firsts[-1] is not None else ""))
    if turn.tools:
        parts.append("tools: " + ", ".join(t.name for t in turn.tools))
    if turn.tts_s:
        parts.append(f"slowest TTS {max(turn.tts_s):.1f}s")
    return ("Breakdown: " + "; ".join(parts) + ".") if parts else ""
