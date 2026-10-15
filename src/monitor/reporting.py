"""Discord post and approved-fix entry text for monitor findings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime

from .detectors import Finding
from .turn import LiveTurn

POST_PREFIX = "**Bot monitor:"
REACTION_HINT = "React ✅ if something went wrong, ❌ if it was fine or didn't disrupt the flow."
# Posts made before the wording changed can still be reacted to
_HINTS = (REACTION_HINT, "React ✅ if this should be fixed, ❌ to ignore.")
SEEN_PREFIX = "Seen "
RECORDING_PREFIX = "Recording: "
CONFIRMED = "confirmed"
DISMISSED = "dismissed"
DISCORD_LIMIT = 2000

# Shapes of API keys and tokens, in case one ever shows up in an error message
_SECRET_SHAPES = (
    re.compile(r"\b(sk|gsk|csk|xai|pk)[-_][A-Za-z0-9_\-]{12,}"),
    re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=\-]{8,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,}"),  # Discord token
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)=([^\s&]+)"),
    re.compile(r"\b[A-Za-z0-9_\-]{40,}={0,2}"),  # long opaque strings (base64 keys etc.)
)
_SECRET_SETTINGS = ("discord_token", "groq_api_key", "cerebras_api_key", "inworld_api_key",
                    "nanobanana_api_key")


class Redactor:
    """Removes configured secret values and anything shaped like a key."""

    def __init__(self, settings=None, extra: Iterable[str] = ()):
        values = [getattr(settings, name, None) for name in _SECRET_SETTINGS] if settings else []
        self._values = sorted({v for v in [*values, *extra] if isinstance(v, str) and len(v) >= 8},
                              key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for value in self._values:
            text = text.replace(value, "[redacted]")
        for pattern in _SECRET_SHAPES:
            text = pattern.sub("[redacted]", text)
        return text


def _quote(text: str, limit: int) -> str:
    """User/bot text for Discord: one line, cut, no markdown or mentions."""
    text = " ".join((text or "").split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    text = re.sub(r"([*_~`|>\\])", r"\\\1", text)
    return text.replace("@", "@​")


def _fmt(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:.1f}s"


def timing_line(turn: LiveTurn) -> str:
    parts = [f"first audio {_fmt(turn.first_audio_s)}", f"answer {_fmt(turn.answer_audio_s)}"]
    if turn.stt_s is not None:
        parts.append(f"STT {_fmt(turn.stt_s)}")
    if turn.llm_calls:
        totals = [c.get("total_s") for c in turn.llm_calls if c.get("total_s") is not None]
        first = turn.llm_calls[0].get("first_text_s")
        parts.append(f"LLM {len(turn.llm_calls)} call(s) {'+'.join(f'{t:.1f}s' for t in totals)}"
                     + (f", first text {first:.1f}s" if first is not None else ""))
    if turn.tts_s:
        parts.append(f"TTS max {max(turn.tts_s):.1f}s")
    if turn.answer_gaps_s:
        parts.append(f"gaps {turn.answer_gaps_s}")
    return ", ".join(parts)


def tools_line(turn: LiveTurn) -> str:
    out = []
    for t in turn.tools:
        status = "no results" if t.no_results else ("ok" if t.ok else "error")
        out.append(f"{t.name} {_quote(t.args, 80)} ({status})".replace("  ", " "))
    return "; ".join(out)


def format_post(turn: LiveTurn, findings: list[Finding], redact: Redactor) -> str:
    """Short plain-English Discord message for one turn's findings."""
    title = findings[0].summary
    also = [f.summary.lower() for f in findings[1:]]
    who = turn.speaker_name or "someone"
    lines = [f"{POST_PREFIX} {title}**" + (f" (also: {', '.join(also)})" if also else "")]
    if turn.heard:
        lines.append(f'Heard from {_quote(who, 40)}: "{_quote(turn.heard, 200)}"')
    elif turn.raw_heard:
        lines.append(f'Heard from {_quote(who, 40)}: "{_quote(turn.raw_heard, 200)}"')
    else:
        lines.append(f"Speaker: {_quote(who, 40)}")
    if turn.reply:
        lines.append(f'Bot said: "{_quote(turn.reply, 300)}"')
    lines.append("What went wrong:")
    lines += [f"- {_quote(f.detail, 400)}" for f in findings]
    lines.append(f"Timings: {timing_line(turn)}")
    if turn.tools:
        lines.append(f"Tools: {tools_line(turn)}")
    if turn.audio_file:
        lines.append(f"{RECORDING_PREFIX}{turn.audio_file}")
    lines.append(REACTION_HINT)
    return _fit(redact("\n".join(lines)))


def with_seen_count(content: str, count: int, last_seen: datetime) -> str:
    """The post with a 'Seen N times' line (replacing any earlier one)."""
    lines = [ln for ln in content.split("\n") if not ln.startswith(SEEN_PREFIX)]
    seen = f"{SEEN_PREFIX}{count} times, last at {last_seen:%H:%M}."
    if lines and lines[-1] in _HINTS:
        lines.insert(len(lines) - 1, seen)
    else:
        lines.append(seen)
    return _fit("\n".join(lines))


def recording_path(content: str) -> str | None:
    """The repo-relative recording path from a post's Recording line, if any."""
    for ln in (content or "").split("\n"):
        if ln.startswith(RECORDING_PREFIX):
            return ln[len(RECORDING_PREFIX):].strip() or None
    return None


def _fit(text: str) -> str:
    if len(text) <= DISCORD_LIMIT:
        return text
    hint = "\n" + REACTION_HINT if text.endswith(REACTION_HINT) else ""
    body = text[: len(text) - len(hint)] if hint else text
    return body[: DISCORD_LIMIT - len(hint) - 4].rstrip() + " ..." + hint


def approval_entry(
    message_id: int,
    content: str,
    jump_url: str,
    posted: datetime | None,
    approved: datetime,
    redact: Redactor,
    verdict: str = CONFIRMED,
) -> str:
    """Markdown entry for the approved-fixes (check) or dismissed-findings (cross) file,
    built from the Discord post."""
    lines = [ln for ln in content.split("\n") if ln and ln not in _HINTS]
    title = lines[0] if lines else "Monitor finding"
    title = title.replace(POST_PREFIX, "").replace("**", "").strip() or "Monitor finding"
    out = [
        f"## {(posted or approved):%Y-%m-%d %H:%M} - {title}",
        "",
        f"<!-- discord-message:{message_id} -->",
        (f"Status: the owner confirmed something went wrong ({approved:%Y-%m-%d %H:%M}). "
         "The diagnosis below is the monitor's guess: check the logs and recording for "
         "what actually happened before fixing.")
        if verdict == CONFIRMED else
        (f"Status: the owner dismissed this ({approved:%Y-%m-%d %H:%M}): no problem, or it "
         "didn't disrupt the flow. Fix only if the fix has no tradeoffs or edge cases."),
    ]
    if jump_url:
        out.append(f"Discord: {jump_url}")
    out.append("")
    in_details = False  # the "- detail" lines under "What went wrong:" become sub-bullets
    for ln in lines[1:]:
        if ln.startswith("- "):
            out.append(f"  {ln}" if in_details else ln)
        else:
            in_details = ln.startswith("What went wrong")
            out.append(f"- {ln}")
    out.append("")
    return redact("\n".join(out))
