"""Run end-to-end voice scenarios against the real pipeline (live Groq, Cerebras, Inworld).

    .venv/Scripts/python scripts/e2e/run_voice_e2e.py              # all scenarios
    .venv/Scripts/python scripts/e2e/run_voice_e2e.py basic noisy  # name filter

Writes reports/e2e-<timestamp>/report.md, results.json and one stereo WAV per
scenario (left = people, right = bot) so timing and pauses can be listened to.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.e2e.harness import (  # noqa: E402
    FakeMember,
    RunResult,
    Speech,
    overlap,
    word_error_rate,
)

TESTER = FakeMember(700000000000000001, "Tester", activity="Dune: Awakening")
BYSTANDER = FakeMember(700000000000000002, "Bystander")
MORGAN = FakeMember(700000000000000003, "Morgan")
MEMBERS = [TESTER, BYSTANDER, MORGAN]

SPOKEN_TAG_WORDS = {"whispering", "laughing", "happy", "sad", "angry", "surprised", "fearful",
                    "disgusted", "cheerful", "sigh", "laugh", "breathe", "cough", "clear throat"}


# ---------------------------------------------------------------- checks

Check = Callable[[RunResult], tuple[bool, str]]


def heard(expected: str, max_wer: float = 0.35, turn: int = 0) -> Check:
    def check(r):
        if len(r.turns) <= turn:
            return False, "no turn recorded"
        wer = word_error_rate(expected, r.turns[turn].user_text)
        return wer <= max_wer, f"WER {wer:.2f}: heard {r.turns[turn].user_text!r}"
    check.__name__ = f"stt_heard[{turn}]"
    return check


def _fuzzy_in(word: str, text: str) -> bool:
    """word appears in text, allowing a near-miss spelling from Whisper (Canberra/Candera)."""
    from difflib import SequenceMatcher

    if word in text:
        return True
    if " " in word or len(word) < 5:
        return False
    return any(SequenceMatcher(None, word, w).ratio() >= 0.8 for w in re.findall(r"[a-z']+", text))


def says_any(*words: str, turn: int = 0) -> Check:
    """Answer text from the LLM AND the bot's actual audio both contain one of words."""
    def check(r):
        if len(r.turns) <= turn:
            return False, "no turn recorded"
        answer = r.turns[turn].answer.lower()
        audio = r.bot_transcript.lower()
        ok_text = any(_fuzzy_in(w, answer) for w in words)
        ok_audio = any(_fuzzy_in(w, audio) for w in words)
        return ok_text and ok_audio, f"text={ok_text} audio={ok_audio}: {r.turns[turn].answer!r}"
    check.__name__ = f"answer_mentions[{turn}]({'/'.join(words)})"
    return check


def gives_temperature(turn: int = 0) -> Check:
    """Answer states a temperature (a number with degrees, or a bare number + weather words)."""
    def check(r):
        if len(r.turns) <= turn:
            return False, "no turn recorded"
        answer = r.turns[turn].answer.lower()
        with_unit = re.search(r"\b\d{1,3}\s*(°|degrees?\b|f\b)", answer)
        bare = re.search(r"\b\d{1,3}\b", answer) and re.search(r"high|low|temp|warm|cold", answer)
        ok = bool(with_unit or bare)
        return ok, repr(r.turns[turn].answer)
    check.__name__ = f"gives_temperature[{turn}]"
    return check


def used_tool(name: str, turn: int = 0) -> Check:
    def check(r):
        tools = r.turns[turn].tools if len(r.turns) > turn else []
        return name in tools, f"tools={tools}"
    check.__name__ = f"used_tool[{turn}]({name})"
    return check


def posted(pattern: str) -> Check:
    def check(r):
        return any(re.search(pattern, p, re.I) for p in r.posts), f"posts={r.posts}"
    check.__name__ = f"posted({pattern})"
    return check


def silent(r: RunResult) -> tuple[bool, str]:
    return not r.segments and not r.turns, f"{len(r.segments)} audio segments, {len(r.turns)} turns"


def no_llm_turn(r: RunResult) -> tuple[bool, str]:
    return not r.turns, f"{len(r.turns)} turns"


def played_something(r: RunResult) -> tuple[bool, str]:
    return bool(r.segments), " | ".join(s.transcript for s in r.segments) or "nothing played"


def heard_not(*words: str) -> Check:
    def check(r):
        text = " ".join(t.user_text for t in r.turns).lower()
        leaked = [w for w in words if w in text]
        return not leaked, f"heard {text!r}" + (f" (leaked: {leaked})" if leaked else "")
    check.__name__ = f"isolation(no {'/'.join(words)})"
    return check


# Applied to every scenario that produces an answer


def answer_segments(r: RunResult, turn) -> list:
    return [s for s in r.segments if s.transcript and overlap(s.transcript, turn.answer) >= 0.5]


def tts_fidelity(r: RunResult) -> tuple[bool, str]:
    worst = 0.0
    for turn in r.turns:
        if not turn.answer:
            continue
        spoken = " ".join(s.transcript for s in answer_segments(r, turn))
        worst = max(worst, word_error_rate(turn.answer, spoken))
    return worst <= 0.3, f"worst answer-vs-audio WER {worst:.2f}"


def no_tags_or_urls_spoken(r: RunResult) -> tuple[bool, str]:
    spoken = r.bot_transcript.lower()
    said = " ".join(re.sub(r"\[[^\]]+\]", " ", t.answer) for t in r.turns).lower()
    problems = [w for w in SPOKEN_TAG_WORDS if re.search(rf"\b{w}\b", spoken)
                and not re.search(rf"\b{w}\b", said)]
    problems += [u for u in ("http", "www", "dot com", ".com") if u in spoken]
    return not problems, f"spoken artifacts: {problems}" if problems else "clean"


def no_error(r: RunResult) -> tuple[bool, str]:
    apology = "encountered an error" in r.bot_transcript.lower()
    return not r.error and not apology, r.error or ("bot apologized for an error" if apology else "ok")


# ---------------------------------------------------------------- scenarios


@dataclass
class Scenario:
    name: str
    speeches: list[Speech]
    checks: list[Check]
    answers: bool = True  # apply the global answer checks
    between: Optional[float] = None
    note: str = ""
    expect_fail: bool = False  # known limitation: reported, not counted as a regression


Q_CAPITAL = "what's the capital of Australia?"

SCENARIOS = [
    Scenario("basic", [Speech(TESTER, [f"Hey Jarvis, {Q_CAPITAL}"])],
             [heard(Q_CAPITAL), says_any("canberra")],
             note="One breath: wake word + question, Discord voice-activity packets."),
    Scenario("pause_after_wake", [Speech(TESTER, ["Hey Jarvis.", 0.8, "how many legs does a spider have?"])],
             [heard("how many legs does a spider have"), says_any("eight", "8")],
             note="Natural pause between wake word and question."),
    Scenario("wake_only", [Speech(TESTER, ["Hey Jarvis."])],
             [played_something, no_llm_turn], answers=False,
             note="Only the wake word: expect an ack, then back to idle without an LLM turn."),
    Scenario("wake_then_late_question",
             [Speech(TESTER, ["Hey Jarvis."], at=0.5),
              Speech(TESTER, ["What's the boiling point of water in Fahrenheit?"], at=3.2)],
             [heard("what's the boiling point of water in fahrenheit"), says_any("212")],
             note="Wake word, wait for the ack, then ask (wake-only follow-up path)."),
    Scenario("no_wake_word", [Speech(TESTER, ["What's the capital of France? I always forget."])],
             [silent], answers=False, note="No wake word: the bot must stay silent."),
    Scenario("near_miss", [Speech(TESTER, ["Hey Travis, can you pass me the salt?"])],
             [silent], answers=False, note="Similar-sounding phrase (false-trigger probe)."),
    Scenario("web_weather", [Speech(TESTER, ["Hey Jarvis, what's the weather in Chicago right now?"])],
             [used_tool("web_search"), gives_temperature()],
             note="Live web lookup through the tools."),
    Scenario("post_to_chat", [Speech(TESTER, ["Hey Jarvis, post pizza at eight in the chat."])],
             [used_tool("post_to_chat"), posted("pizza")],
             note="Discord action; reply should be a brief ack."),
    Scenario("tag_user", [Speech(TESTER, ["Hey Jarvis, tag Morgan and tell her dinner is ready."])],
             [used_tool("tag_user"), posted(rf"<@{MORGAN.id}>")],
             note="Tag a member found by name in the guild."),
    Scenario("memory",
             [Speech(TESTER, ["Hey Jarvis, remember that my favorite color is green."]),
              Speech(TESTER, ["Hey Jarvis, what's my favorite color?"])],
             [says_any("green", turn=1)], between=1.5,
             note="Two separate turns; second relies on conversation history."),
    Scenario("crosstalk_isolation",
             [Speech(TESTER, [f"Hey Jarvis, {Q_CAPITAL}"], at=0.5),
              Speech(BYSTANDER, ["I think we should go to the north base and grab more water first."],
                     at=1.3, voice="carter")],
             [heard_not("north", "water"), says_any("canberra")],
             note="Someone else talks over the question; only the wake-word speaker should be heard."),
    Scenario("noisy_10db", [Speech(TESTER, [f"Hey Jarvis, {Q_CAPITAL}"], noise_snr_db=10)],
             [heard(Q_CAPITAL, max_wer=0.4), says_any("canberra")],
             note="Background noise at 10 dB SNR."),
    Scenario("open_mic", [Speech(TESTER, [f"Hey Jarvis, {Q_CAPITAL}"], mode="open")],
             [heard(Q_CAPITAL), says_any("canberra")],
             note="Open mic: packets keep flowing after speech, so the VAD must end the turn."),
    Scenario("male_voice", [Speech(TESTER, [f"Hey Jarvis, {Q_CAPITAL}"], voice="dennis")],
             [heard(Q_CAPITAL), says_any("canberra")], note="Different speaker voice."),
    Scenario("barge_in",
             [Speech(TESTER, ["Hey Jarvis, tell me a short story about a dragon."], at=0.5),
              Speech(TESTER, ["Hey Jarvis, stop."], at=9.0)],
             [no_llm_turn], answers=False, expect_fail=True,
             note="Known gap: no barge-in. Records what happens when the user interrupts."),
]


# ---------------------------------------------------------------- wake word sweep


async def wake_sweep(rig, threshold: float) -> list[dict]:
    """Offline: max wake score for 'Hey Jarvis' in many voices (and some negatives)."""
    import numpy as np
    from openwakeword.model import Model
    from scipy.signal import resample_poly

    model = Model(wakeword_models=[rig.settings.wake_word_model], inference_framework="onnx")
    clips = [("Hey Jarvis, what time is it?", v) for v in
             ("ashley", "dennis", "olivia", "carter", "priya", "theodore", "hades", "pixie")]
    clips += [(t, "ashley") for t in ("Hey Travis, what time is it?", "Hey, are you there?",
                                     "Jarvis is a cool name.", "Hey Casey, what's up?")]
    rows = []
    for text, voice in clips:
        audio = await rig.speech.say(text, voice)
        audio16 = resample_poly(audio, 2, 3)
        pcm = (np.clip(np.concatenate([np.zeros(8000), audio16, np.zeros(16000)]), -1, 1)
               * 32767).astype(np.int16)
        model.reset()
        best = 0.0
        for i in range(0, len(pcm) - 1280, 1280):
            scores = model.predict(pcm[i:i + 1280])
            best = max(best, max(scores.values()))
        positive = "jarvis" in text.lower() and text.lower().startswith("hey jarvis")
        rows.append({"text": text, "voice": voice, "score": round(best, 3),
                     "positive": positive, "detected": best >= threshold})
    return rows


# ---------------------------------------------------------------- reporting


def compute_metrics(r: RunResult) -> dict:
    """Latency from the end of the first speech that carries the question."""
    if not r.speeches:
        return {}
    asker = next((sp for sp in r.speeches if len(sp.text.split()) > 3), r.speeches[0])
    end = asker.ended_at
    m = {}
    if end is not None and r.segments:
        after = [sg for sg in r.segments if sg.start >= end - 0.5]
        if after:
            m["first_audio_s"] = round(after[0].start - end, 2)
        if r.turns and r.turns[0].answer:
            segs = [s for s in r.segments if s.transcript
                    and overlap(s.transcript, r.turns[0].answer) >= 0.5]
            if segs:
                m["answer_audio_s"] = round(segs[0].start - end, 2)
        gaps = [round(b.start - a.end, 2) for a, b in zip(r.segments, r.segments[1:])]
        m["gaps_s"] = gaps
    return m


def render_report(results: list[RunResult], scenarios: dict, sweep, threshold, out: Path):
    lines = ["# Voice E2E report", "",
             f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}  ", "",
             "| Scenario | Result | First audio | Answer audio | Notes |",
             "|---|---|---|---|---|"]
    for r in results:
        sc = scenarios[r.name]
        ok = all(c[1] for c in r.checks)
        status = "PASS" if ok else ("KNOWN GAP" if sc.expect_fail else "FAIL")
        lines.append(f"| {r.name} | {status} | {r.metrics.get('first_audio_s', '-')} s | "
                     f"{r.metrics.get('answer_audio_s', '-')} s | {sc.note} |")
    lines += ["", "Times are measured from the last packet of the user's speech.", ""]

    if sweep:
        lines += ["## Wake word sweep (offline)", "",
                  f"Threshold {threshold}. Positives should be detected, negatives should not.", "",
                  "| Phrase | Voice | Max score | Detected | Expected |", "|---|---|---|---|---|"]
        for row in sweep:
            lines.append(f"| {row['text']} | {row['voice']} | {row['score']} | "
                         f"{'yes' if row['detected'] else 'no'} | {'yes' if row['positive'] else 'no'} |")
        lines.append("")

    for r in results:
        sc = scenarios[r.name]
        lines += [f"## {r.name}", "", sc.note, ""]
        for sp in r.speeches:
            lines.append(f"- **{sp.user.display_name}** ({sp.voice}, {sp.mode}"
                         f"{f', noise {sp.noise_snr_db} dB' if sp.noise_snr_db else ''}) at "
                         f"{sp.started_at:.1f}s: \"{sp.text}\"")
        for i, t in enumerate(r.turns):
            lines.append(f"- Turn {i}: heard \"{t.user_text}\"; tools {t.tools or '-'}; "
                         f"answer \"{t.answer}\"")
        if r.posts:
            lines.append(f"- Posted to chat: {r.posts}")
        lines.append("- Bot audio:")
        for s in r.segments:
            lines.append(f"  - {s.start:6.2f}s +{s.duration:4.1f}s{' (cut)' if s.interrupted else ''}: "
                         f"\"{s.transcript}\"")
        if not r.segments:
            lines.append("  - (none)")
        if r.metrics:
            lines.append(f"- Metrics: {r.metrics}")
        lines.append("- Checks:")
        for name, ok, detail in r.checks:
            lines.append(f"  - {'PASS' if ok else 'FAIL'} {name}: {detail}")
        if r.error:
            lines.append(f"- Error: {r.error}")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def to_json(r: RunResult) -> dict:
    return {
        "name": r.name, "error": r.error, "duration": round(r.duration, 2),
        "metrics": r.metrics,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in r.checks],
        "turns": [t.__dict__ for t in r.turns], "posts": r.posts,
        "segments": [{"start": round(s.start, 2), "duration": round(s.duration, 2),
                      "interrupted": s.interrupted, "transcript": s.transcript} for s in r.segments],
    }


# ---------------------------------------------------------------- main


async def main(filters: list[str]) -> int:
    from scripts.e2e.harness import Rig
    from src.config import get_settings

    settings = get_settings()
    out = ROOT / "reports" / f"e2e-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler()],
    )
    for noisy in ("httpx", "httpcore", "openai", "primp", "ddgs", "discord"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    rig = Rig(settings, out)
    selected = [s for s in SCENARIOS if not filters or any(f in s.name for f in filters)]
    results = []
    for sc in selected:
        print(f"\n===== {sc.name} =====", flush=True)
        r = await rig.run(sc.name, sc.speeches, MEMBERS, between=sc.between)
        checks = list(sc.checks) + ([no_error, tts_fidelity, no_tags_or_urls_spoken] if sc.answers else [])
        for check in checks:
            try:
                ok, detail = check(r)
            except Exception as e:
                ok, detail = False, f"check crashed: {e}"
            r.checks.append((check.__name__, ok, detail))
        r.metrics = compute_metrics(r)
        results.append(r)
        print(f"  -> {'PASS' if all(c[1] for c in r.checks) else 'FAIL'} {r.metrics}", flush=True)

    sweep = None
    if not filters or "sweep" in filters:
        sweep = await wake_sweep(rig, settings.wake_word_confidence)

    scenarios = {s.name: s for s in SCENARIOS}
    render_report(results, scenarios, sweep, settings.wake_word_confidence, out)
    (out / "results.json").write_text(
        json.dumps({"scenarios": [to_json(r) for r in results], "wake_sweep": sweep}, indent=2),
        encoding="utf-8")
    print(f"\nReport: {out / 'report.md'}")
    failed = [r.name for r in results
              if not all(c[1] for c in r.checks) and not scenarios[r.name].expect_fail]
    print(f"Failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
