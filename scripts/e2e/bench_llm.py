"""Benchmark the voice LLM at each Cerebras reasoning level, plus web search latency.

    .venv/Scripts/python scripts/e2e/bench_llm.py            # everything
    .venv/Scripts/python scripts/e2e/bench_llm.py llm 3      # LLM only, 3 reps
    .venv/Scripts/python scripts/e2e/bench_llm.py search     # ddgs backends only

LLM runs go through the real ResponseStage + tool loop + system prompt (text only,
Discord actions faked). Writes reports/bench-<timestamp>/report.md and results.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EFFORTS = ["none", "low", "medium", "high"]


@dataclass
class Prompt:
    name: str
    text: str
    check: Callable[[str, list[str]], bool]  # (answer, tools) -> correct?
    kind: str


def has(*words):
    return lambda a, t: any(w in a.lower() for w in words)


def tool(name, *words):
    return lambda a, t: name in t and (not words or any(w in a.lower() for w in words))


PROMPTS = [
    Prompt("fact", "what's the capital of Australia?", has("canberra"), "plain"),
    Prompt("arithmetic", "if I have three stacks of 64 iron and use 50 for a forge, how much is left?",
           has("142", "hundred forty-two", "hundred and forty-two", "hundred forty two"), "reasoning"),
    Prompt("percent", "what's 17 percent of 2340?", has("397.8", "397 point 8", "three hundred ninety-seven"),
           "reasoning"),
    Prompt("letters", "how many r's are in the word strawberry?", has("three", "3"), "reasoning"),
    Prompt("chitchat", "how are you doing today?", lambda a, t: not t, "plain"),
    Prompt("weather", "what's the weather in Chicago right now?",
           lambda a, t: "web_search" in t and bool(re.search(r"\d", a)), "tool"),
    Prompt("post", "post pizza at eight in the chat", tool("post_to_chat"), "tool"),
    Prompt("misheard", "what's the recipe for heavy dirts in dune awakening?", has("dart"), "tool"),
]


class TurnLog(logging.Handler):
    """Collects the per-turn [TIMING] lines the Cerebras client logs."""

    PATTERN = re.compile(r"first_text=([\d.]+)s total=([\d.]+)s reasoning_chars=(\d+)")

    def __init__(self):
        super().__init__()
        self.turns = []

    def emit(self, record):
        m = self.PATTERN.search(record.getMessage())
        if m:
            self.turns.append((float(m[1]), float(m[2]), int(m[3])))


def format_issues(answer: str) -> list[str]:
    issues = []
    body = re.sub(r"\[[a-z ]+\]", "", answer)
    if re.search(r"[*#`]|^\s*[-•]\s", body, re.M):
        issues.append("markdown")
    if re.search(r"https?://|www\.|\.com\b", body):
        issues.append("url")
    sentences = [s for s in re.split(r"[.!?]+\s|\n", body.strip()) if s.strip()]
    if len(sentences) > 4:
        issues.append(f"{len(sentences)} sentences")
    return issues


async def run_llm(settings, reps: int) -> list[dict]:
    from src.llm import create_llm_client
    from src.pipeline.base import PipelineContext
    from src.pipeline.response_stage import ResponseStage

    log = TurnLog()
    logging.getLogger("src.llm.cerebras_client").addHandler(log)

    async def tagged(name, msg=""):
        return f"Done! I've tagged {name} in general chat."

    async def posted(msg):
        return "Done! I've posted that to general chat."

    rows = []
    for effort in EFFORTS:
        svc = create_llm_client(settings.model_copy(update={"cerebras_reasoning_effort": effort}),
                                on_user_tagged=tagged, on_message_posted=posted)
        await svc.warm_up()
        for rep in range(reps):
            for p in PROMPTS:
                stage = ResponseStage(svc.chat_orchestrator, svc.build_system_prompt)
                log.turns.clear()
                t0 = time.perf_counter()
                first_spoken = None
                chunks = []
                async for c in stage.process_stream(p.text, PipelineContext(
                        speaker_name="Tester", speaker_activity="Playing Dune: Awakening")):
                    if first_spoken is None and c.strip():
                        first_spoken = time.perf_counter() - t0
                    chunks.append(c)
                total = time.perf_counter() - t0
                hist = stage.get_messages()
                tools = [tc["function"]["name"] for m in hist for tc in (m.get("tool_calls") or [])]
                answer = "".join(chunks).strip()
                row = {
                    "effort": effort, "rep": rep, "prompt": p.name, "kind": p.kind,
                    "first_spoken_s": round(first_spoken or total, 2), "total_s": round(total, 2),
                    "llm_calls": len(log.turns),
                    "llm_ttft_s": [t[0] for t in log.turns],
                    "reasoning_chars": sum(t[2] for t in log.turns),
                    "tools": tools, "correct": p.check(answer, tools),
                    "issues": format_issues(answer), "answer": answer,
                }
                rows.append(row)
                print(f"{effort:6} {p.name:10} rep{rep} {row['first_spoken_s']:5.2f}s "
                      f"{'ok ' if row['correct'] else 'BAD'} {row['issues'] or ''} "
                      f"{tools or ''} {answer[:70]!r}", flush=True)
        await svc.chat_orchestrator._llm.close()
    return rows


SEARCH_QUERIES = ["current weather in Chicago", "Singapore Grand Prix 2026 winner",
                  "heavy darts recipe dune awakening", "capital of Australia",
                  "how many legs does a spider have", "cerebras qwen 3.8 27b",
                  "best pizza dough hydration", "python 3.13 release date"]
SEARCH_BACKENDS = ["auto", "duckduckgo", "brave", "bing", "google", "mojeek", "wikipedia"]


async def run_search() -> list[dict]:
    from ddgs import DDGS

    def one(backend, q):
        t0 = time.perf_counter()
        try:
            hits = DDGS(timeout=10).text(q, region="us-en", max_results=5, backend=backend) or []
            err = None
        except Exception as e:
            hits, err = [], f"{type(e).__name__}: {str(e)[:80]}"
        return {"backend": backend, "query": q, "s": round(time.perf_counter() - t0, 2),
                "results": len(hits), "error": err}

    rows = []
    for q in SEARCH_QUERIES:
        for b in SEARCH_BACKENDS:
            try:
                row = await asyncio.wait_for(asyncio.to_thread(one, b, q), timeout=40)
            except TimeoutError:
                row = {"backend": b, "query": q, "s": 40.0, "results": 0, "error": "hung >40s"}
            rows.append(row)
            print(f"search {b:10} {row['s']:6.2f}s {row['results']} results "
                  f"{row['error'] or ''} | {q}", flush=True)
    return rows


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else float("nan")


def report(llm_rows, search_rows, out: Path):
    lines = ["# LLM + search benchmark", "", f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if llm_rows:
        lines += ["## Cerebras qwen-3.8-27b by reasoning level", "",
                  "Through the real tool loop and voice system prompt. \"First spoken\" = time until "
                  "the first non-blank text chunk (what TTS can start on), including tool round trips.",
                  "", "| Effort | Correct | Median first spoken | p90 first spoken | Median (no tools) | "
                  "Median LLM TTFT | Max LLM TTFT | Reasoning chars (median) | Format issues |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for e in EFFORTS:
            rs = [r for r in llm_rows if r["effort"] == e]
            if not rs:
                continue
            fs = [r["first_spoken_s"] for r in rs]
            plain = [r["first_spoken_s"] for r in rs if r["kind"] != "tool"]
            ttft = [t for r in rs for t in r["llm_ttft_s"]]
            issues = sum(bool(r["issues"]) for r in rs)
            lines.append(
                f"| {e} | {sum(r['correct'] for r in rs)}/{len(rs)} | {statistics.median(fs):.2f}s | "
                f"{pct(fs, .9):.2f}s | {statistics.median(plain):.2f}s | "
                f"{statistics.median(ttft):.2f}s | {max(ttft):.2f}s | "
                f"{int(statistics.median([r['reasoning_chars'] for r in rs]))} | {issues}/{len(rs)} |")
        lines += ["", "### Per prompt (correct / median first spoken)", "",
                  "| Prompt | " + " | ".join(EFFORTS) + " |", "|---|" + "---|" * len(EFFORTS)]
        for p in PROMPTS:
            cells = []
            for e in EFFORTS:
                rs = [r for r in llm_rows if r["effort"] == e and r["prompt"] == p.name]
                if rs:
                    cells.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}, "
                                 f"{statistics.median(r['first_spoken_s'] for r in rs):.2f}s")
                else:
                    cells.append("-")
            lines.append(f"| {p.name} | " + " | ".join(cells) + " |")
        lines += ["", "### Wrong or badly formatted answers", ""]
        for r in llm_rows:
            if not r["correct"] or r["issues"]:
                lines.append(f"- {r['effort']} / {r['prompt']} rep{r['rep']}: "
                             f"{'WRONG' if not r['correct'] else ''} {r['issues'] or ''} "
                             f"tools={r['tools']} — {r['answer'][:200]!r}")
        lines.append("")
    if search_rows:
        lines += ["## Web search backends (ddgs)", "",
                  "| Backend | Median | p90 | Max | Failures / empty |", "|---|---|---|---|---|"]
        for b in SEARCH_BACKENDS:
            rs = [r for r in search_rows if r["backend"] == b]
            s = [r["s"] for r in rs]
            bad = sum(1 for r in rs if r["error"] or not r["results"])
            lines.append(f"| {b} | {statistics.median(s):.2f}s | {pct(s, .9):.2f}s | {max(s):.2f}s | "
                         f"{bad}/{len(rs)} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def main(args):
    from src.config import get_settings

    what = args[0] if args and args[0] in ("llm", "search") else "all"
    reps = int(args[1]) if len(args) > 1 else 3
    out = ROOT / "reports" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("src.llm.cerebras_client").setLevel(logging.INFO)

    search_rows = await run_search() if what in ("all", "search") else []
    llm_rows = await run_llm(get_settings(), reps) if what in ("all", "llm") else []
    (out / "results.json").write_text(json.dumps({"llm": llm_rows, "search": search_rows}, indent=2),
                                      encoding="utf-8")
    report(llm_rows, search_rows, out)
    print(f"\nReport: {out / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
