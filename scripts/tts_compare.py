"""Side-by-side Inworld TTS model comparison: same bot lines, each model, with timings.

Writes reports/tts-compare/index.html with playable clips (embedded WAVs) and a price table.

    PYTHONPATH=. .venv/Scripts/python scripts/tts_compare.py
"""

import asyncio
import base64
import io
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = "https://api.inworld.ai/tts/v1/voice:stream"
RATE = 24000

# Lines the bot actually says, tags included (the prompt asks for them)
LINES = [
    "Sure Darren...",
    "[happy] It's four.",
    "[laughing] Oh, you're really warming up on me, huh?",
    "The glider needs eight shroud wood, crafted at the workbench.",
    "[whispering] Honestly, it's mostly hype right now.",
    "[sigh] Fine. Enshrouded supports up to sixteen players in co-op. [laugh] Just don't let "
    "Riley build the base.",
]

# (label, model, extra request fields, $ per 1M chars on-demand / Creator plan)
CONFIGS = [
    ("1.5 mini (now)", "inworld-tts-1.5-mini", {"temperature": 1.1}, None),
    ("TTS-2 balanced", "inworld-tts-2", {"deliveryMode": "BALANCED"}, (25, 20)),
    ("TTS-2 creative", "inworld-tts-2", {"deliveryMode": "CREATIVE"}, (25, 20)),
    ("TTS-2 Flash", "inworld-tts-2-flash", {}, (15, 10)),
]


async def synth(client: httpx.AsyncClient, key: str, model: str, extra: dict, text: str):
    body = {"text": text, "voiceId": "Sarah", "modelId": model,
            "audioConfig": {"encoding": "LINEAR16", "sampleRateHertz": RATE,
                            "speakingRate": 1.1}, **extra}
    started = time.perf_counter()
    first = None
    pcm = bytearray()
    async with client.stream("POST", URL, json=body,
                             headers={"Authorization": f"Basic {key}"}) as response:
        if response.status_code != 200:
            raise RuntimeError(f"{model}: HTTP {response.status_code} "
                               f"{(await response.aread())[:200]!r}")
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            result = chunk.get("result", chunk)
            audio_b64 = result.get("audioContent") or result.get("audio")
            if not audio_b64:
                continue
            if first is None:
                first = time.perf_counter() - started
            data = base64.b64decode(audio_b64)
            if data[:4] == b"RIFF":
                data = data[44:]  # each chunk carries its own WAV header
            pcm += data
    return first, time.perf_counter() - started, bytes(pcm)


def wav_b64(pcm: bytes) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


async def main():
    from src.config import get_settings

    key = get_settings().inworld_api_key
    results = {}  # (label, line index) -> (ttfb, total, pcm) or error
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        for label, model, extra, _ in CONFIGS:
            try:
                await synth(client, key, model, extra, "Hi.")  # warm the connection
            except Exception as e:
                print(f"{label}: warm-up failed: {e}")
            for i, line in enumerate(LINES):
                try:
                    results[label, i] = await synth(client, key, model, extra, line)
                    ttfb, total, pcm = results[label, i]
                    print(f"{label:16} ttfb {ttfb:.2f}s total {total:.2f}s "
                          f"audio {len(pcm) / 2 / RATE:.1f}s | {line[:50]}")
                except Exception as e:
                    results[label, i] = str(e)
                    print(f"{label:16} FAILED {e}")
    write_page(results)


def write_page(results):
    out = ROOT / "reports" / "tts-compare"
    out.mkdir(parents=True, exist_ok=True)
    head = "".join(f"<th>{label}</th>" for label, *_ in CONFIGS)
    rows = []
    for i, line in enumerate(LINES):
        cells = []
        for label, *_ in CONFIGS:
            r = results.get((label, i))
            if isinstance(r, tuple):
                ttfb, total, pcm = r
                cells.append(f'<td><audio controls preload="none" src="data:audio/wav;base64,'
                             f'{wav_b64(pcm)}"></audio><div class="m">first audio '
                             f'{ttfb * 1000:.0f} ms &middot; {len(pcm) / 2 / RATE:.1f}s</div></td>')
            else:
                cells.append(f'<td class="err">{r or "missing"}</td>')
        rows.append(f"<tr><td class=line>{line}</td>{''.join(cells)}</tr>")
    medians = []
    for label, *_ in CONFIGS:
        t = [r[0] for i in range(len(LINES)) if isinstance(r := results.get((label, i)), tuple)]
        medians.append(f"<td><b>{statistics.median(t) * 1000:.0f} ms</b></td>" if t else "<td>-</td>")
    price_rows = []
    for label, model, _, price in CONFIGS:
        if price is None:
            price_rows.append(f"<tr><td>{label}</td><td>{model}</td><td colspan=3>deprecated, "
                              f"no longer on the pricing page</td></tr>")
            continue
        on_demand, creator = price
        # ~50 spoken answers a day at ~200 chars, plus uncached openers: ~300k chars/month
        price_rows.append(f"<tr><td>{label}</td><td>{model}</td><td>${on_demand}</td>"
                          f"<td>${creator}</td><td>${0.3 * on_demand:.2f}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TTS Model Compare</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e3e3e8; --head:#f5f5f7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#f2f2f4; --muted:#9a9aa1;
  --line:#2c2c30; --head:#1f1f22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif;
  margin:0; padding:24px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:28px 0 8px; }}
p {{ color:var(--muted); margin:0 0 16px; max-width:70ch; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; min-width:100%; }}
th, td {{ border-bottom:1px solid var(--line); padding:10px 12px; text-align:left;
  vertical-align:top; }}
th {{ background:var(--head); font-weight:600; white-space:nowrap; }}
td.line {{ max-width:260px; font-family:ui-monospace, monospace; font-size:13px; }}
audio {{ width:200px; height:34px; }} .m {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.err {{ color:#c0392b; font-size:12px; }}
</style></head><body>
<h1>Inworld TTS models, side by side</h1>
<p>Same lines, voice Sarah at speed 1.1. "First audio" is measured from this PC, network
included. Style tags like [happy], [laughing] and [whispering] are ignored by TTS-2 Flash;
sound tags like [laugh] and [sigh] work on all of them.</p>
<div class="wrap"><table><tr><th>Line</th>{head}</tr>{''.join(rows)}
<tr><td><b>Median first audio</b></td>{''.join(medians)}</tr></table></div>
<h2>Price per 1M characters</h2>
<div class="wrap"><table><tr><th>Config</th><th>Model</th><th>On-demand</th>
<th>Creator plan ($25/mo)</th><th>~300k chars/month, on-demand</th></tr>
{''.join(price_rows)}</table></div>
<p>Monthly estimate assumes about 50 spoken answers a day at ~200 characters. Tags count as
characters. On-demand includes up to 70 free minutes of TTS. Source: inworld.ai/pricing.</p>
</body></html>"""
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")


asyncio.run(main())
