"""Real voice audio kept on disk for the IRL test set (MONITOR_SAVE_AUDIO).

- turns/: each question's audio, exactly what Whisper was given, with a JSON sidecar
  (what was heard, the reply, outcome, timings, findings, and later the owner's
  reaction to the monitor post).
- wake/: about 1.5 s around every wake word hit and every near miss scoring at least
  MONITOR_WAKE_CLIP_MIN_SCORE, for tuning the wake word.

Files are 16 kHz mono 16-bit WAV. Local only (the folder is gitignored); once it
passes MONITOR_AUDIO_MAX_MB the oldest recordings are deleted. Writes happen on a
worker thread so neither the event loop nor the audio thread waits on the disk.
`scripts/e2e/run_voice_e2e.py --recordings` plays them back through the real pipeline
and adds the wake clips to its wake word sweep.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import SAMPLE_RATE_PROCESSING

logger = logging.getLogger(__name__)

TURNS = "turns"
WAKE = "wake"


def _slug(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "unknown").lower()).strip("-")[:24] or "unknown"


def write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE_PROCESSING) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono WAV -> float32 in [-1, 1] (at the file's own rate)."""
    with wave.open(str(path), "rb") as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32767


class AudioArchive:
    def __init__(self, root: Path, max_mb: float = 1000.0, wake_min_score: float = 0.2,
                 names: dict[int, str] | None = None):
        self.root = root
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.wake_min_score = wake_min_score
        self._names = names or {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recordings")
        self._lock = threading.Lock()  # unique names across threads

    def _reserve(self, folder: str, stem: str) -> Path:
        """A free path for stem (adds -2, -3... if this second already has one)."""
        directory = self.root / folder
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            path, n = directory / f"{stem}.wav", 1
            while path.exists():
                n += 1
                path = directory / f"{stem}-{n}.wav"
            path.touch()  # claim it before the worker writes it
        return path

    def save_turn(self, turn: Any, findings: list[Any]) -> Path | None:
        """Queue the turn's question audio and sidecar. Returns the .wav path."""
        audio = getattr(turn, "question_audio", None)
        if audio is None or len(audio) == 0:
            return None
        stem = f"{turn.started:%Y%m%d-%H%M%S}_{_slug(turn.speaker_name)}_{turn.outcome}"
        path = self._reserve(TURNS, stem)
        meta = {
            "time": turn.started.isoformat(timespec="seconds"),
            "speaker_id": turn.speaker_id,
            "speaker_name": turn.speaker_name,
            "raw_heard": turn.raw_heard,
            "heard": turn.heard,
            "reply": turn.reply,
            "outcome": turn.outcome,
            "audio_seconds": round(len(audio) / SAMPLE_RATE_PROCESSING, 2),
            "timings": {
                "stt_s": turn.stt_s,
                "first_audio_s": turn.first_audio_s,
                "answer_audio_s": turn.answer_audio_s,
                "duration_s": turn.duration_s,
                "llm_calls": turn.llm_calls,
            },
            "tools": [asdict(t) for t in turn.tools],
            "findings": [{"detector": f.detector, "summary": f.summary, "detail": f.detail}
                         for f in findings],
            "events": [asdict(e) for e in turn.events],
        }
        self._pool.submit(self._write, path, audio.copy(), meta)
        return path

    def save_wake(self, user_id: int, audio: np.ndarray, score: float, hit: bool) -> None:
        """Queue a wake word clip (any thread). Near misses below wake_min_score are skipped."""
        if audio is None or len(audio) == 0 or (not hit and score < self.wake_min_score):
            return
        now = datetime.now()
        name = self._names.get(user_id) or str(user_id)
        kind = "hit" if hit else "miss"
        stem = f"{now:%Y%m%d-%H%M%S}_{_slug(name)}_{kind}_{score:.2f}"
        meta = {"time": now.isoformat(timespec="seconds"), "user_id": user_id,
                "speaker_name": self._names.get(user_id), "score": round(score, 3), "hit": hit}
        try:
            self._pool.submit(self._write_new, WAKE, stem, audio.copy(), meta)
        except RuntimeError:
            pass  # closed (shutting down)

    def note_reaction(self, wav_path: Path, reaction: str, discord_url: str = "") -> None:
        """Record the owner's reaction to the monitor post about this recording."""
        self._pool.submit(self._update, wav_path.with_suffix(".json"),
                          {"owner_reaction": reaction, "discord": discord_url})

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------ worker thread

    def _write_new(self, folder: str, stem: str, audio: np.ndarray, meta: dict) -> None:
        try:
            path = self._reserve(folder, stem)
        except Exception as e:
            logger.warning(f"[RECORDINGS] Could not save {stem}: {e}")
            return
        self._write(path, audio, meta)

    def _write(self, path: Path, audio: np.ndarray, meta: dict) -> None:
        try:
            write_wav(path, audio)
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str),
                                                 encoding="utf-8")
            self._prune()
        except Exception as e:
            logger.warning(f"[RECORDINGS] Could not save {path.name}: {e}")

    def _update(self, sidecar: Path, fields: dict) -> None:
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            meta.update({k: v for k, v in fields.items() if v})
            sidecar.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[RECORDINGS] Could not update {sidecar.name}: {e}")

    def _prune(self) -> None:
        files = sorted((p for p in self.root.rglob("*") if p.is_file()),
                       key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
