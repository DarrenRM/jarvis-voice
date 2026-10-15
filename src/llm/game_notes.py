"""Official game notes (patch notes) added to the prompt while that game is in play.

Right after a big launch the model's own knowledge is out of date and so are most
web pages. data/game_notes/<game>/ holds the official notes plus an info.json:

    {"game": "Enshrouded", "version": "1.0", "released": "2026-10-15",
     "notes": "1.0.txt", "aliases": ["enshrouded"], "source": "...",
     "misheard": ["in shrouded", "enshrowded"]}

"misheard" lists how speech-to-text gets the name wrong; transcripts are corrected
before the model sees them (corrections()), so searches use the real name.

The whole notes file goes into the first system prompt (Qwen only accepts one, first)
when the speaker is playing the game, the question names it, or it came up in the last
few minutes. Measured on Cerebras: 15.5k prompt tokens add ~0.1-0.3 s to the first
spoken word, and broad questions ("what's new in 1.0?") get proper answers, which
keyword search over the notes would not.
"""

import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "game_notes"
STICKY_SECONDS = 300.0  # follow-ups ("and the new biome?") rarely name the game again


@dataclass
class GameNoteSet:
    game: str
    version: str
    released: dt.date
    text: str
    aliases: list[str]
    source: str
    misheard: list[str]

    def mentioned_in(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(rf"\b{re.escape(name)}\b", low)
                   for name in self.aliases + self.misheard)

    def block(self, today: dt.date) -> str:
        when = self.released.strftime("%B %d, %Y").replace(" 0", " ")
        out = "came out on" if today >= self.released else "comes out on"
        return (
            f"OFFICIAL {self.game.upper()} {self.version} PATCH NOTES, from {self.source}. "
            f"{self.game} {self.version} {out} {when}. What you remember about {self.game} "
            f"is from before {self.version} and is often out of date, and so are many web "
            f"pages for a while. These notes are official: where your memory or a web page "
            f"disagrees with them, the notes are right. Answer from them when they cover the "
            f"question. They list what changed, not full recipes or locations, so search for "
            f"those; {self.version} reworked many recipes, so say that amounts from older "
            f"pages may have changed.\n<<<\n{self.text.strip()}\n>>>"
        )


class GameNotes:
    """All note sets, and which of them belong in this turn's prompt."""

    def __init__(self, sets: list[GameNoteSet], clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], dt.date] = dt.date.today):
        self.sets = sets
        self._clock = clock
        self._today = today
        self._last_seen: dict[str, float] = {}  # game -> when it last came up

    @classmethod
    def load(cls, root: Path = DEFAULT_ROOT, **kwargs) -> "GameNotes":
        sets = []
        for info_path in sorted(root.glob("*/info.json")) if root.is_dir() else []:
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                text = (info_path.parent / info["notes"]).read_text(encoding="utf-8")
                sets.append(GameNoteSet(
                    game=info["game"], version=str(info["version"]),
                    released=dt.date.fromisoformat(info["released"]), text=text,
                    aliases=[a.lower() for a in info.get("aliases") or [info["game"]]],
                    source=info.get("source", "the developer"),
                    misheard=[m.lower() for m in info.get("misheard") or []],
                ))
            except Exception as e:
                logger.warning(f"[NOTES] Skipping {info_path.parent.name}: {e}")
        if sets:
            logger.info("[NOTES] Loaded " + ", ".join(
                f"{s.game} {s.version} ({len(s.text) // 1000}k chars)" for s in sets))
        return cls(sets, **kwargs)

    def corrections(self) -> dict[str, str]:
        """Misheard game name -> real name, for the transcription stage."""
        return {wrong: notes.game for notes in self.sets for wrong in notes.misheard}

    def prompt_block(self, text: str, activity: Optional[str],
                     channel_activities: Sequence[str] = ()) -> str:
        """Notes for games in play this turn ("" if none). channel_activities: what the
        others in the voice channel are doing (a friend playing it counts too: not
        everyone shows their game on Discord)."""
        now = self._clock()
        blocks = []
        for notes in self.sets:
            why = ("playing it" if activity and notes.mentioned_in(activity)
                   else "someone in the channel is playing it"
                   if any(notes.mentioned_in(a) for a in channel_activities if a)
                   else "named" if notes.mentioned_in(text)
                   else "came up recently"
                   if now - self._last_seen.get(notes.game, -1e9) < STICKY_SECONDS
                   else None)
            if why is None:
                continue
            self._last_seen[notes.game] = now
            logger.info(f"[NOTES] {notes.game} {notes.version} notes in the prompt ({why})")
            blocks.append(notes.block(self._today()))
        return "\n\n".join(blocks)
