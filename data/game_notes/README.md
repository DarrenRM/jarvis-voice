# Game notes

Official patch notes for the games the group plays. While someone in the voice channel is
playing a game that has a set here, the notes go into the bot's system prompt, so answers
about a fresh release don't rest on the model's out-of-date memory. See
`src/llm/game_notes.py` for how they are matched and injected.

Each game is a folder with an `info.json` and the notes as a text file:

```json
{"game": "Enshrouded", "version": "1.0", "released": "2026-10-15", "notes": "1.0.txt",
 "aliases": ["enshrouded"], "misheard": ["in shrouded"], "source": "the developer"}
```

The notes themselves are the publisher's text, so `*.txt` files here are gitignored and
not part of this repository. Paste the notes into the file named by `notes` to enable a set;
a folder whose notes file is missing is skipped with a log warning.
