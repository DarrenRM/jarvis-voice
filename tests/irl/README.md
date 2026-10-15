# IRL regression tests

Real incidents from the bot's logs, replayed offline through the real code (sink,
VoiceSession turn/follow-up loops, addressee judge, transcription and TTS text paths) so
fixed bugs stay fixed. No network, keys or model downloads; voice timelines run at
`SCALE` (0.1x) against the real `Settings` defaults. Each test's docstring says when it
happened and what went wrong.

Adding a case: put the real data (who, ids, what was said, timings, expected outcome) in
`cases.py`, marking anything the log didn't record as assumed or paraphrased, then add a
test next to the similar ones; `replay.py` has the helpers (`Room.say` sends speech to
the sink as Discord packets, `ScriptedPipeline` stands in for the LLM/TTS pipeline).

This is layer 1. Live evals against the real services live elsewhere: the addressee
judge eval in `scripts/irl/` and the IRL voice scenarios in `scripts/e2e`.
