"""Run end-to-end voice scenarios against the real pipeline (live Groq, Cerebras, Inworld).

    .venv/Scripts/python scripts/e2e/run_voice_e2e.py              # all scenarios
    .venv/Scripts/python scripts/e2e/run_voice_e2e.py basic noisy  # name filter
    .venv/Scripts/python scripts/e2e/run_voice_e2e.py --no-web --repeat=3

    .venv/Scripts/python scripts/e2e/run_voice_e2e.py --irl        # only the irl_* group
    .venv/Scripts/python scripts/e2e/run_voice_e2e.py --irl --list # show, don't run

--no-web skips the live web scenarios (web_*); --repeat=N runs each scenario N times.
--irl runs only the irl_* scenarios (real interactions from the live bot, no web).
--list prints the selected scenarios and exits (nothing is called or synthesized).
--recordings[=DIR] adds one rec_* scenario per question clip the live bot saved in
DIR/turns (DIR defaults to recordings/), played as recorded (see recording_scenarios),
and adds DIR/wake clips to the wake word sweep (`--recordings sweep`: sweep only).
--monitor runs each turn through the live monitor (src/monitor) and lists in the report
what it would have posted to Discord (nothing is sent).
--listen plays the run (people and bot) into the voice channel you're in, using the
bot's token: stop the live bot first. MONITOR_OWNER_ID picks your channel if set.

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
    WEB_TOOLS,
    FakeMember,
    Recording,
    RunResult,
    Speech,
    overlap,
    read_sidecar,
    word_error_rate,
)

# Shared with the live monitor (src/monitor)
from src.monitor.text_checks import give_up_phrases, spoken_artifacts  # noqa: E402

TESTER = FakeMember(700000000000000001, "Tester")  # no game: answers stay generic
BYSTANDER = FakeMember(700000000000000002, "Bystander")
MORGAN = FakeMember(700000000000000003, "Morgan")
RILEY = FakeMember(700000000000000004, "Riley", activity="Enshrouded")
# The irl_* scenarios: real people, both "Playing Enshrouded" (so the 1.0 notes are in
# the prompt, as they were live)
DARREN = FakeMember(700000000000000005, "Darren", activity="Enshrouded")
SAM = FakeMember(700000000000000006, "Sam", activity="Enshrouded")
MEMBERS = [TESTER, BYSTANDER, MORGAN, RILEY, DARREN, SAM]


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


def web_searches_at_most(limit: int) -> Check:
    """Keeps live search load (and tokens) down across the whole scenario."""
    def check(r):
        n = sum(t.tools.count("web_search") for t in r.turns)
        return n <= limit, f"web_search calls={n}"
    check.__name__ = f"web_searches<={limit}"
    return check


def answers_confidently(turn: int = 0) -> Check:
    """The bot gave an answer rather than a "couldn't find it" apology."""
    def check(r):
        if len(r.turns) <= turn:
            return False, "no turn recorded"
        hit = give_up_phrases(r.turns[turn].answer)
        return not hit, f"{hit or ''} {r.turns[turn].answer!r}"
    check.__name__ = f"answers_confidently[{turn}]"
    return check


def no_tools(r: RunResult) -> tuple[bool, str]:
    tools = [t for turn in r.turns for t in turn.tools]
    return not tools, f"tools={tools}"


def turns_at_least(n: int) -> Check:
    def check(r):
        return len(r.turns) >= n, f"{len(r.turns)} turns: {[t.user_text for t in r.turns]}"
    check.__name__ = f"turns>={n}"
    return check


def announced_post(r: RunResult) -> tuple[bool, str]:
    said = "post this in chat" in r.bot_transcript.lower()
    return said, "said it" if said else f"never said it: {r.bot_transcript[-120:]!r}"


def no_post(r: RunResult) -> tuple[bool, str]:
    return not r.posts, f"posts={r.posts}" if r.posts else "nothing posted"


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


def heard_includes(*words: str, turn: int = 0) -> Check:
    """The turn's transcript has every one of words (near-miss spellings allowed)."""
    def check(r):
        if len(r.turns) <= turn:
            return False, "no turn recorded"
        text = r.turns[turn].user_text.lower()
        missing = [w for w in words if not _fuzzy_in(w, text)]
        return not missing, f"heard {r.turns[turn].user_text!r}" + (
            f" (missing: {missing})" if missing else "")
    check.__name__ = f"heard_includes[{turn}]({'/'.join(words)})"
    return check


def no_web_lookup(r: RunResult) -> tuple[bool, str]:
    """No web_search or web_extract call (scenarios with web_limit=0 never go online)."""
    calls = [t for turn in r.turns for t in turn.tools if t in WEB_TOOLS]
    return not calls, f"web calls={calls}"


def talked_over_bot(speech: int) -> Check:
    """Setup check: the speech started while the bot was still playing something."""
    def check(r):
        start = r.speeches[speech].started_at
        playing = [s for s in r.segments if s.start <= start < s.end]
        return bool(playing), (f"started {start:.2f}s, over {playing[0].transcript!r}"
                               if playing else f"started {start:.2f}s, bot was quiet")
    check.__name__ = f"talked_over_bot[{speech}]"
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
    problems = spoken_artifacts(r.bot_transcript, " ".join(t.answer for t in r.turns))
    return not problems, f"spoken artifacts: {problems}" if problems else "clean"


def no_error(r: RunResult) -> tuple[bool, str]:
    apology = "encountered an error" in r.bot_transcript.lower()
    return not r.error and not apology, r.error or ("bot apologized for an error" if apology else "ok")


# Follow-ups without the wake word (speech index = position in the scenario's speeches)


def turns_exactly(n: int) -> Check:
    def check(r):
        return len(r.turns) == n, f"{len(r.turns)} turns: {[t.user_text for t in r.turns]}"
    check.__name__ = f"turns=={n}"
    return check


def asked_question(turn: int = 0) -> Check:
    def check(r):
        from src.llm.addressee_judge import is_question

        if len(r.turns) <= turn:
            return False, "no turn recorded"
        answer = r.turns[turn].answer
        return is_question(re.sub(r"\[[^\]]*\]", "", answer)), f"answer {answer!r}"
    check.__name__ = f"bot_asked[{turn}]"
    return check


def plain_answer(turn: int = 0) -> Check:
    """Setup check: the bot answered without asking anything back (the short window)."""
    asked = asked_question(turn)

    def check(r):
        ok, detail = asked(r)
        return (not ok and detail != "no turn recorded"), detail
    check.__name__ = f"plain_answer[{turn}]"
    return check


def old_window_closed(speech: int, window: float) -> Check:
    """Setup check: the speech started more than `window` seconds after the bot last
    stopped talking, so the old fixed follow-up window would have closed."""
    def check(r):
        start = r.speeches[speech].started_at
        before = [s.end for s in r.segments if s.end <= start]
        if not before:
            return False, "bot never spoke before it"
        gap = start - max(before)
        return gap > window, f"started {gap:.1f}s after the bot stopped"
    check.__name__ = f"old_window_closed[{speech}]({window}s)"
    return check


def _after(r: RunResult, speech: int) -> list:
    end = r.speeches[speech].ended_at
    return [s for s in r.segments if s.start >= end - 0.2]


def quiet_after(speech: int) -> Check:
    def check(r):
        after = _after(r, speech)
        return not after, " | ".join(s.transcript for s in after) or "bot stayed quiet"
    check.__name__ = f"quiet_after[{speech}]"
    return check


def replied_within(speech: int, seconds: float) -> Check:
    def check(r):
        after = _after(r, speech)
        if not after:
            return False, "no reply"
        delay = after[0].start - r.speeches[speech].ended_at
        return delay <= seconds, f"first audio {delay:.2f}s after they stopped: {after[0].transcript!r}"
    check.__name__ = f"replied_within[{speech}]({seconds}s)"
    return check


def no_affirmation_after(speech: int) -> Check:
    """Follow-ups skip the "Sure, <name>..." opening (the name only appears there)."""
    def check(r):
        after = _after(r, speech)
        if not after:
            return False, "no reply"
        name = r.speeches[speech].user.display_name.lower()
        first = after[0].transcript
        return name not in first.lower(), f"first clip {first!r}"
    check.__name__ = f"no_affirmation[{speech}]"
    return check


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
    web_limit: int | None = None  # web lookups allowed; past it the tool errors offline


Q_CAPITAL = "what's the capital of Australia?"
TRIVIA = "Hey Jarvis, ask me an easy trivia question."
TIMES = "Hey Jarvis, what's twelve times eight?"

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
    # The only web scenario: two game questions, at most two searches in the whole run
    Scenario("web_enshrouded",
             [Speech(RILEY, ["Hey Jarvis, how many people can play together in Enshrouded?"]),
              Speech(RILEY, ["Hey Jarvis, what do I need to craft a glider?"])],
             [used_tool("web_search"), says_any("16", "sixteen"), answers_confidently(),
              answers_confidently(turn=1), web_searches_at_most(2)], between=1.5,
             note="Live web lookup for game info (the bot's main use). The second question doesn't "
                  "name the game, so it has to come from Riley's 'Playing Enshrouded' activity."),
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
    # Small talk: no tools, short, in character; read the answers in the report
    Scenario("smalltalk_day", [Speech(TESTER, ["Hey Jarvis, how's your day going?"])],
             [no_tools, answers_confidently()], note="Chit-chat; should answer without tools."),
    Scenario("smalltalk_joke", [Speech(TESTER, ["Hey Jarvis, tell me a joke about zombies."])],
             [no_tools], note="Short joke, no tools."),
    Scenario("smalltalk_opinion", [Speech(TESTER, ["Hey Jarvis, is pineapple on pizza a crime?"])],
             [no_tools], note="Opinion question; personality, no tools."),
    Scenario("smalltalk_thanks", [Speech(TESTER, ["Hey Jarvis, thanks for the help earlier."])],
             [no_tools], note="Should be a brief reply."),
    Scenario("arithmetic",
             [Speech(TESTER, ["Hey Jarvis, if I have three stacks of 64 iron and use 50, "
                              "how much is left?"])],
             [no_tools, says_any("142", "hundred forty-two", "hundred and forty-two",
                                 "hundred forty two")],
             note="Mental maths with reasoning, no tools."),
    Scenario("followup_no_wake",
             [Speech(TESTER, ["Hey Jarvis, what's your favorite color?"]),
              Speech(TESTER, ["Why that one?"], after_reply=1.0)],
             [turns_at_least(2), no_tools], between=1.0,
             note="Second question has no wake word, 1 s after the answer ends: "
                  "inside the follow-up window, so it's answered."),
    Scenario("quick_second_question",
             [Speech(TESTER, ["Hey Jarvis, what's two plus two?"]),
              Speech(TESTER, ["Hey Jarvis, and what's ten times ten?"])],
             [says_any("four", "4"), says_any("hundred", "100", turn=1)], between=0.3,
             note="Second question right after the first answer ends."),
    Scenario("barge_in",
             [Speech(TESTER, ["Hey Jarvis, tell me a short story about a dragon."], at=0.5),
              Speech(TESTER, ["Hey Jarvis, stop."], at=9.0)],
             [no_llm_turn], answers=False, expect_fail=True,
             note="Known gap: no barge-in. Records what happens when the user interrupts."),

    # ---- Follow-ups without the wake word. After the bot asks something, the asker has
    # FOLLOWUP_WINDOW_SECONDS (10) to start replying; after a plain answer they have
    # FOLLOWUP_AFTER_ANSWER_SECONDS (4), and only a new question or request counts.
    Scenario("fu_asked_reply",
             [Speech(TESTER, [TRIVIA]),
              Speech(TESTER, ["I'm going to guess Paris."], after_reply=1.0)],
             [asked_question(0), turns_exactly(2), replied_within(1, 3.0),
              no_affirmation_after(1)], between=1.0,
             note="Bot asks; the asker answers without the wake word (not a question)."),
    Scenario("fu_asked_crosstalk",
             [Speech(TESTER, [TRIVIA]),
              Speech(TESTER, ["Riley, grab the loot by the door before we go."], after_reply=1.0)],
             [asked_question(0), turns_exactly(1), quiet_after(1)], between=1.0,
             note="Bot asks, but the asker turns to a friend: the check must say not for the bot."),
    Scenario("fu_asked_other_person",
             [Speech(TESTER, [TRIVIA]),
              Speech(RILEY, ["I'm going to guess Paris."], after_reply=1.0, voice="dennis")],
             [asked_question(0), turns_exactly(1), quiet_after(1)], between=1.0,
             note="Someone else answers the bot's question: only the asker is heard."),
    Scenario("fu_clarify",
             [Speech(TESTER, ["Hey Jarvis, convert thirty degrees for me."]),
              Speech(TESTER, ["Celsius to Fahrenheit."], after_reply=1.0)],
             [asked_question(0), says_any("86", "eighty-six", "eighty six", turn=1)],
             between=1.0,
             note="Ambiguous request: the prompt should make it ask which way first."),
    Scenario("fu_answer_new_question",
             [Speech(TESTER, [TIMES]),
              Speech(TESTER, ["And what's that divided by four?"], after_reply=1.0)],
             [says_any("96", "ninety-six", "ninety six"),
              says_any("24", "twenty-four", "twenty four", turn=1),
              replied_within(1, 3.0), no_affirmation_after(1)], between=1.0,
             note="After a plain answer, a new question without the wake word is answered."),
    Scenario("fu_answer_chatter",
             [Speech(TESTER, [TIMES]),
              Speech(TESTER, ["Nice, told you it was easy."], after_reply=1.0)],
             [turns_exactly(1), quiet_after(1)], between=1.0,
             note="After a plain answer, a reaction to friends gets no reply."),
    Scenario("fu_answer_thanks",
             [Speech(TESTER, [TIMES]),
              Speech(TESTER, ["Okay cool, thanks."], after_reply=1.0)],
             [turns_exactly(1), quiet_after(1)], between=1.0,
             note="Said to the bot, but not a question or request: no reply."),
    Scenario("fu_answer_window_closed",
             [Speech(TESTER, [TIMES]),
              Speech(TESTER, ["And what's that divided by four?"], after_reply=6.0)],
             [turns_exactly(1), quiet_after(1)], between=1.0,
             note="Starts after the 4 s window: needs the wake word again."),
    Scenario("fu_answer_late_start",
             [Speech(TESTER, [TIMES]),
              Speech(TESTER, ["And if I split that evenly between four people, "
                              "how many would each of us get?"], after_reply=3.0)],
             [says_any("24", "twenty-four", "twenty four", turn=1)], between=1.0,
             note="Starts inside the window and keeps talking past its end: still heard."),
    Scenario("fu_other_wake_in_window",
             [Speech(TESTER, [TRIVIA]),
              Speech(RILEY, ["Hey Jarvis, what's two plus two?"], after_reply=1.0, voice="dennis")],
             [asked_question(0), turns_exactly(2), says_any("four", "4", turn=1),
              replied_within(1, 4.0)], between=1.0,
             note="Someone else says the wake word during the window: their turn starts now."),

    # ---- Reference answers also posted to chat (Jev decides, Qwen writes the post)
    Scenario("post_recipe",
             [Speech(TESTER, ["Hey Jarvis, what do I need to craft a ghost glider in Enshrouded?"])],
             [posted(r"glider"), announced_post],
             note="Crafting recipe: answered out loud, announced, and posted (may search)."),
    Scenario("post_ingredients",
             [Speech(TESTER, ["Hey Jarvis, what ingredients do I need for basic pancakes?"])],
             [says_any("flour"), no_post],
             note="Ingredient list from the model's memory, nothing looked up: no post."),
    Scenario("post_not_for_small_talk",
             [Speech(TESTER, [TIMES])],
             [no_post], note="A single number: no post."),

    # ---- Official patch notes in the prompt while the game is in play (Riley: Enshrouded)
    Scenario("notes_max_level",
             [Speech(RILEY, ["Hey Jarvis, what's the new max level?"], voice="dennis")],
             [says_any("50", "fifty"), web_searches_at_most(0)],
             note="Answered straight from the 1.0 notes (50), no search needed."),
    Scenario("notes_new_region",
             [Speech(RILEY, ["Hey Jarvis, what's in Cinder's Cradle?"], voice="dennis")],
             [says_any("lava", "floating", "island"), web_searches_at_most(1)],
             note="Open-ended question about the new region, from the notes."),
    Scenario("notes_recipe_caveat",
             [Speech(RILEY, ["Hey Jarvis, what do I need to craft a ghost glider?"],
                     voice="dennis")],
             [used_tool("web_search"), says_any("changed", "change", "updated", "older", "rework")],
             note="Recipe from the web, with a warning that 1.0 may have changed it."),
]


# ---------------------------------------------------------------- IRL scenarios
# Real interactions from the live bot (2026-10-15, Darren and Sam playing Enshrouded
# on 1.0 launch day), replayed with their real timing. Synthesized for now; swap a text
# part for Recording("recordings/turns/<...>.wav") once the live bot saves the clips.
# None of them may search: web_limit=0 blocks lookups offline, no_web_lookup flags them.

DULUTH_Q = "What lake is Duluth, Minnesota on again? I always forget."
TOOL_Q = "Hey Jarvis, what can I use instead of a hammer to hang a picture?"
CUP_Q = "Hey Jarvis, how many ounces are in a cup?"
CIRCLE_Q = "Hey Jarvis, what does the circle above an enemy mean in action games?"
CIRCLE_FIX = ("No, I mean the circle above their head that looks like they're stunned, "
              "they don't attack at all.")
SOURDOUGH_Q = "Hey Jarvis, how does a sourdough starter work?"
# A friend still in the fight while the bot waits for Darren (~9-10 s and ~5 s)
SAM_LONG = ["Hang on, I'm still fighting this thing.", 0.4,
               "It keeps doing that spin attack and I can't get behind it.", 0.5,
               "Okay, okay, it's down. Let me grab the loot before we head back."]
SAM_SHORT = ["Wait, there's another one behind you.", 0.4,
                "No, the big one, by the tree. Okay, got it."]

IRL_SCENARIOS = [
    Scenario("irl_cut_off_duluth",
             [Speech(SAM, ["Hey Jarvis.", 3.5, "Sup.", 3.0, DULUTH_Q], voice="theodore")],
             # Whisper can mishear the town name; without web the model may guess the lake, so
             # this checks the cut-off (one turn, the whole question), not geography.
             # The cancelled "Sup" answer must not be in history (turns_exactly).
             [turns_exactly(1), heard_includes("lake", "minnesota")],
             web_limit=0,
             note="Real: 2026-10-15 10:26, Sam said 'Hey Jarvis.', waited ~3.5 s while the "
                  "bot played its ack, said 'Sup.', paused ~3 s, then asked about Duluth, Minnesota "
                  "(probably the weather; a no-web question of the same shape here). The turn "
                  "ended after 'Sup', the bot answered that alone and the question was lost. "
                  "With the 2 s end-of-question pause and the carry-on restart it must be one "
                  "turn that answers the question."),
    Scenario("irl_correction_followup",
             [Speech(DARREN, [CIRCLE_Q], voice="dennis"),
              Speech(DARREN, [CIRCLE_FIX], after_reply=1.0, voice="dennis")],
             # 6s: the 2s end-of-question pause, STT, judge, LLM and TTS
             [plain_answer(0), turns_exactly(2), replied_within(1, 6.0), no_affirmation_after(1),
              says_any("stun", "daze", "stagger", turn=1)],  # it may try a (blocked) lookup
             between=1.0, web_limit=0,
             note="Real: 2026-10-15 12:12, Darren asked a game question (Enshrouded; a no-web "
                  "question of the same kind here), the answer missed what he meant, and he "
                  "corrected it 1 s later without the wake word ('No, I mean the circle above "
                  "their head...'). The bot ignored the correction: the "
                  "follow-up judge only let new questions through. It now asks whether the "
                  "speech calls for a reply, so a correction must be answered. (plain_answer "
                  "is a setup check: if the bot asked something back, the 4 s after-answer path "
                  "wasn't exercised.)"),
    Scenario("irl_ack_stays_quiet",
             [Speech(DARREN, [TOOL_Q], voice="dennis"),
              Speech(DARREN, ["Well, I guess that works."], after_reply=1.0, voice="dennis")],
             [turns_exactly(1), quiet_after(1), no_web_lookup], between=1.0, web_limit=0,
             note="Real: 2026-10-15 10:25, Darren reacted 'Well, I guess that works.' right "
                  "after an answer. Correctly ignored live; keep it that way (a reaction, not "
                  "a request)."),
    Scenario("irl_ack_stays_quiet_okay",
             [Speech(SAM, [CUP_Q], voice="theodore"),
              Speech(SAM, ["Okay."], after_reply=1.0, voice="theodore")],
             [turns_exactly(1), quiet_after(1), no_web_lookup], between=1.0, web_limit=0,
             note="Real: 2026-10-15 10:26:52, Sam said just 'Okay.' after an answer. "
                  "Correctly ignored live; keep it that way."),
    Scenario("irl_others_talk_then_reply",
             [Speech(DARREN, [TRIVIA], voice="dennis"),
              Speech(SAM, SAM_LONG, after_reply=0.3, voice="theodore"),
              Speech(DARREN, ["I'm going to guess Paris."], after_prev=1.0, voice="dennis")],
             [asked_question(0), old_window_closed(2, 10.0), turns_exactly(2),
              replied_within(2, 6.0), no_affirmation_after(2), heard_not("spin", "loot"),
              no_web_lookup],
             between=1.0, web_limit=0,
             note="The owner's worry (2026-10-15): the bot asks Darren something, Sam talks "
                  "~10 s straight away, and Darren answers ~1 s after he stops. The old fixed "
                  "10 s window (from the end of the bot's speech) had closed by then; the new "
                  "one pauses while others talk and gives 3 s grace after they stop. Needs "
                  "SPEAKING_DETECTION_ENABLED (on in the live .env)."),
    Scenario("irl_others_talk_after_answer",
             [Speech(DARREN, [TIMES], voice="dennis"),
              Speech(SAM, SAM_SHORT, after_reply=0.3, voice="theodore"),
              Speech(DARREN, ["And what's that divided by four?"], after_prev=1.0,
                     voice="dennis")],
             [says_any("96", "ninety-six", "ninety six"), old_window_closed(2, 4.0),
              turns_exactly(2), replied_within(2, 6.0),
              says_any("24", "twenty-four", "twenty four", turn=1), heard_not("tree"),
              no_web_lookup],
             between=1.0, web_limit=0,
             note="Same worry after a plain answer (old window: 4 s): Sam talks ~5 s, then "
                  "Darren asks a real follow-up 1 s later. It must still be answered."),
    Scenario("irl_talk_during_answer",
             [Speech(DARREN, [SOURDOUGH_Q], voice="dennis"),
              Speech(DARREN, ["Wait, how often do I have to feed it?"], during_answer=1.5,
                     voice="dennis")],
             [talked_over_bot(1), turns_exactly(2), heard_includes("feed", turn=1),
              says_any("day", "daily", "hours", "twice", "once", turn=1), no_web_lookup],
             between=1.0, web_limit=0,
             note="Darren starts a follow-up while the bot is still answering (happens live "
                  "when answers run long). It must be kept and answered after the bot "
                  "finishes, not dropped. talked_over_bot is a setup check: if the answer "
                  "ended within 1.5 s, it was just an ordinary follow-up."),
]
SCENARIOS += IRL_SCENARIOS


# ---------------------------------------------------------------- recorded turns


def recording_scenarios(folder: Path) -> list[Scenario]:
    """One rec_* scenario per question the live bot saved (folder/*.wav + .json sidecar).

    The clip plays as recorded, as its speaker. If the transcript has no wake word (a
    follow-up, or it was clipped), a synthesized "Hey Jarvis." goes first so a fresh
    session listens. Optional sidecar "label" keys:
      expect: "reply" (the bot should answer) or "quiet" (it shouldn't: false triggers;
              no wake word is added)
      heard: the correct transcript (default: the sidecar's "heard")
      activity: the speaker's game, e.g. "Enshrouded"
      web_limit: web lookups allowed (default 0, to keep the live search budget)
      note: shown in the report
    """
    scenarios = []
    for wav in sorted(folder.glob("*.wav")):
        meta = read_sidecar(wav)
        label = meta.get("label") if isinstance(meta.get("label"), dict) else {}
        speaker_id = int(meta.get("speaker_id") or 700000000000000099)
        user = FakeMember(speaker_id, str(meta.get("speaker_name") or "Speaker"),
                          activity=label.get("activity") or meta.get("activity"))
        expect = label.get("expect")
        parts: list = [Recording(wav)]
        if expect != "quiet" and "jarvis" not in str(meta.get("raw_heard", "")).lower():
            parts = ["Hey Jarvis.", 0.4] + parts
        reference = label.get("heard") or meta.get("heard")
        checks: list[Check] = [heard(reference)] if reference and expect != "quiet" else []
        if expect == "reply":
            checks.append(turns_at_least(1))
        elif expect == "quiet":
            checks.append(no_llm_turn)
        web_limit = int(label.get("web_limit", 0))
        if web_limit == 0:
            checks.append(no_web_lookup)
        scenarios.append(Scenario(
            f"rec_{wav.stem}", [Speech(user, parts)], checks, answers=expect != "quiet",
            web_limit=web_limit,
            note=(label.get("note") or f"Recorded {meta.get('time', wav.stem)}, "
                  f"{user.display_name}, live outcome {meta.get('outcome', '?')}: "
                  f"heard {meta.get('raw_heard', '?')!r}, replied {meta.get('reply', '')!r}")))
    return scenarios


# ---------------------------------------------------------------- wake word sweep


async def wake_sweep(rig, threshold: float, wake_dir: Path | None = None) -> list[dict]:
    """Offline: max wake score for 'Hey Jarvis' in many voices (and some negatives), plus
    the live bot's saved wake clips in wake_dir (recordings/wake/*.wav + .json)."""
    import numpy as np
    from openwakeword.model import Model
    from scipy.signal import resample_poly

    from scripts.e2e.harness import load_recording

    model = Model(wakeword_models=[rig.settings.wake_word_model], inference_framework="onnx")

    def max_score(audio24) -> float:
        audio16 = resample_poly(audio24, 2, 3)
        pcm = (np.clip(np.concatenate([np.zeros(8000), audio16, np.zeros(16000)]), -1, 1)
               * 32767).astype(np.int16)
        model.reset()
        best = 0.0
        for i in range(0, len(pcm) - 1280, 1280):
            scores = model.predict(pcm[i:i + 1280])
            best = max(best, max(scores.values()))
        return best

    clips = [("Hey Jarvis, what time is it?", v) for v in
             ("ashley", "dennis", "olivia", "carter", "priya", "theodore", "hades", "pixie")]
    clips += [(t, "ashley") for t in ("Hey Travis, what time is it?", "Hey, are you there?",
                                     "Jarvis is a cool name.", "Hey Casey, what's up?")]
    rows = []
    for text, voice in clips:
        best = max_score(await rig.speech.say(text, voice))
        positive = "jarvis" in text.lower() and text.lower().startswith("hey jarvis")
        rows.append({"text": text, "voice": voice, "score": round(best, 3),
                     "positive": positive, "detected": best >= threshold})

    # Expected: the sidecar label's "wake" (true/false) if someone labeled it, else
    # whatever the live bot decided ("hit")
    for wav in sorted(wake_dir.glob("*.wav")) if wake_dir and wake_dir.is_dir() else []:
        meta = read_sidecar(wav)
        label = meta.get("label") if isinstance(meta.get("label"), dict) else {}
        best = max_score(load_recording(Recording(wav)))
        rows.append({"text": f"{wav.name} (live score {meta.get('score', '?')})",
                     "voice": "recorded", "score": round(best, 3),
                     "positive": bool(label.get("wake", meta.get("hit"))),
                     "detected": best >= threshold})
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


def timing_summary(results: list[RunResult]) -> list[str]:
    """Median and p90 of first audio and answer audio over the runs that had them."""
    lines = []
    for key, label in (("first_audio_s", "First audio"), ("answer_audio_s", "Answer audio")):
        values = sorted(r.metrics[key] for r in results if r.metrics.get(key) is not None)
        if not values:
            continue
        n = len(values)
        median = (values[(n - 1) // 2] + values[n // 2]) / 2
        p90 = values[min(n - 1, -(-9 * n // 10) - 1)]  # nearest rank
        lines.append(f"{label}: median {median:.2f} s, p90 {p90:.2f} s ({n} runs)  ")
    return lines


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
    lines += ["", "Times are measured from the last packet of the user's speech, so they include "
              "the MIN_SILENCE_MS pause that ends a question.", ""]
    lines += timing_summary(results) + [""]

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
        if r.monitor_posts:
            lines.append("- Live monitor would post (not sent):")
            for post in r.monitor_posts:
                first, *rest = post.splitlines()
                lines += [f"  - {first}"] + [f"    {ln}" for ln in rest]
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def monitor_posts(turns: list, settings) -> list[str]:
    """What the live monitor (src/monitor) would post for these turns. Nothing is sent."""
    from src.monitor import Redactor, Thresholds, detect, format_post

    thresholds, redact = Thresholds.from_settings(settings), Redactor(settings)
    posts = []
    for turn in turns:
        findings = detect(turn, thresholds)
        if findings:
            posts.append(format_post(turn, findings, redact))
    return posts


def to_json(r: RunResult) -> dict:
    return {
        "name": r.name, "error": r.error, "duration": round(r.duration, 2),
        "metrics": r.metrics,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in r.checks],
        "turns": [t.__dict__ for t in r.turns], "posts": r.posts,
        "segments": [{"start": round(s.start, 2), "duration": round(s.duration, 2),
                      "interrupted": s.interrupted, "transcript": s.transcript} for s in r.segments],
        "monitor_posts": r.monitor_posts,
    }


# ---------------------------------------------------------------- main


def select_scenarios(args: list[str]) -> tuple[list[Scenario], Path | None]:
    """Scenarios picked by the command line, and the recordings folder if one was given."""
    filters = [a for a in args if not a.startswith("--")]
    no_web = "--no-web" in args
    rec_arg = next((a for a in args if a == "--recordings" or a.startswith("--recordings=")),
                   None)
    rec_dir = None
    if rec_arg is not None:
        rec_dir = Path(rec_arg.split("=", 1)[1]) if "=" in rec_arg else ROOT / "recordings"
    pool = list(SCENARIOS)
    if rec_dir is not None:
        pool += recording_scenarios(rec_dir / "turns")
    if "--irl" in args:
        pool = [s for s in pool if s.name.startswith("irl_")]
    selected = [s for s in pool
                if (not filters or any(f in s.name for f in filters))
                and not (no_web and s.name.startswith("web_"))]
    return selected, rec_dir


def list_scenarios(selected: list[Scenario]) -> None:
    for sc in selected:
        web = "" if sc.web_limit is None else f" [web<={sc.web_limit}]"
        print(f"{sc.name}{web}")
        for sp in sc.speeches:
            when = (f"+{sp.after_reply}s after reply" if sp.after_reply is not None
                    else f"+{sp.during_answer}s into answer" if sp.during_answer is not None
                    else f"+{sp.after_prev}s after previous" if sp.after_prev is not None
                    else f"at {sp.at}s")
            print(f"    {sp.user.display_name} ({sp.voice}, {when}): {sp.text}")
        print(f"    checks: {', '.join(c.__name__ for c in sc.checks)}")
    print(f"{len(selected)} scenario(s)")


async def main(args: list[str]) -> int:
    from scripts.e2e.harness import Rig
    from src.config import get_settings

    selected, rec_dir = select_scenarios(args)
    if "--list" in args:
        list_scenarios(selected)
        return 0

    settings = get_settings()
    out = ROOT / "reports" / f"e2e-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler()],
    )
    for noisy in ("httpx", "httpcore", "openai", "primp", "ddgs", "discord"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    filters = [a for a in args if not a.startswith("--")]
    repeat = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--repeat=")), 1)

    # --monitor: feed each turn through the live monitor's recorder and detectors
    recorder, live_turns = None, []
    if "--monitor" in args:
        from src.monitor import TurnRecorder

        recorder = TurnRecorder(live_turns.append)
        recorder.install()

    rig = Rig(settings, out)
    if "--listen" in args:
        from scripts.e2e.discord_mirror import DiscordMirror

        rig.mirror = await DiscordMirror.join(settings)
    results = []
    scenarios = {}
    runs = [(sc, rep) for rep in range(repeat) for sc in selected]
    for sc, rep in runs:
        run_name = sc.name if repeat == 1 else f"{sc.name}-r{rep + 1}"
        scenarios[run_name] = sc
        print(f"\n===== {run_name} =====", flush=True)
        members = MEMBERS + list({sp.user.id: sp.user for sp in sc.speeches
                                  if all(sp.user.id != m.id for m in MEMBERS)}.values())
        r = await rig.run(run_name, sc.speeches, members, between=sc.between, observer=recorder,
                          web_limit=sc.web_limit)
        if recorder:
            r.monitor_posts = monitor_posts(live_turns, settings)
            live_turns.clear()
        checks =list(sc.checks) + ([no_error, tts_fidelity, no_tags_or_urls_spoken] if sc.answers else [])
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
    if not args or "sweep" in filters:
        sweep = await wake_sweep(rig, settings.wake_word_confidence,
                                 rec_dir / "wake" if rec_dir else None)
    if rig.tts is not None:
        await rig.tts.close()
    if rig.mirror is not None:
        await rig.mirror.close()

    render_report(results, scenarios, sweep, settings.wake_word_confidence, out)
    (out / "results.json").write_text(
        json.dumps({"scenarios": [to_json(r) for r in results], "wake_sweep": sweep}, indent=2,
                   default=float),
        encoding="utf-8")
    print(f"\nReport: {out / 'report.md'}")
    failed = [r.name for r in results
              if not all(c[1] for c in r.checks) and not scenarios[r.name].expect_fail]
    print(f"Failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
