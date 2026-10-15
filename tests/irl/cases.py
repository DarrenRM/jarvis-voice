"""Real interactions from the bot's logs, replayed by the tests in this directory.

Times are seconds on the incident's own clock (10:26:35 -> 35.0 in a 10:26 incident).
Values the log didn't record are marked "assumed" or "paraphrased".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    name: str
    id: int


SAM = Speaker("Sam", 100000000000000002)
DARREN = Speaker("Darren", 100000000000000001)


@dataclass(frozen=True)
class Speech:
    start: float
    end: float
    text: str = ""

    @property
    def seconds(self) -> float:
        return self.end - self.start


# --- 1. Sam cut off (2026-10-15 10:26) ---------------------------------------------


@dataclass(frozen=True)
class CarryOnCase:
    when: str
    speaker: Speaker
    wake: Speech  # in the pre-buffer when the wake word fired
    ack: Speech  # the bot's ack, before he went on
    question: Speech  # what ended the turn (Whisper heard all of the first turn as `heard`)
    heard: str
    carried_on: Speech  # thrown away in real life
    old_reply: str
    answer_after: float  # assumed: turn end -> the answer starts playing


SAM_CUT_OFF = CarryOnCase(
    when="2026-10-15 10:26",
    speaker=SAM,
    wake=Speech(26.0, 26.78, "Hey Jarvis"),
    ack=Speech(27.0, 28.0, "Sup?"),
    question=Speech(30.0, 32.0, "Sup."),
    heard="Hey Jarvis. Sup.",
    carried_on=Speech(35.0, 44.0, "(a question about Duluth, Minnesota)"),
    old_reply="Sup yourself...",
    answer_after=4.0,
)


# --- 2. Follow-ups without the wake word and the Jev addressee check --------------------


@dataclass(frozen=True)
class FollowUpCase:
    when: str
    speaker: Speaker | None  # None: not recorded
    bot_said: str
    said: str
    p_bot: float
    wants_reply: float
    should_reply: bool
    old_asks: float | None = None  # the old "question or request?" score, when logged
    confidence: float = 0.97  # assumed (real addressed ones came with 0.96+)


# Darren, 2026-10-15 12:10-12:12. Q1 had the wake word; every follow-up started within
# the 4 s window after the answer. The bot's lines are paraphrased (not in the log
# excerpt); wants_reply values are representative of the 14-case eval (0.93+ on those
# needing a reply, 0.07 or less on acknowledgements); the correction's p(bot) and old
# asks score are from the log.
DARREN_Q1 = "Hey Jarvis, what does the power attack do now in 1.0?"
DARREN_FOLLOWUPS = [
    FollowUpCase(
        "2026-10-15 12:10", DARREN,
        "In 1.0 the power attack staggers enemies and breaks their guard.",
        "Do they count as overpowered when you've used that on them?",
        p_bot=0.98, wants_reply=0.95, should_reply=True,
    ),
    FollowUpCase(
        "2026-10-15 12:11", DARREN,
        "Not that I know of. Staggered enemies just can't block for a moment.",
        "Can you look that up on the web? I'm pretty sure they're like in some "
        "overpowered or stunned state.",
        p_bot=0.99, wants_reply=0.97, should_reply=True,
    ),
    FollowUpCase(
        "2026-10-15 12:11", DARREN,
        "The patch notes say a power attack leaves enemies staggered, open to more hits.",
        "Then what is the thing I see above their head?",
        p_bot=0.99, wants_reply=0.96, should_reply=True,
    ),
    # WRONGLY ignored in real life: a correction, but "asks" scored it 0.04
    FollowUpCase(
        "2026-10-15 12:12", DARREN,
        "That's probably the health bar of the enemy you're fighting.",
        "No, it's a circle above their head that looks like they're stunned and they "
        "don't attack at all.",
        p_bot=0.99, wants_reply=0.94, should_reply=True, old_asks=0.04,
    ),
]

# Acknowledgements: must stay unanswered
ACKNOWLEDGEMENTS = [
    FollowUpCase(
        "2026-10-15 10:25", DARREN,
        "You're playing Enshrouded, so the wiki's crafting page is your best bet.",
        "Well, I guess that works.",
        p_bot=0.93, wants_reply=0.05, should_reply=False,
    ),
    FollowUpCase(
        "2026-10-15 10:26:52", SAM,
        SAM_CUT_OFF.old_reply,
        "Okay.",
        p_bot=0.90, wants_reply=0.04, should_reply=False,
    ),
    FollowUpCase(
        "2026-10-14 20:41", None,
        "Ten shroud wood and four linen at the workbench.",
        "Okay, cool, thanks.",
        p_bot=0.95, wants_reply=0.03, should_reply=False,
    ),
]


# --- 3. Follow-up window while other people talk ------------------------------------------
# Not a logged incident: the owner's concern, built on Darren's 12:12 correction. Times
# are seconds after the bot's answer ended.


@dataclass(frozen=True)
class WindowCase:
    asker: Speaker
    other: Speaker
    other_talks: Speech
    reply: Speech


POLITE_WAIT = WindowCase(
    asker=DARREN,
    other=SAM,
    other_talks=Speech(0.0, 6.0, "(Sam chatting right after the answer)"),
    reply=Speech(7.5, 12.5, DARREN_FOLLOWUPS[3].said),
)

# The asker starts replying over the end of the answer (the answer plays 0-3 s)
TALKS_OVER_ANSWER = WindowCase(
    asker=DARREN,
    other=SAM,
    other_talks=Speech(0.0, 0.0),
    reply=Speech(2.0, 6.0, DARREN_FOLLOWUPS[3].said),
)
ANSWER_PLAYS = Speech(0.0, 3.0, DARREN_FOLLOWUPS[3].bot_said)


# --- 4. Text the pipeline gets wrong -------------------------------------------------------

# (when, what Whisper heard, what the model must get)
TRANSCRIPTS = [
    ("2026-10-14 21:15", "Hey Jarvis, where do I get in shrouded oil?",
     "where do I get Enshrouded oil?"),
    ("2026-10-15 10:26", "Hey Jarvis. Sup.", "Sup."),
    ("date not recorded", "Hey Jarvis, what's 12 times 8?", "what's 12 times 8?"),
    ("2026-10-15 12:10", DARREN_Q1, "what does the power attack do now in 1.0?"),
]

# 2026-10-14: the bot read "1.0" as "one". The reply is paraphrased.
VERSION_REPLY = ("Enshrouded 1.0 is the big launch update.",
                 "Enshrouded one point oh is the big launch update.")


# --- 5. Wake word near-miss log (2026-10-15 10:26) -----------------------------------------
# Sam's wake word scored 1.00 and fired, then the next gap logged "near miss 1.00".
# The scores before the hit are assumed (a usual build-up).
SAM_WAKE_SCORES = [0.12, 0.31, 1.00]
