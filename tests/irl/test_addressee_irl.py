"""Real follow-ups (no wake word) through the session's addressee gate and the real Jev
judge, with Jev's HTTP answers faked from the logged scores."""

import json

import httpx
import pytest

from src.bot.voice_session import VoiceSession
from src.llm.addressee_judge import FollowUp, JevAddresseeJudge, is_question
from src.pipeline.transcription_stage import TranscriptionResult

from .cases import ACKNOWLEDGEMENTS, DARREN_FOLLOWUPS, FollowUpCase


def gate(case: FollowUpCase, sent: list):
    """VoiceSession._is_addressed with a Jev judge that answers with the case's scores."""

    def handler(request):
        sent.append(json.loads(request.content))
        answers = {
            "addressee": {"type": "choice", "choice": "assistant",
                          "confidence": case.confidence,
                          "probabilities": {"assistant": case.p_bot,
                                            "someone_else": 1 - case.p_bot}},
            "wants_reply": {"type": "noul", "noul": case.wants_reply},
        }
        if case.old_asks is not None:  # an old-style answer must not matter
            answers["asks"] = {"type": "noul", "noul": case.old_asks}
        return httpx.Response(200, json={"model": "jev-1", "answers": answers})

    session = VoiceSession.__new__(VoiceSession)
    session._addressee_judge = JevAddresseeJudge(
        "ts-test-key", transport=httpx.MockTransport(handler))
    session._current_speaker_name = case.speaker.name if case.speaker else None
    return session


async def ask(case: FollowUpCase) -> tuple[bool, dict]:
    sent: list = []
    followup = FollowUp(case.bot_said, bot_asked=is_question(case.bot_said))
    assert not followup.bot_asked  # all after plain answers: the stricter check applies
    addressed = await gate(case, sent)._is_addressed(
        followup, TranscriptionResult(text=case.said, raw_text=case.said))
    return addressed, sent[0]


@pytest.mark.parametrize("case", DARREN_FOLLOWUPS, ids=lambda c: c.said[:40])
async def test_darrens_follow_ups_get_replies(case):
    """2026-10-15 12:10-12:12, Darren (100000000000000001), after "Hey Jarvis, what does
    the power attack do now in 1.0?": four follow-ups without the wake word, each inside
    the 4 s window. The first three got replies; the last, "No, it's a circle above their
    head...", was ignored: Jev put p(bot)=0.99 on it but the old "question or request?"
    check scored the correction 0.04. Every one of them needs a reply."""
    addressed, body = await ask(case)
    assert addressed == case.should_reply
    assert case.said in body["state"] and case.bot_said in body["state"]


@pytest.mark.parametrize("case", ACKNOWLEDGEMENTS, ids=lambda c: c.said)
async def test_acknowledgements_stay_unanswered(case):
    """"Well, I guess that works." (Darren, 2026-10-15 10:25, after "You're playing
    Enshrouded..."), "Okay." (Sam, 10:26:52) and "Okay, cool, thanks." (2026-10-14
    20:41) were said to the bot but need no reply: talking back would be noise."""
    addressed, _ = await ask(case)
    assert not addressed


async def test_the_question_asked_covers_corrections():
    """The 12:12 miss came from the question Jev was asked, so check its wording: it
    asks whether the bot should respond (corrections included), not "is it a question
    or a request"."""
    _, body = await ask(DARREN_FOLLOWUPS[3])
    assert set(body["questions"]) == {"addressee", "wants_reply"}
    wording = body["questions"]["wants_reply"]["instructions"]
    assert "got something wrong" in wording and "misunderstood" in wording
    assert "acknowledgement" in wording
