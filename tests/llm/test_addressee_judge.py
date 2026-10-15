"""Tests for the follow-up addressee check (Jev and Qwen judges). No real network."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from src.llm.addressee_judge import (
    FollowUp,
    JevAddresseeJudge,
    QwenAddresseeJudge,
    is_question,
)

KEY = "ts-test-key"
ASKED = FollowUp("Do you mean the regular glider or the ghost glider?", bot_asked=True)
ANSWERED = FollowUp("The glider needs eight shroud wood.", bot_asked=False)


class TestIsQuestion:
    @pytest.mark.parametrize("reply", [
        "Which glider, regular or ghost?",
        "Oh, you're really warming up on me, huh? [laughing]",
        'He said "what?"',
        "Want me to post it? ",
        "[happy] What's the fastest land mammal on earth?\nTake your best shot.",
        "What creature do Fremen fear? Take your time. I'll be here. [breathe]",
        # e2e 2026-10-15: the ellipsis used to count as two more sentences
        "[laughing] Easy, I promise. How many continents does Earth have?\n\n"
        "Take your time. I'll wait... eventually.",
    ])
    def test_questions(self, reply):
        assert is_question(reply)

    @pytest.mark.parametrize("reply", [
        None, "", "It's four.", "Is it hard? Not really. Twelve times eight. It's ninety-six.", "[sigh] Fine.",
    ])
    def test_not_questions(self, reply):
        assert not is_question(reply)


def jev(p_bot=1.0, confidence=1.0, p_reply=0.9, status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
        if status != 200:
            return httpx.Response(status, text="nope")
        return httpx.Response(200, json={"model": "jev-1", "answers": {
            "addressee": {"type": "choice", "choice": "assistant", "confidence": confidence,
                          "probabilities": {"assistant": p_bot, "someone_else": 1 - p_bot}},
            "wants_reply": {"type": "noul", "noul": p_reply},
        }})
    return JevAddresseeJudge(KEY, transport=httpx.MockTransport(handler))


class TestJev:
    async def test_reply_to_question_is_for_the_bot(self):
        seen = {}
        verdict = await jev(p_reply=0.02, seen=seen).judge(ASKED, "the regular one", "Darren")
        assert verdict.addressed
        assert seen["auth"] == f"Bearer {KEY}"
        assert "asked Darren" in seen["body"]["state"]
        assert set(seen["body"]["questions"]) == {"addressee", "wants_reply"}

    async def test_talking_to_someone_else(self):
        verdict = await jev(p_bot=0.05).judge(ASKED, "Riley grab the loot", "Darren")
        assert not verdict.addressed

    async def test_unsure_counts_as_not_addressed(self):
        verdict = await jev(p_bot=0.75, confidence=0.44).judge(ASKED, "hold on", "Darren")
        assert not verdict.addressed

    async def test_after_answer_needs_a_question_or_request(self):
        assert not (await jev(p_reply=0.02).judge(ANSWERED, "ok cool thanks", "Darren")).addressed
        verdict = await jev(p_reply=0.96).judge(ANSWERED, "where do I find shroud wood", "D")
        assert verdict.addressed

    async def test_after_question_a_plain_reply_is_enough(self):
        verdict = await jev(p_reply=0.02).judge(ASKED, "ghost", "Darren")
        assert verdict.addressed

    async def test_http_error_is_not_addressed_and_hides_key(self):
        verdict = await jev(status=401).judge(ASKED, "ghost", "Darren")
        assert not verdict.addressed
        assert "401" in verdict.reason and KEY not in verdict.reason


class FakeQwen:
    def __init__(self, answer=None, delay=0.0, error=None):
        self.prompts = []
        self._answer, self._delay, self._error = answer, delay, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        assert kwargs["reasoning_effort"] == "none"
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestQwen:
    async def test_yes_and_no(self):
        assert (await QwenAddresseeJudge("k", client=FakeQwen("YES")).judge(
            ASKED, "ghost", "Darren")).addressed
        assert not (await QwenAddresseeJudge("k", client=FakeQwen("No.")).judge(
            ASKED, "Riley grab the loot", "Darren")).addressed

    async def test_prompt_matches_what_the_bot_did(self):
        fake = FakeQwen("NO")
        judge = QwenAddresseeJudge("k", client=fake)
        await judge.judge(ASKED, "ghost", "Darren")
        await judge.judge(ANSWERED, "thanks", "Darren")
        assert "just asked Darren" in fake.prompts[0]
        assert "just answered Darren" in fake.prompts[1] and "new question" in fake.prompts[1]
        assert "got something wrong" in fake.prompts[1]

    async def test_failure_or_garbage_is_not_addressed(self):
        for fake in (FakeQwen(error=RuntimeError("down")), FakeQwen("Maybe")):
            verdict = await QwenAddresseeJudge("k", client=fake).judge(ASKED, "ghost", "D")
            assert not verdict.addressed

    async def test_timeout_is_not_addressed(self, monkeypatch):
        monkeypatch.setattr("src.llm.addressee_judge.JUDGE_TIMEOUT_SECONDS", 0.05)
        judge = QwenAddresseeJudge("k", client=FakeQwen("YES", delay=1.0))
        assert not (await judge.judge(ASKED, "ghost", "Darren")).addressed
