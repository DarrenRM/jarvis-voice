"""Decide whether speech in the follow-up window was meant for the bot.

After the bot speaks, the person it answered can carry on without the wake word for a
moment. But they may just as well turn to someone else in the channel, so each
follow-up is checked before the bot replies:

- The bot asked them something: a reply that fits ("the regular one") is for the bot.
- The bot answered them: only something that calls for a reply counts: a new question
  or request, or telling the bot it got something wrong ("no, the circle above their
  head"). Acknowledgements ("okay", "I guess that works") and chatting about the answer
  with friends ("nice, told you") must not get a reply.

Speaker isolation (only the person the bot answered is heard) is the session's job.
Anything uncertain (errors, timeouts, unclear answers) counts as not addressed: staying
quiet beats butting into a conversation.

Two interchangeable judges: TypeSafe's Jev (calibrated probabilities, ~0.1s) and Qwen
on Cerebras (YES/NO, ~0.2-0.5s).
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx

logger = logging.getLogger(__name__)

# Dead air while this runs, so keep it short
JUDGE_TIMEOUT_SECONDS = 2.0


@dataclass
class FollowUp:
    """What the bot said last, which the person may be responding to."""

    bot_said: str
    bot_asked: bool  # the bot's reply ended in a question


@dataclass
class Verdict:
    addressed: bool
    confidence: float  # 0-1; models without calibrated output report 1.0 or 0.0
    reason: str = ""


class AddresseeJudge(Protocol):
    async def judge(self, followup: FollowUp, utterance: str, speaker: str) -> Verdict: ...


QWEN_PROMPT = """You watch a Discord voice channel with several people and a voice
assistant named {assistant}.
{assistant} just {verb} {speaker}:
"{bot_said}"

Right after that, {speaker} said:
"{utterance}"

{rule}
Speech-to-text may mishear words.
Answer with exactly one word: YES or NO."""

QWEN_RULE_ASKED = """Was {speaker} talking to {assistant} (answering its question, or asking it
something else), rather than to someone else in the channel? A short reply that fits the
question ("the first one", "yeah", "iron") is for {assistant}."""

QWEN_RULE_ANSWERED = """Does what {speaker} said call for {assistant} to respond:
a new question, a request, or telling {assistant} it got something wrong or misunderstood?
Answer NO if it was just an acknowledgement or reaction ("ok", "thanks", "nice", "I guess
that works") or they were talking to someone else in the channel."""


class QwenAddresseeJudge:
    """Asks Qwen on Cerebras (reasoning off) for a YES/NO."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-3.8-27b",
        base_url: str = "https://api.cerebras.ai/v1",
        assistant_name: str = "Jarvis",
        client=None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._assistant = assistant_name
        self._client = client

    def _ensure_client(self) -> None:
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url,
                timeout=JUDGE_TIMEOUT_SECONDS, max_retries=0,
            )

    async def judge(self, followup: FollowUp, utterance: str, speaker: str) -> Verdict:
        self._ensure_client()
        speaker = speaker or "the user"
        rule = QWEN_RULE_ASKED if followup.bot_asked else QWEN_RULE_ANSWERED
        prompt = QWEN_PROMPT.format(
            assistant=self._assistant, speaker=speaker,
            verb="asked" if followup.bot_asked else "answered",
            bot_said=followup.bot_said, utterance=utterance,
            rule=rule.format(assistant=self._assistant, speaker=speaker),
        )
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=3,
                    temperature=0,
                    reasoning_effort="none",
                ),
                timeout=JUDGE_TIMEOUT_SECONDS,
            )
            answer = (response.choices[0].message.content or "").strip().upper()
        except Exception as e:
            return Verdict(False, 0.0, f"check failed: {type(e).__name__}")
        if answer.startswith("YES"):
            return Verdict(True, 1.0, "qwen: yes")
        if answer.startswith("NO"):
            return Verdict(False, 1.0, "qwen: no")
        return Verdict(False, 0.0, f"unclear answer {answer[:20]!r}")


class JevAddresseeJudge:
    """Asks TypeSafe's Jev two questions in one call (evaluated in parallel):

    - addressee (Choice): was it said to the bot or to someone else?
    - wants_reply (Noul): does it call for the bot to respond (a question, a request, or
      a correction), rather than just acknowledge the answer?

    Addressed when Jev puts at least `min_probability` on the bot with at least
    `min_confidence`; after an answer (not a question) `wants_reply` must also reach
    `min_probability`. On hand-labelled cases its one false "yes" came with confidence
    0.44 and the real ones with 0.96+. `wants_reply` replaced "is it a question or a
    request", which missed corrections: on 14 follow-ups it scored 0.93+ on the ones
    that needed a reply and 0.07 or less on acknowledgements (the old one missed all
    four corrections). Docs: https://docs.typesafe.ai
    """

    URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        assistant_name: str = "Jarvis",
        min_probability: float = 0.6,
        min_confidence: float = 0.6,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._assistant = assistant_name
        self._min_probability = min_probability
        self._min_confidence = min_confidence
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=JUDGE_TIMEOUT_SECONDS, trust_env=False, transport=self._transport,
            )
        return self._client

    async def judge(self, followup: FollowUp, utterance: str, speaker: str) -> Verdict:
        speaker = speaker or "the user"
        bot = self._assistant
        verb = "asked" if followup.bot_asked else "answered"
        state = (
            f"A Discord voice channel with several people and a voice assistant named "
            f"{bot}. Speech-to-text may mishear words.\n"
            f'{bot} {verb} {speaker}: "{followup.bot_said}"\n'
            f'Then {speaker} said: "{utterance}"'
        )
        body = {
            "model": self._model,
            "state": state,
            "questions": {
                "addressee": {
                    "type": "choice",
                    "instructions": f"Who was {speaker} talking to when they said that?",
                    "criteria": {
                        "assistant": f"{bot}: answering it, or asking it something",
                        "someone_else": "Another person in the channel, not the assistant",
                    },
                },
                "wants_reply": {
                    "type": "noul",
                    "instructions": (
                        f"What {speaker} said last calls for {bot} to respond: a question, "
                        f"a request, or telling {bot} it got something wrong or "
                        "misunderstood. Not just an acknowledgement, thanks, or a reaction "
                        "like 'ok', 'nice', 'I guess that works'"
                    ),
                },
            },
        }
        try:
            response = await self._http().post(self.URL, json=body)
            response.raise_for_status()
            answers = response.json()["answers"]
            p_bot = float(answers["addressee"]["probabilities"]["assistant"])
            confidence = float(answers["addressee"].get("confidence", 0.0))
            p_reply = float(answers["wants_reply"]["noul"])
        except Exception as e:
            reason = (f"HTTP {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError)
                      else type(e).__name__)  # never the request (it carries the key)
            return Verdict(False, 0.0, f"check failed: {reason}")
        addressed = p_bot >= self._min_probability and confidence >= self._min_confidence
        if not followup.bot_asked:
            addressed = addressed and p_reply >= self._min_probability
        return Verdict(addressed, confidence,
                       f"jev: p(bot)={p_bot:.2f} conf={confidence:.2f} p(reply)={p_reply:.2f}")

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


def is_question(reply: Optional[str]) -> bool:
    """True if the bot's reply asks something in its last three sentences: it often adds
    asides after the question ("...sandworm fear? Take your time. I'll be here.")."""
    if not reply:
        return False
    text = re.sub(r"\[[^\]]*\]", " ", reply)  # emotion/non-verbal tags
    text = re.sub(r"\.{2,}|…", ",", text)  # "I'll wait... eventually." is one sentence
    sentences = [s for s in re.findall(r"[^.?!]+[.?!]*", text) if s.strip(" \"')\n")]
    return any(s.rstrip(" \"')\n").endswith("?") for s in sentences[-3:])
