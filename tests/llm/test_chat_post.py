"""Tests for posting reference answers to chat (Jev gate, Qwen writer, pipeline hook).
No real network."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np

from src.llm.chat_post import (
    ANNOUNCEMENT,
    ChatPoster,
    JevPostJudge,
    PostMaterial,
    PostVerdict,
    QwenPostWriter,
    already_posted,
    format_post,
)
from src.pipeline.base import PipelineContext
from src.pipeline.response_stage import ResponseStage, ResponseTurn
from src.pipeline.transcription_stage import TranscriptionResult
from src.pipeline.voice_pipeline import VoicePipeline

KEY = "ts-test-key"
RECIPE = PostMaterial("what do I need for a ghost glider",
                      "Ten shroud wood, four linen and two ghost essence, at the workbench.",
                      "Ghost Glider: 10 Shroud Wood, 4 Linen, 2 Ghost Essence")


def turn(text="[happy] Ten shroud wood and four linen.", ok=True, tools=()):
    messages = []
    for i, name in enumerate(tools):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "function": {"name": name, "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"{name} result"})
    messages.append({"role": "assistant", "content": text})
    return ResponseTurn(text=text, ok=ok, messages=messages)


def jev(kind="recipe", p_none=0.05, confidence=0.9, keep=0.9, status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
        if status != 200:
            return httpx.Response(status, text="nope")
        probabilities = {"recipe": 0, "steps": 0, "stats": 0, "reference": 0, "none": p_none}
        if kind != "none":
            probabilities[kind] = 1 - p_none
        return httpx.Response(200, json={"answers": {
            "kind": {"type": "choice", "choice": kind, "confidence": confidence,
                     "probabilities": probabilities},
            "keep": {"type": "noul", "noul": keep},
        }})
    return JevPostJudge(KEY, transport=httpx.MockTransport(handler))


class TestJevGate:
    async def test_recipe_worth_keeping_is_posted(self):
        seen = {}
        verdict = await jev(seen=seen).judge(RECIPE)
        assert verdict.post and verdict.kind == "recipe"
        assert seen["auth"] == f"Bearer {KEY}"
        assert set(seen["body"]["questions"]) == {"kind", "keep"}
        assert "Ghost Glider" in seen["body"]["state"]

    async def test_small_talk_is_not_posted(self):
        assert not (await jev(kind="none", p_none=0.95).judge(RECIPE)).post

    async def test_not_worth_keeping_is_not_posted(self):
        assert not (await jev(keep=0.3).judge(RECIPE)).post

    async def test_unsure_is_not_posted(self):
        assert not (await jev(p_none=0.4).judge(RECIPE)).post

    async def test_unsure_which_kind_still_posts(self):
        # A recipe that's also a list: sure it's worth posting, unsure which kind
        assert (await jev(confidence=0.51, keep=0.79).judge(RECIPE)).post

    async def test_http_error_is_not_posted_and_hides_key(self):
        verdict = await jev(status=500).judge(RECIPE)
        assert not verdict.post
        assert "500" in verdict.reason and KEY not in verdict.reason


class FakeQwen:
    def __init__(self, content=None, error=None):
        self.prompts = []
        self._content, self._error = content, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestWriter:
    async def test_table(self):
        content = json.dumps({"title": "Ghost Glider", "table": {
            "headers": ["Material", "Qty"], "rows": [["Shroud Wood", "10"], ["Linen", "4"]]},
            "bullets": None})
        fake = FakeQwen(content)
        post = await QwenPostWriter("k", client=fake).write(RECIPE)
        assert post.startswith("**Ghost Glider**\n```")
        assert "Shroud Wood" in post and "Qty" in post
        assert "not instructions" in fake.prompts[0]
        assert "Ghost Glider: 10 Shroud Wood" in fake.prompts[0]

    async def test_bullets_with_text_around_the_json(self):
        content = 'Here you go: {"title": "Steps", "table": null, "bullets": ["Chop", "Fry"]}'
        post = await QwenPostWriter("k", client=FakeQwen(content)).write(RECIPE)
        assert post == "**Steps**\n- Chop\n- Fry"

    async def test_garbage_or_error_gives_no_post(self):
        for fake in (FakeQwen("no json here"), FakeQwen(error=RuntimeError("down")),
                     FakeQwen('{"title": "x", "table": null, "bullets": []}')):
            assert await QwenPostWriter("k", client=fake).write(RECIPE) is None

    def test_too_long_gives_no_post(self):
        rows = [["x" * 25, "y" * 25]] * 15
        assert format_post({"title": "t", "table": {"headers": ["a", "b"], "rows": rows}}) \
            is not None
        bullets = ["z" * 200] * 15
        assert format_post({"title": "t", "bullets": bullets}) is None

    def test_table_with_notes(self):
        post = format_post({"title": "Ghost Glider", "table": {
            "headers": ["Material", "Qty"], "rows": [["Shroud Wood", "3"]]},
            "bullets": ["Unlock: gold chest atop the Sun Temple", "Craft at the Carpenter"]})
        assert post.endswith("```\n- Unlock: gold chest atop the Sun Temple\n"
                             "- Craft at the Carpenter")

    def test_notes_dropped_before_the_table_when_too_long(self):
        rows = [["x" * 25, "y" * 25]] * 15
        post = format_post({"title": "t", "table": {"headers": ["a", "b"], "rows": rows},
                            "bullets": ["n" * 200] * 5})
        assert post is not None and post.endswith("```")


class StubJudge:
    def __init__(self, post=True):
        self.calls = []
        self._post = post

    async def judge(self, material):
        self.calls.append(material)
        return PostVerdict(self._post, "recipe" if self._post else "none", "stub")

    async def close(self):
        pass


class StubWriter:
    def __init__(self, text="**Post**"):
        self.text = text

    async def write(self, material):
        return self.text


def poster(judge=None, writer=None):
    posted = []

    async def post(text):
        posted.append(text)
        return "Done!"

    speak = AsyncMock(return_value=np.ones(100, dtype=np.float32))
    p = ChatPoster(judge or StubJudge(), writer or StubWriter(), post=post, speak=speak)
    return p, posted, speak


class TestChatPoster:
    async def test_posts_reference_answer(self):
        p, posted, _ = poster()
        material = await p.check("ghost glider?", turn(tools=["web_search"]))
        assert material is not None
        assert material.answer == "Ten shroud wood and four linen."  # tags removed
        assert material.found == "web_search result"
        p.write_and_post(material)
        await p.wait()
        assert posted == ["**Post**"]

    async def test_skips_when_model_already_posted(self):
        judge = StubJudge()
        p, _, _ = poster(judge=judge)
        assert await p.check("q", turn(tools=["post_table"])) is None
        assert judge.calls == []

    async def test_skips_failed_turns(self):
        p, _, _ = poster()
        assert await p.check("q", turn(ok=False)) is None
        assert await p.check("q", None) is None

    def test_already_posted(self):
        assert already_posted(turn(tools=["post_to_chat"]))
        assert not already_posted(turn(tools=["web_search"]))

    async def test_nothing_looked_up_is_not_posted(self):
        # Answers from the model's memory (a pancake recipe) don't get posted
        judge = StubJudge()
        p, _, _ = poster(judge=judge)
        assert await p.check("pancakes?", turn()) is None
        assert await p.check("tag riley", turn(tools=["tag_user"])) is None
        assert judge.calls == []

    async def test_follow_up_uses_earlier_lookup(self):
        p, _, _ = poster()
        material = await p.check("and the regular glider?", turn(), earlier_found="wiki page")
        assert material is not None and material.found == "wiki page"

    def test_earlier_lookups_from_history(self):
        stage = ResponseStage(MagicMock(), lambda: "")
        stage._history = [
            {"role": "user", "content": "glider?"},
            *turn(tools=["web_search", "post_table"]).messages,
            {"role": "user", "content": "and the regular one?"},
            *turn(tools=["web_extract"]).messages,
        ]
        assert stage.earlier_lookups() == "web_search result"
        stage._history = stage._history[-3:]  # only the latest turn
        assert stage.earlier_lookups() == ""


def make_pipeline():
    transcription = MagicMock()
    transcription.process = AsyncMock(return_value=TranscriptionResult(
        text="ghost glider recipe", raw_text="hey jarvis ghost glider recipe"))
    transcription.cancel = AsyncMock()
    response = MagicMock()

    async def stream(text, context):
        response.last_turn = turn(tools=["web_search"])
        yield "Ten shroud wood."

    response.process_stream = stream
    tts = MagicMock()

    async def process_stream(response_stream, context, callback):
        async for _ in response_stream:
            await callback(np.zeros(10, dtype=np.float32))

    tts.process_stream = process_stream
    tts.cancel = AsyncMock()
    return VoicePipeline(transcription, response, tts)


class TestPipelineHook:
    async def run(self, pipeline):
        played = []

        async def play(audio):
            played.append(len(audio))

        result = await pipeline.process(
            audio=np.zeros(1000, dtype=np.float32), context=PipelineContext(speaker_name="A"),
            play_audio=play, wait_for_silence=AsyncMock(return_value=True))
        return result, played

    async def test_announces_after_answer_and_posts(self):
        pipeline = make_pipeline()
        pipeline.poster, posted, speak = poster()
        result, played = await self.run(pipeline)
        await pipeline.poster.wait()
        assert result is None
        assert played == [10, 100]  # the answer, then "I'll post this in chat too!"
        speak.assert_awaited_once_with(ANNOUNCEMENT)
        assert posted == ["**Post**"]

    async def test_no_announcement_when_not_worth_posting(self):
        pipeline = make_pipeline()
        pipeline.poster, posted, speak = poster(judge=StubJudge(post=False))
        _, played = await self.run(pipeline)
        assert played == [10]
        speak.assert_not_awaited()
        assert posted == []

    async def test_checker_failure_does_not_break_the_turn(self):
        pipeline = make_pipeline()
        broken = MagicMock()
        broken.check = AsyncMock(side_effect=RuntimeError("boom"))
        pipeline.poster = broken
        result, played = await self.run(pipeline)
        assert result is None and played == [10]
