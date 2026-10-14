"""Tests for ChatOrchestrator.

Uses a scripted fake LLM that streams like the Cerebras client: text chunks
followed by exactly one TurnResult per call.
"""

import json
from typing import Optional

import pytest

from src.llm.chat_orchestrator import (
    DISCORD_ACTION_ACK_NOTE,
    ERROR_REPLY,
    GRACEFUL_FAILURE_REPLY,
    HOLD_PRE_TOOL_CHARS,
    ChatOrchestrator,
    ToolCall,
    TurnOutcome,
    TurnResult,
)
from src.tools import ToolDefinition, ToolExecutor, ToolHandler, ToolRegistry, ToolResult


class FakeLLM:
    """Plays back one scripted turn per stream_turn call and records requests."""

    def __init__(self, turns: list):
        self._turns = list(turns)
        self.requests: list[dict] = []

    async def stream_turn(self, messages: list[dict], tools: Optional[list[dict]] = None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        chunks, result = turn
        for chunk in chunks:
            yield chunk
        yield result


class EchoHandler(ToolHandler):
    """Tool that records calls and returns a fixed output."""

    def __init__(self, name: str = "echo", output: str = "echoed", fail: bool = False):
        self._name = name
        self._output = output
        self._fail = fail
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="test tool",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )

    async def execute(self, args):
        self.calls.append(args)
        if self._fail:
            raise RuntimeError("boom")
        return ToolResult(success=True, output=self._output)


def make_orchestrator(turns, handlers=(), max_turns=5):
    registry = ToolRegistry()
    for handler in handlers:
        registry.register(handler)
    llm = FakeLLM(turns)
    return ChatOrchestrator(llm, ToolExecutor(registry), max_turns=max_turns), llm


def base_messages():
    return [
        {"role": "system", "content": "You are Jarvis."},
        {"role": "user", "content": "[Alice]: hi"},
    ]


async def collect(orchestrator, messages, outcome=None):
    return [c async for c in orchestrator.chat_stream(messages, outcome)]


def tool_turn(*calls, content=""):
    return ([content] if content else [], TurnResult(content=content, tool_calls=list(calls),
                                                     finish_reason="tool_calls"))


def text_turn(*chunks):
    return (list(chunks), TurnResult(content="".join(chunks), finish_reason="stop"))


class TestPlainResponse:
    async def test_short_answer_released_when_turn_ends(self):
        orch, _ = make_orchestrator([text_turn("Hello ", "world!")])
        assert await collect(orch, base_messages()) == ["Hello world!"]

    async def test_long_answer_streams_without_waiting_for_turn_end(self):
        first = "a" * (HOLD_PRE_TOOL_CHARS + 1)
        orch, _ = make_orchestrator([text_turn("Hi. ", first, " more", " text.")])
        chunks = await collect(orch, base_messages())
        assert chunks == ["Hi. " + first, " more", " text."]

    async def test_single_llm_call_for_plain_answer(self):
        orch, llm = make_orchestrator([text_turn("Hi.")])
        await collect(orch, base_messages())
        assert len(llm.requests) == 1

    async def test_appends_final_assistant_message_and_marks_ok(self):
        orch, _ = make_orchestrator([text_turn("Hi ", "there.")])
        messages = base_messages()
        outcome = TurnOutcome()
        await collect(orch, messages, outcome)
        assert messages[-1] == {"role": "assistant", "content": "Hi there."}
        assert outcome.ok

    async def test_no_tools_sent_when_registry_empty(self):
        orch, llm = make_orchestrator([text_turn("Hi.")])
        await collect(orch, base_messages())
        assert llm.requests[0]["tools"] is None


class TestToolLoop:
    async def test_executes_tool_then_answers(self):
        handler = EchoHandler()
        orch, llm = make_orchestrator(
            [tool_turn(ToolCall("call_1", "echo", '{"text": "x"}')), text_turn("Done.")],
            handlers=[handler],
        )
        messages = base_messages()
        outcome = TurnOutcome()
        chunks = await collect(orch, messages, outcome)

        assert chunks == ["Done."]
        assert handler.calls == [{"text": "x"}]
        assert outcome.ok
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert messages[2]["tool_calls"][0]["id"] == "call_1"
        assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "echoed"}
        # Second request sees the tool result
        assert llm.requests[1]["messages"][-1]["role"] == "tool"

    async def test_tool_definitions_sent(self):
        orch, llm = make_orchestrator([text_turn("Hi.")], handlers=[EchoHandler()])
        await collect(orch, base_messages())
        assert [t["function"]["name"] for t in llm.requests[0]["tools"]] == ["echo"]

    async def test_short_pre_tool_filler_is_not_spoken_or_stored(self):
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "echo", "{}"), content="Let me check."),
             text_turn("Done.")],
            handlers=[EchoHandler()],
        )
        messages = base_messages()
        chunks = await collect(orch, messages)
        assert chunks == ["Done."]
        assert messages[2]["content"] is None
        assert messages[2]["tool_calls"][0]["id"] == "c1"

    async def test_long_pre_tool_text_is_spoken_and_flushed(self):
        long_text = "Sure. " + "x" * HOLD_PRE_TOOL_CHARS
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "echo", "{}"), content=long_text), text_turn("Done.")],
            handlers=[EchoHandler()],
        )
        messages = base_messages()
        chunks = await collect(orch, messages)
        assert chunks == [long_text, "\n", "Done."]
        assert messages[2]["content"] == long_text.strip()

    async def test_whitespace_pre_tool_text_stored_as_none(self):
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "echo", "{}"), content="\n\n"), text_turn("Done.")],
            handlers=[EchoHandler()],
        )
        messages = base_messages()
        await collect(orch, messages)
        assert messages[2]["content"] is None

    async def test_parallel_tool_calls_all_answered_in_order(self):
        handler = EchoHandler()
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("a", "echo", "{}"), ToolCall("b", "echo", "{}")),
             text_turn("Done.")],
            handlers=[handler],
        )
        messages = base_messages()
        await collect(orch, messages)
        assert [m.get("tool_call_id") for m in messages[3:5]] == ["a", "b"]
        assert len(handler.calls) == 2


class TestBadToolCalls:
    @pytest.mark.parametrize("arguments", ["not json", "[1, 2]", "null", '"str"'])
    async def test_invalid_arguments_still_answered(self, arguments):
        handler = EchoHandler()
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "echo", arguments)), text_turn("Sorry.")],
            handlers=[handler],
        )
        messages = base_messages()
        outcome = TurnOutcome()
        await collect(orch, messages, outcome)
        assert handler.calls == []
        assert messages[3] == {"role": "tool", "tool_call_id": "c1",
                               "content": "Error: Invalid arguments"}
        assert outcome.ok

    @pytest.mark.parametrize("arguments", ["", None])
    async def test_empty_arguments_treated_as_no_args(self, arguments):
        handler = EchoHandler()
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "echo", arguments)), text_turn("Ok.")], handlers=[handler]
        )
        await collect(orch, base_messages())
        assert handler.calls == [{}]

    async def test_unknown_tool_gets_error_result(self):
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "nope", "{}")), text_turn("Hmm.")]
        )
        messages = base_messages()
        await collect(orch, messages)
        assert messages[3]["content"] == "Error: Unknown tool: nope"

    async def test_tool_exception_becomes_error_result(self):
        orch, _ = make_orchestrator(
            [tool_turn(ToolCall("c1", "echo", "{}")), text_turn("Hmm.")],
            handlers=[EchoHandler(fail=True)],
        )
        messages = base_messages()
        await collect(orch, messages)
        assert messages[3]["content"].startswith("Error: Tool error: boom")


class TestDiscordActions:
    async def test_brief_ack_note_carried_in_tool_result(self):
        tag = EchoHandler(name="tag_user", output="Done! Tagged Bob.")
        other = EchoHandler(name="echo")
        orch, llm = make_orchestrator(
            [tool_turn(ToolCall("a", "tag_user", "{}"), ToolCall("b", "echo", "{}")),
             text_turn("Tagged.")],
            handlers=[tag, other],
        )
        await collect(orch, base_messages())
        sent = llm.requests[1]["messages"]
        # Qwen rejects system messages after the first one
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "tool"]
        assert sent[3]["content"].startswith("Done! Tagged Bob.")
        assert DISCORD_ACTION_ACK_NOTE in sent[3]["content"]
        assert DISCORD_ACTION_ACK_NOTE not in sent[4]["content"]

    async def test_no_ack_note_when_action_failed(self):
        tag = EchoHandler(name="tag_user", output="Could not find user Bob.")
        orch, llm = make_orchestrator(
            [tool_turn(ToolCall("a", "tag_user", "{}")), text_turn("Hmm.")], handlers=[tag]
        )
        await collect(orch, base_messages())
        assert llm.requests[1]["messages"][-1]["content"] == "Could not find user Bob."


def test_only_one_system_message_ever_sent():
    """Every request keeps a single system message at index 0."""
    import asyncio

    orch, llm = make_orchestrator(
        [tool_turn(ToolCall("a", "post_to_chat", "{}")), text_turn("Done.")],
        handlers=[EchoHandler(name="post_to_chat", output="Done! Posted.")],
    )
    asyncio.run(collect(orch, base_messages()))
    for request in llm.requests:
        roles = [m["role"] for m in request["messages"]]
        assert roles[0] == "system" and roles.count("system") == 1


class TestFailures:
    async def test_api_error_speaks_apology_and_not_ok(self):
        orch, _ = make_orchestrator([RuntimeError("503")])
        messages = base_messages()
        outcome = TurnOutcome()
        chunks = await collect(orch, messages, outcome)
        assert chunks == [ERROR_REPLY]
        assert not outcome.ok
        assert messages == base_messages()  # error text never enters history

    async def test_empty_answer_is_graceful_failure(self):
        orch, _ = make_orchestrator([text_turn("\n\n")])
        outcome = TurnOutcome()
        chunks = await collect(orch, base_messages(), outcome)
        assert chunks == ["\n\n", GRACEFUL_FAILURE_REPLY]
        assert not outcome.ok

    async def test_max_turns_exhausted(self):
        call = ToolCall("c", "echo", "{}")
        orch, llm = make_orchestrator(
            [tool_turn(call), tool_turn(call)], handlers=[EchoHandler()], max_turns=2
        )
        outcome = TurnOutcome()
        chunks = await collect(orch, base_messages(), outcome)
        assert chunks == [GRACEFUL_FAILURE_REPLY]
        assert len(llm.requests) == 2
        assert not outcome.ok

    async def test_outcome_optional(self):
        orch, _ = make_orchestrator([text_turn("Hi.")])
        assert await collect(orch, base_messages()) == ["Hi."]


def test_tool_message_has_no_legacy_name_field():
    """Cerebras API v2 is strict; tool messages carry only role/tool_call_id/content."""
    import asyncio

    orch, _ = make_orchestrator(
        [tool_turn(ToolCall("c1", "echo", json.dumps({"text": "x"}))), text_turn("Ok.")],
        handlers=[EchoHandler()],
    )
    messages = base_messages()
    asyncio.run(collect(orch, messages))
    assert set(messages[3]) == {"role", "tool_call_id", "content"}
