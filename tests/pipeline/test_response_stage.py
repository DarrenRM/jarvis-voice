"""Tests for ResponseStage.

Tests the wrapper around ChatOrchestrator that adds speaker context and
keeps a bounded conversation history.
"""


from src.llm.chat_orchestrator import ChatOrchestrator, ToolCall, TurnResult
from src.pipeline.base import PipelineContext
from src.pipeline.response_stage import ResponseStage
from src.tools import ToolDefinition, ToolExecutor, ToolHandler, ToolRegistry, ToolResult


class FakeLLM:
    """Answers every turn with a fixed reply, or plays scripted turns."""

    def __init__(self, reply=("Hello ", "world!"), script=None):
        self._reply = reply
        self._script = list(script or [])
        self.requests: list[list[dict]] = []

    async def stream_turn(self, messages, tools=None):
        self.requests.append([dict(m) for m in messages])
        if self._script:
            turn = self._script.pop(0)
            if isinstance(turn, Exception):
                raise turn
            yield turn
            return
        for chunk in self._reply:
            yield chunk
        yield TurnResult(content="".join(self._reply), finish_reason="stop")


class Echo(ToolHandler):
    @property
    def name(self):
        return "echo"

    @property
    def definition(self):
        return ToolDefinition(name="echo", description="echo", parameters={"type": "object"})

    async def execute(self, args):
        return ToolResult(success=True, output="echoed")


def make_stage(llm=None, prompt="You are a helpful assistant.", max_history=60):
    llm = llm or FakeLLM()
    registry = ToolRegistry()
    registry.register(Echo())
    orchestrator = ChatOrchestrator(llm, ToolExecutor(registry))
    return ResponseStage(orchestrator, lambda: prompt, max_history_messages=max_history), llm


async def run(stage, text, context=None):
    return [c async for c in stage.process_stream(text, context or PipelineContext())]


class TestResponseStage:
    def test_name(self):
        stage, _ = make_stage()
        assert stage.name == "response"

    async def test_streams_response(self):
        stage, _ = make_stage()
        assert await run(stage, "hello") == ["Hello ", "world!"]

    async def test_system_prompt_first_in_request(self):
        stage, llm = make_stage()
        await run(stage, "hello")
        assert llm.requests[0][0] == {"role": "system", "content": "You are a helpful assistant."}

    async def test_history_excludes_system_prompt(self):
        stage, _ = make_stage()
        await run(stage, "hello", PipelineContext(speaker_name="Alice"))
        assert stage.get_messages() == [
            {"role": "user", "content": "[Alice]: hello"},
            {"role": "assistant", "content": "Hello world!"},
        ]


class TestSpeakerContext:
    async def test_speaker_name_prefix(self):
        stage, llm = make_stage()
        await run(stage, "hello", PipelineContext(speaker_name="Bob"))
        assert llm.requests[0][-1] == {"role": "user", "content": "[Bob]: hello"}

    async def test_no_speaker_name(self):
        stage, llm = make_stage()
        await run(stage, "hello")
        assert llm.requests[0][-1] == {"role": "user", "content": "hello"}

    async def test_speaker_activity_in_system_prompt(self):
        stage, llm = make_stage()
        await run(stage, "hello", PipelineContext(speaker_activity="playing Minecraft"))
        assert "The user is currently: playing Minecraft" in llm.requests[0][0]["content"]

    async def test_activity_is_per_turn_not_frozen(self):
        stage, llm = make_stage()
        await run(stage, "hi", PipelineContext(speaker_activity="playing Minecraft"))
        await run(stage, "hi", PipelineContext(speaker_activity="playing Dune"))
        assert "Dune" in llm.requests[1][0]["content"]
        assert "Minecraft" not in llm.requests[1][0]["content"]


class TestSystemPromptRefresh:
    async def test_prompt_rebuilt_every_turn(self):
        prompts = iter(["prompt at 10:00", "prompt at 10:05"])
        llm = FakeLLM()
        orchestrator = ChatOrchestrator(llm, ToolExecutor(ToolRegistry()))
        stage = ResponseStage(orchestrator, lambda: next(prompts))
        await run(stage, "one")
        await run(stage, "two")
        assert llm.requests[1][0]["content"] == "prompt at 10:05"
        assert sum(m["role"] == "system" for m in llm.requests[1]) == 1


class TestConversationHistory:
    async def test_history_persists(self):
        stage, llm = make_stage()
        await run(stage, "hello")
        await run(stage, "how are you")
        roles = [m["role"] for m in llm.requests[1]]
        assert roles == ["system", "user", "assistant", "user"]

    async def test_reset_conversation(self):
        stage, _ = make_stage()
        await run(stage, "hello")
        stage.reset_conversation()
        assert stage.get_messages() == []

    def test_get_messages_returns_copy(self):
        stage, _ = make_stage()
        stage.get_messages().append({"role": "test"})
        assert stage.get_messages() == []

    async def test_failed_turn_not_added_to_history(self):
        llm = FakeLLM(script=[RuntimeError("503")])
        stage, _ = make_stage(llm)
        chunks = await run(stage, "hello")
        assert chunks  # apology is still spoken
        assert stage.get_messages() == []

    async def test_tool_turn_kept_together_in_history(self):
        llm = FakeLLM(script=[
            TurnResult(tool_calls=[ToolCall("c1", "echo", "{}")], finish_reason="tool_calls"),
            TurnResult(content="Done.", finish_reason="stop"),
        ])
        stage, _ = make_stage(llm)
        await run(stage, "hello")
        roles = [m["role"] for m in stage.get_messages()]
        assert roles == ["user", "assistant", "tool", "assistant"]


class TestHistoryTrim:
    async def test_trims_oldest_whole_exchanges(self):
        stage, llm = make_stage(max_history=5)
        for i in range(4):
            await run(stage, f"msg {i}")
        history = stage.get_messages()
        assert len(history) <= 5
        assert history[0]["role"] == "user"
        assert history[-1]["role"] == "assistant"

    async def test_trim_never_starts_on_tool_message(self):
        script = []
        for _ in range(3):
            script += [
                TurnResult(tool_calls=[ToolCall("c", "echo", "{}")], finish_reason="tool_calls"),
                TurnResult(content="Done.", finish_reason="stop"),
            ]
        stage, _ = make_stage(FakeLLM(script=script), max_history=6)
        for i in range(3):
            await run(stage, f"msg {i}")
        history = stage.get_messages()
        assert history[0]["role"] == "user"
        # Every tool message directly follows an assistant tool_calls message or another tool
        for prev, msg in zip(history, history[1:]):
            if msg["role"] == "tool":
                assert prev["role"] == "tool" or prev.get("tool_calls")
