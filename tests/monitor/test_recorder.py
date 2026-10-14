"""TurnRecorder: hooks and log records in, one LiveTurn out."""

import asyncio
import json
import logging

import numpy as np
import pytest

from src.llm.chat_orchestrator import ERROR_REPLY, TurnOutcome
from src.monitor import turn as tr
from src.monitor.turn import TurnRecorder, tools_from_messages
from src.pipeline.base import PipelineContext
from src.pipeline.response_stage import ResponseStage, ResponseTurn
from src.pipeline.transcription_stage import TranscriptionResult
from src.pipeline.tts_stage import TTSStage
from src.pipeline.voice_pipeline import VoicePipeline

from .helpers import FakeClock


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(24000 * seconds), dtype=np.float32)


@pytest.fixture
def setup():
    clock = FakeClock(100.0)
    turns = []
    recorder = TurnRecorder(turns.append, clock=clock)
    return recorder, turns, clock


def web_messages():
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "web_search",
                          "arguments": json.dumps({"query": "perth weather"})}},
            {"id": "c2", "type": "function",
             "function": {"name": "web_extract",
                          "arguments": json.dumps({"urls": ["https://bom.gov.au/wa/perth?x=1"]})}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": 'No results for "perth weather". Try...'},
        {"role": "tool", "tool_call_id": "c2", "content": "Error: Could not read any of the pages"},
        {"role": "assistant", "content": "It's sunny."},
    ]


class TestTurnFlow:
    def test_full_turn(self, setup):
        recorder, turns, clock = setup
        token = recorder.turn_started(7, "Sam", speech_end=99.5)  # last packet 0.5s ago
        clock.advance(0.2)
        opening = recorder.clip_started(audio(0.8))
        clock.advance(0.3)
        recorder.transcribed("Hey Jarvis, weather?", "weather?")
        clock.advance(0.5)
        recorder.clip_ended(opening)
        clock.advance(2.0)
        recorder.answer_clip_next()
        first = recorder.clip_started(audio(1.0))
        clock.advance(1.0)
        recorder.clip_ended(first)
        clock.advance(0.3)
        recorder.answer_clip_next()
        second = recorder.clip_started(audio(1.0))
        clock.advance(1.0)
        recorder.clip_ended(second)
        recorder.replied(ResponseTurn(text="It's sunny.", ok=True, messages=web_messages()))
        recorder.pipeline_result(None)
        recorder.turn_ended(token)

        assert len(turns) == 1
        turn = turns[0]
        assert turn.speaker_name == "Sam" and turn.heard == "weather?"
        assert turn.outcome == tr.ANSWERED
        assert turn.first_audio_s == 0.7  # measured from the last packet, like the e2e harness
        assert turn.answer_audio_s == 3.5
        assert turn.answer_gaps_s == [0.3]
        assert turn.stt_s == 1.0
        assert [t.name for t in turn.tools] == ["web_search", "web_extract"]
        assert turn.tools[0].no_results and turn.tools[0].args == '"perth weather"'
        assert not turn.tools[1].ok and turn.tools[1].args == "bom.gov.au"
        assert turn.used_web

    def test_stale_speech_end_falls_back_to_now(self, setup):
        recorder, turns, clock = setup
        token = recorder.turn_started(1, "Sam", speech_end=10.0)  # 90s ago: not this question
        clip = recorder.clip_started(audio(0.5))
        recorder.clip_ended(clip)
        recorder.turn_ended(token)
        assert turns[0].first_audio_s == 0.0

    def test_error_reply_outcome(self, setup):
        recorder, turns, _ = setup
        token = recorder.turn_started(1, "Sam")
        recorder.replied(ResponseTurn(text=ERROR_REPLY, ok=False))
        recorder.turn_ended(token)
        assert turns[0].outcome == tr.ERROR_REPLY_OUTCOME

    def test_pipeline_result_wins(self, setup):
        recorder, turns, _ = setup
        token = recorder.turn_started(1, "Sam")
        recorder.pipeline_result("empty")
        recorder.turn_ended(token)
        assert turns[0].outcome == tr.EMPTY

    def test_exception_and_cancel(self, setup):
        recorder, turns, _ = setup
        token = recorder.turn_started(1, "Sam")
        recorder.turn_ended(token, RuntimeError("boom"))
        token = recorder.turn_started(1, "Sam")
        recorder.turn_ended(token, asyncio.CancelledError())
        assert [t.outcome for t in turns] == [tr.CRASHED, tr.CANCELLED]
        assert "boom" in turns[0].error

    def test_nested_turn_replaces_outer(self, setup):
        recorder, turns, _ = setup
        outer = recorder.turn_started(1, "Sam")
        recorder.pipeline_result("wake_word_only")
        inner = recorder.turn_started(1, "Sam")  # follow-up question after the ack
        recorder.replied(ResponseTurn(text="Sure.", ok=True))
        recorder.turn_ended(inner)
        recorder.turn_ended(outer)
        assert [t.outcome for t in turns] == [tr.ANSWERED]

    def test_clips_outside_a_turn_are_ignored(self, setup):
        recorder, turns, _ = setup
        assert recorder.clip_started(audio(1.0)) is None
        recorder.clip_ended(None)
        assert turns == []

    def test_stale_response_turn_is_ignored(self, setup):
        recorder, turns, _ = setup
        last = ResponseTurn(text="Old answer.", ok=True)
        token = recorder.turn_started(1, "Sam")
        recorder.replied(last)
        recorder.turn_ended(token)
        token = recorder.turn_started(1, "Sam")
        recorder.replied(last)  # LLM never ran this time
        recorder.turn_ended(token)
        assert turns[1].reply == "" and turns[1].outcome == tr.NO_REPLY

    def test_hooks_never_raise(self, setup):
        recorder, _, _ = setup
        recorder.turn_started(1, "Sam")
        assert recorder.clip_started(None) is None  # len(None) fails inside; swallowed
        recorder.replied(object())

    def test_handler_errors_do_not_reach_caller(self):
        def broken(turn):
            raise RuntimeError("handler bug")

        recorder = TurnRecorder(broken)
        token = recorder.turn_started(1, "Sam")
        recorder.turn_ended(token)  # no exception


class TestLogs:
    def test_collects_warnings_and_timings(self, setup):
        recorder, turns, _ = setup
        log = logging.getLogger("src.fake_component")
        log.setLevel(logging.INFO)
        recorder.install()
        try:
            log.info("before the turn: ignored", extra={"llm_turn": {"total_s": 9}})
            token = recorder.turn_started(1, "Sam")
            log.info("[TIMING] LLM turn", extra={"llm_turn": {"first_text_s": 0.4, "total_s": 1.1,
                                                               "tool_calls": 1}})
            log.info("Inworld TTS generated", extra={"tts_seconds": 0.734})
            log.info("plain info: not kept")
            log.warning("possible repeat - synthesizing again")
            logging.getLogger("src.monitor.service").warning("own log: not kept")
            recorder.turn_ended(token)
            log.error("after the turn: ignored")
        finally:
            recorder.uninstall()
        turn = turns[0]
        assert turn.llm_calls == [{"first_text_s": 0.4, "total_s": 1.1, "tool_calls": 1}]
        assert turn.tts_s == [0.73]
        assert [e.message for e in turn.events] == ["possible repeat - synthesizing again"]
        assert turn.events[0].logger == "src.fake_component"

    def test_uninstall_removes_handler(self):
        recorder = TurnRecorder(lambda t: None)
        root = logging.getLogger()
        before = len(root.handlers)
        recorder.install()
        recorder.install()  # idempotent
        assert len(root.handlers) == before + 1
        recorder.uninstall()
        assert len(root.handlers) == before


class TestHang:
    def test_reports_once_then_final_turn_is_marked(self, setup):
        recorder, turns, clock = setup
        token = recorder.turn_started(1, "Sam")
        recorder.transcribed("hey jarvis what", "what")
        clock.advance(10)
        assert recorder.check_hung(30) is None
        clock.advance(25)
        snapshot = recorder.check_hung(30)
        assert snapshot.outcome == tr.HUNG and snapshot.idle_s == 35.0
        assert snapshot.last_step == "waiting for the LLM's first response"
        assert recorder.check_hung(30) is None  # once per turn
        recorder.replied(ResponseTurn(text=ERROR_REPLY))
        recorder.turn_ended(token)
        assert [t.outcome for t in turns] == [tr.HUNG, tr.ERROR_REPLY_OUTCOME]
        assert turns[1].hang_reported

    def test_waiting_for_follow_up_is_not_a_hang(self, setup):
        recorder, _, clock = setup
        recorder.turn_started(1, "Sam")
        recorder.pipeline_result("wake_word_only")
        clock.advance(60)
        assert recorder.check_hung(30) is None

    def test_playing_audio_is_progress(self, setup):
        recorder, _, clock = setup
        recorder.turn_started(1, "Sam")
        recorder.clip_started(audio(40.0))  # a long story
        clock.advance(35)
        assert recorder.check_hung(30) is None


class TestToolsFromMessages:
    def test_bad_arguments_and_errors(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "a", "function": {"name": "post_to_chat", "arguments": "{bad"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "Error: Invalid arguments"},
        ]
        [use] = tools_from_messages(msgs)
        assert use.args == "(unparseable arguments)" and use.error == "Invalid arguments"


class TestPipelineWiring:
    """The real VoicePipeline/ResponseStage/TTSStage notify the recorder."""

    async def test_answer_clips_reply_and_tools(self):
        turns = []
        recorder = TurnRecorder(turns.append)

        class Orchestrator:
            async def chat_stream(self, messages, outcome: TurnOutcome):
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": "t1", "function": {"name": "web_search",
                                              "arguments": '{"query": "q"}'}}]})
                messages.append({"role": "tool", "tool_call_id": "t1", "content": "Results..."})
                yield "First sentence. "
                yield "Second one."
                messages.append({"role": "assistant", "content": "First sentence. Second one."})
                outcome.ok = True

        class Synth:
            async def synthesize(self, text):
                return audio(0.1)

        class Transcriber:
            async def process(self, audio_in, context):
                return TranscriptionResult(text="question", raw_text="hey jarvis question")

            async def cancel(self):
                pass

        pipeline = VoicePipeline(Transcriber(), ResponseStage(Orchestrator(), lambda: "sys"),
                                 TTSStage(Synth(), inter_sentence_pause_ms=0))
        pipeline.observer = recorder

        async def play(chunk):
            clip = recorder.clip_started(chunk)
            await asyncio.sleep(0)
            recorder.clip_ended(clip)

        async def quiet():
            return True

        token = recorder.turn_started(1, "Sam")
        result = await pipeline.process(audio(0.5), PipelineContext(speaker_name="Sam"),
                                        play_audio=play, wait_for_silence=quiet)
        recorder.pipeline_result(result)
        recorder.turn_ended(token)

        [turn] = turns
        assert turn.heard == "question"
        assert turn.reply == "First sentence. Second one."
        assert turn.reply_ok and turn.outcome == tr.ANSWERED
        assert [c.answer for c in turn.clips] == [True, True]
        assert [t.name for t in turn.tools] == ["web_search"]
