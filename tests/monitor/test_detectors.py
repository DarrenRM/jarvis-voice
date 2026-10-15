"""Detectors fed synthetic turn records."""

from src.llm.chat_orchestrator import ERROR_REPLY, GRACEFUL_FAILURE_REPLY
from src.monitor import turn as tr
from src.monitor.detectors import Finding, Thresholds, detect, normalize_cause
from src.monitor.turn import Clip

from .helpers import event, good_turn, tool


def detectors(turn, th=Thresholds()) -> list[str]:
    return [f.detector for f in detect(turn, th)]


class TestQuietOnGoodTurns:
    def test_good_turn_has_no_findings(self):
        assert detect(good_turn()) == []

    def test_web_turn_within_limits(self):
        turn = good_turn(tools=[tool("web_search")],
                         clips=[Clip(0.8, 0.8, 1.6), Clip(9.0, 2.0, 11.0, answer=True)])
        assert detect(turn) == []

    def test_cancelled_turn_is_ignored(self):
        turn = good_turn(outcome=tr.CANCELLED, events=[event("LLM API error: boom")])
        assert detect(turn) == []

    def test_casual_dont_know_without_lookup_is_fine(self):
        assert detectors(good_turn(reply="I don't know, pineapple is fine by me.")) == []

    def test_wake_word_only_is_quiet(self):
        assert detect(good_turn(outcome=tr.WAKE_WORD_ONLY, reply="", clips=[])) == []


class TestFailures:
    def test_error_reply_names_the_llm_error(self):
        turn = good_turn(outcome=tr.ERROR_REPLY_OUTCOME, reply=ERROR_REPLY,
                         events=[event("LLM API error: Request timed out.")])
        findings = detect(turn)
        assert findings[0].detector == "error_reply"
        assert "Request timed out" in findings[0].detail
        # The LLM error is explained by the error reply, not posted again
        assert "unexpected_error" not in detectors(turn)

    def test_graceful_failure_is_not_also_a_give_up(self):
        turn = good_turn(outcome=tr.GRACEFUL_FAILURE, reply=GRACEFUL_FAILURE_REPLY,
                         events=[event("LLM returned no text (finish=length)", "WARNING")])
        assert detectors(turn) == ["graceful_failure"]

    def test_crash(self):
        turn = good_turn(outcome=tr.CRASHED, error="RuntimeError: boom", reply="", clips=[])
        assert detectors(turn)[0] == "crash"

    def test_no_reply(self):
        assert detectors(good_turn(outcome=tr.NO_REPLY, reply="", clips=[]))[0] == "no_reply"

    def test_hang(self):
        turn = good_turn(outcome=tr.HUNG, idle_s=31.0,
                         last_step="waiting for the LLM's first response")
        findings = detect(turn)
        assert [f.detector for f in findings] == ["hang"]
        assert "31s" in findings[0].detail and "first response" in findings[0].detail


class TestAnswers:
    def test_give_up_strong_phrase(self):
        turn = good_turn(reply="[sigh] My weather feeds just ghosted me.",
                         tools=[tool("web_search")])
        assert "give_up" in detectors(turn)

    def test_give_up_weak_phrase_needs_a_lookup(self):
        turn = good_turn(reply="I'm not sure, the wiki was vague.", tools=[tool("web_search")])
        assert "give_up" in detectors(turn)

    def test_unclear_question(self):
        assert "unclear_question" in detectors(good_turn(reply="Sorry, I didn't catch that."))

    def test_markdown_and_links(self):
        turn = good_turn(reply="**Canberra** is the capital. See https://example.com/x")
        finding = next(f for f in detect(turn) if f.detector == "formatting")
        assert "link" in finding.detail and "bold" in finding.detail

    def test_tts_mismatch_when_retranscribed(self):
        repeated = "It's Canberra. It's Canberra. It's Canberra."
        assert "tts_mismatch" in detectors(good_turn(reply="It's Canberra.", spoken=repeated))
        clean = good_turn(reply="It's Canberra.", spoken="It's Canberra.")
        assert "tts_mismatch" not in detectors(clean)

    def test_tag_read_aloud(self):
        turn = good_turn(reply="[laughing] Sure.", spoken="Laughing. Sure.")
        assert "spoken_artifacts" in detectors(turn)


class TestTiming:
    def test_slow_first_audio(self):
        turn = good_turn()
        turn.clips[0] = Clip(4.0, 0.8, 4.8)
        assert "slow_first_audio" in detectors(turn)

    def test_first_audio_ignored_when_answer_is_first(self):
        # Affirmation skipped (someone talking): the answer is the first sound
        turn = good_turn(clips=[Clip(4.0, 1.0, 5.0, answer=True)])
        assert detect(turn) == []

    def test_slow_answer_without_web(self):
        turn = good_turn(clips=[Clip(0.7, 0.8, 1.5), Clip(9.5, 1.0, 10.5, answer=True)])
        finding = next(f for f in detect(turn) if f.detector == "slow_answer")
        assert "9.5s" in finding.detail and "LLM call" in finding.detail

    def test_slow_answer_with_web_uses_web_limit(self):
        clips = [Clip(0.7, 0.8, 1.5), Clip(25.0, 1.0, 26.0, answer=True)]
        assert "slow_answer" in detectors(good_turn(clips=clips, tools=[tool("web_search")]))

    def test_long_gap_between_sentences(self):
        turn = good_turn()
        turn.clips[2] = Clip(8.0, 1.5, 9.5, answer=True)  # 3.6s after the previous one ended
        assert "long_gap" in detectors(turn)

    def test_answer_never_played(self):
        turn = good_turn(clips=[Clip(0.7, 0.8, 1.5)])
        assert "no_answer_audio" in detectors(turn)

    def test_timing_skipped_after_hang_report(self):
        turn = good_turn(clips=[Clip(0.7, 0.8, 1.5), Clip(40.0, 1.0, 41.0, answer=True)],
                         hang_reported=True)
        assert detect(turn) == []


class TestTools:
    def test_timeout(self):
        turn = good_turn(tools=[tool("web_search", ok=False,
                                     error="Search timed out. Answer from what you know.")])
        assert "tool_timeout" in detectors(turn)

    def test_recovered_error_is_not_reported(self):
        turn = good_turn(tools=[tool("web_search", ok=False, error="Search failed: 502"),
                                tool("web_search")])
        assert detectors(turn) == []

    def test_web_search_found_nothing(self):
        turn = good_turn(tools=[tool("web_search", '"post pizza ate"', no_results=True)])
        finding = next(f for f in detect(turn) if f.detector == "web_no_results")
        assert "post pizza ate" in finding.detail

    def test_no_results_then_success_is_fine(self):
        turn = good_turn(tools=[tool("web_search", no_results=True), tool("web_search")])
        assert detectors(turn) == []


class TestLogEvents:
    def test_tts_repeat_guard(self):
        turn = good_turn(events=[event("[TTS] Clip is 9.0s for 40 chars (limit 4.0s), possible "
                                       "repeat - synthesizing again: 'x'", "WARNING",
                                       "src.tts.synthesizer")])
        assert detectors(turn) == ["tts_repeat"]

    def test_thinking_phrase_failure(self):
        message = "Pipeline: Dynamic thinking phrase failed: timeout, skipping"
        turn = good_turn(events=[event(message, "WARNING", "src.pipeline.voice_pipeline")])
        assert detectors(turn) == ["thinking_phrase_failed"]

    def test_stt_error_on_empty_turn(self):
        turn = good_turn(outcome=tr.EMPTY, reply="", clips=[],
                         events=[event("Groq transcription error: 503",
                                       logger="src.stt.groq_whisper")])
        assert detectors(turn) == ["stt_error"]

    def test_unexpected_error(self):
        turn = good_turn(events=[event("Something odd broke", logger="src.bot.voice_session")])
        assert detectors(turn) == ["unexpected_error"]

    def test_library_errors_are_ignored(self):
        turn = good_turn(events=[event("gateway hiccup", logger="discord.gateway")])
        assert detectors(turn) == []

    def test_unrelated_warning_is_ignored(self):
        turn = good_turn(events=[event("[PLAYBACK] Timeout waiting for silence, playing anyway",
                                       "WARNING", "src.bot.voice_session")])
        assert detectors(turn) == []


class TestDedupeKeys:
    def test_numbers_and_urls_collapse(self):
        a = Finding("tool_error", "", "", cause="web_extract Could not read https://a.com/1 (403)")
        b = Finding("tool_error", "", "", cause="web_extract Could not read https://b.org/x (404)")
        assert a.key == b.key

    def test_different_detectors_differ(self):
        assert Finding("a", "", "", "x").key != Finding("b", "", "", "x").key

    def test_normalize_cause_quotes(self):
        assert normalize_cause("No results for 'abc'") == normalize_cause('No results for "xyz"')
