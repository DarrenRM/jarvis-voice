"""The e2e report's latency summary (the median quoted in the README comes from it)."""

from scripts.e2e.run_voice_e2e import RunResult, timing_summary


def run(first=None, answer=None) -> RunResult:
    metrics = {}
    if first is not None:
        metrics["first_audio_s"] = first
    if answer is not None:
        metrics["answer_audio_s"] = answer
    return RunResult(name="s", speeches=[], segments=[], turns=[], posts=[],
                     speech_end={}, duration=0.0, metrics=metrics)


def test_median_and_p90_over_runs_with_audio():
    results = [run(2.1, 5.5), run(2.2, 6.0), run(2.0, 5.0), run(2.6), run()]
    assert timing_summary(results) == [
        "First audio: median 2.15 s, p90 2.60 s (4 runs)  ",
        "Answer audio: median 5.50 s, p90 6.00 s (3 runs)  ",
    ]


def test_nothing_to_summarize():
    assert timing_summary([run()]) == []
