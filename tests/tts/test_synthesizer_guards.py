"""Tests for TTSSynthesizer's repeat guard and phrase cache (no network)."""

import numpy as np

from src.tts.synthesizer import TTSSynthesizer, _max_plausible_seconds


def make_tts(clips):
    """Synthesizer whose Inworld call returns the given clip lengths (seconds) in order."""
    tts = TTSSynthesizer(api_key="test-key")
    calls = []

    async def fake_inworld(text):
        calls.append(text)
        seconds = clips[min(len(calls), len(clips)) - 1]
        return np.zeros(int(seconds * tts.sample_rate), dtype=np.float32)

    tts._synthesize_inworld = fake_inworld
    return tts, calls


class TestRepeatGuard:
    async def test_normal_clip_not_resynthesized(self):
        tts, calls = make_tts([2.0])
        audio = await tts.synthesize("It's sixty eight degrees and sunny in Chicago.")
        assert len(calls) == 1
        assert len(audio) == 2 * tts.sample_rate

    async def test_overlong_clip_resynthesized_and_shorter_kept(self):
        text = "It's sixty eight degrees and sunny in Chicago."
        tts, calls = make_tts([8.0, 2.5])
        audio = await tts.synthesize(text)
        assert len(calls) == 2
        assert len(audio) == int(2.5 * tts.sample_rate)

    async def test_retry_that_is_no_better_is_ignored(self):
        tts, calls = make_tts([8.0, 9.0])
        audio = await tts.synthesize("It's sixty eight degrees and sunny in Chicago.")
        assert len(calls) == 2
        assert len(audio) == 8 * tts.sample_rate

    def test_nonverbal_tags_allow_extra_time(self):
        plain = _max_plausible_seconds("Okay fine.", 1.0)
        tagged = _max_plausible_seconds("[sigh] Okay fine.", 1.0)
        emotion = _max_plausible_seconds("[happy] Okay fine.", 1.0)
        assert tagged > plain
        assert emotion == plain

    def test_short_phrases_have_slack(self):
        # "Sure Tester..." takes ~1.2s in practice; must not trip the guard
        assert _max_plausible_seconds("Sure Tester...", 1.0) > 1.5


class TestPhraseCache:
    async def test_repeat_phrase_served_from_cache(self):
        tts, calls = make_tts([1.0])
        first = await tts.synthesize_cached("Sure Alice...")
        second = await tts.synthesize_cached("Sure Alice...")
        assert len(calls) == 1
        assert second is first

    async def test_empty_audio_not_cached(self):
        tts, calls = make_tts([0.0, 1.0])
        await tts.synthesize_cached("Yes?")
        audio = await tts.synthesize_cached("Yes?")
        assert len(calls) == 2
        assert len(audio) > 0


class TestUrlsNeverSpoken:
    def test_links_replaced(self):
        from src.tts.text_filter import strip_urls_for_tts as strip
        assert strip("Check https://enshrouded.wiki.gg/wiki/Glider for details.") == \
            "Check the link for details."
        assert strip("Go to www.reddit.com.") == "Go to the link."
        assert strip("It's at enshrouded.wiki.gg/wiki/Glider, I think.") == \
            "It's at the link, I think."

    def test_numbers_and_site_names_untouched(self):
        from src.tts.text_filter import strip_urls_for_tts as strip
        text = "Version 2.0 costs 3.5 dollars, e.g. on wiki.gg."
        assert strip(text) == text

    async def test_synthesize_never_sends_url(self):
        tts, calls = make_tts([2.0])
        await tts.synthesize("Read it at https://example.com/page now.")
        assert "http" not in calls[0] and "the link" in calls[0]
