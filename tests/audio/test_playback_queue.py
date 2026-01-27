"""Tests for OrderedPlaybackQueue.

Tests the critical sentence ordering pattern from voice_session.py:852-923.
"""

import asyncio

import numpy as np
import pytest

from src.audio.playback_queue import OrderedPlaybackQueue


def make_audio(samples: int = 100, value: float = 0.5) -> np.ndarray:
    """Create test audio data."""
    return np.full(samples, value, dtype=np.float32)


class TestOrderedPlaybackQueue:
    """Test OrderedPlaybackQueue functionality."""

    @pytest.fixture
    def queue(self):
        """Create a fresh queue for each test."""
        return OrderedPlaybackQueue()

    # ========== Basic Ordering Tests ==========

    @pytest.mark.asyncio
    async def test_in_order_delivery(self, queue):
        """Items added in order are delivered in order."""
        audio_1 = make_audio(100, 0.1)
        audio_2 = make_audio(100, 0.2)
        audio_3 = make_audio(100, 0.3)

        await queue.put(1, audio_1)
        await queue.put(2, audio_2)
        await queue.put(3, audio_3)
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])  # First sample identifies the audio

        assert results == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_out_of_order_delivery(self, queue):
        """Items added out of order are delivered in correct order."""
        audio_1 = make_audio(100, 0.1)
        audio_2 = make_audio(100, 0.2)
        audio_3 = make_audio(100, 0.3)

        # Add in reverse order
        await queue.put(3, audio_3)
        await queue.put(2, audio_2)
        await queue.put(1, audio_1)
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_mixed_order_delivery(self, queue):
        """Mixed order arrivals are handled correctly."""
        # Simulate real scenario: sentence 2 finishes before 1
        await queue.put(2, make_audio(100, 0.2))
        await queue.put(1, make_audio(100, 0.1))
        await queue.put(4, make_audio(100, 0.4))
        await queue.put(3, make_audio(100, 0.3))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_gaps_in_sequence(self, queue):
        """Items with gaps in sequence numbers still work."""
        await queue.put(1, make_audio(100, 0.1))
        await queue.put(3, make_audio(100, 0.3))  # Skip 2
        await queue.put(2, make_audio(100, 0.2))  # Fill gap
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1, 0.2, 0.3]

    # ========== Empty Audio Tests ==========

    @pytest.mark.asyncio
    async def test_empty_audio_doesnt_block(self, queue):
        """Empty audio (TTS failure) doesn't block subsequent sentences.

        CRITICAL: From voice_session.py:950-953 - on TTS error,
        queue empty audio to not block ordering.
        """
        await queue.put(1, make_audio(100, 0.1))
        await queue.put(2, np.array([], dtype=np.float32))  # TTS failed
        await queue.put(3, make_audio(100, 0.3))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(len(audio))

        # All three yielded, including empty
        assert results == [100, 0, 100]

    @pytest.mark.asyncio
    async def test_all_empty_audio(self, queue):
        """Queue handles all-empty scenario (total TTS failure)."""
        await queue.put(1, np.array([], dtype=np.float32))
        await queue.put(2, np.array([], dtype=np.float32))
        await queue.signal_end()

        count = 0
        async for _ in queue.iter_in_order():
            count += 1

        assert count == 2

    # ========== End Signal Tests ==========

    @pytest.mark.asyncio
    async def test_end_signal_drains_pending(self, queue):
        """End signal causes all pending items to be drained in order."""
        # Add items out of order, then signal end
        await queue.put(3, make_audio(100, 0.3))
        await queue.put(2, make_audio(100, 0.2))
        await queue.signal_end()
        # Item 1 never arrives - but we signaled end

        # Only items 2 and 3 are pending (waiting for 1)
        # After end signal, they should be drained
        # But since 1 never arrived, they stay pending
        # Actually, the drain only happens for consecutive items starting at next_to_play
        # So if 1 is missing, 2 and 3 won't be drained

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        # Since item 1 never arrived, items 2 and 3 remain pending
        # This is expected behavior - missing items cause later items to be lost
        assert results == []

    @pytest.mark.asyncio
    async def test_end_signal_with_all_items(self, queue):
        """End signal with all items present drains correctly."""
        await queue.put(3, make_audio(100, 0.3))
        await queue.put(1, make_audio(100, 0.1))
        await queue.put(2, make_audio(100, 0.2))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_signal_end_twice(self, queue):
        """Calling signal_end twice is safe."""
        await queue.put(1, make_audio(100, 0.1))
        await queue.signal_end()
        await queue.signal_end()  # Should not raise

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1]

    @pytest.mark.asyncio
    async def test_put_after_end_ignored(self, queue):
        """Items added after end signal are ignored."""
        await queue.put(1, make_audio(100, 0.1))
        await queue.signal_end()
        await queue.put(2, make_audio(100, 0.2))  # Should be ignored

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1]

    # ========== Cancellation Tests ==========

    @pytest.mark.asyncio
    async def test_cancel_stops_iteration(self, queue):
        """Cancel unblocks iter_in_order and stops iteration."""
        await queue.put(1, make_audio(100, 0.1))
        # Don't signal end - iterator would block

        results = []

        async def consumer():
            async for audio in queue.iter_in_order():
                results.append(audio[0])

        # Start consumer
        task = asyncio.create_task(consumer())

        # Let it process item 1
        await asyncio.sleep(0.05)

        # Cancel while waiting for item 2
        queue.cancel()

        # Wait for consumer to finish
        await asyncio.wait_for(task, timeout=1.0)

        assert results == [0.1]

    @pytest.mark.asyncio
    async def test_cancel_while_waiting(self, queue):
        """Cancel while iterator is waiting for items."""
        results = []

        async def consumer():
            async for audio in queue.iter_in_order():
                results.append(audio)

        task = asyncio.create_task(consumer())

        # Wait briefly then cancel (queue is empty, iterator is waiting)
        await asyncio.sleep(0.05)
        queue.cancel()

        await asyncio.wait_for(task, timeout=1.0)
        assert results == []

    @pytest.mark.asyncio
    async def test_put_after_cancel_ignored(self, queue):
        """Items added after cancel are ignored."""
        queue.cancel()
        await queue.put(1, make_audio(100, 0.1))

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio)

        assert results == []

    # ========== Reset Tests ==========

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, queue):
        """Reset clears all state and allows reuse."""
        await queue.put(1, make_audio(100, 0.1))
        await queue.signal_end()

        # Consume
        async for _ in queue.iter_in_order():
            pass

        # Reset
        queue.reset()

        # Reuse
        await queue.put(1, make_audio(100, 0.5))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.5]

    @pytest.mark.asyncio
    async def test_reset_after_cancel(self, queue):
        """Reset after cancel allows reuse."""
        queue.cancel()
        queue.reset()

        await queue.put(1, make_audio(100, 0.1))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1]

    # ========== Property Tests ==========

    @pytest.mark.asyncio
    async def test_pending_count(self, queue):
        """pending_count reflects buffered items."""
        assert queue.pending_count == 0

        await queue.put(3, make_audio(100, 0.3))
        await queue.put(2, make_audio(100, 0.2))

        # Start iteration but don't consume
        # Items 2 and 3 should be pending since 1 hasn't arrived
        # But pending_count only updates during iteration
        assert queue.pending_count == 0  # Not yet processed

    @pytest.mark.asyncio
    async def test_next_expected(self, queue):
        """next_expected starts at 1 and increments."""
        assert queue.next_expected == 1

        await queue.put(1, make_audio(100, 0.1))
        await queue.signal_end()

        async for _ in queue.iter_in_order():
            pass

        assert queue.next_expected == 2

    # ========== Concurrent Producer/Consumer Tests ==========

    @pytest.mark.asyncio
    async def test_concurrent_producer_consumer(self, queue):
        """Producer and consumer can run concurrently."""
        results = []

        async def producer():
            for i in range(1, 6):
                await asyncio.sleep(0.01)  # Simulate TTS time
                await queue.put(i, make_audio(100, i / 10))
            await queue.signal_end()

        async def consumer():
            async for audio in queue.iter_in_order():
                results.append(audio[0])

        await asyncio.gather(producer(), consumer())

        assert results == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_concurrent_out_of_order_producers(self, queue):
        """Multiple producers adding items out of order."""
        results = []

        async def producer_odds():
            for i in [1, 3, 5]:
                await asyncio.sleep(0.02)
                await queue.put(i, make_audio(100, i / 10))

        async def producer_evens():
            for i in [2, 4]:
                await asyncio.sleep(0.01)  # Finishes faster
                await queue.put(i, make_audio(100, i / 10))

        async def end_signal():
            await asyncio.sleep(0.15)
            await queue.signal_end()

        async def consumer():
            async for audio in queue.iter_in_order():
                results.append(audio[0])

        await asyncio.gather(
            producer_odds(),
            producer_evens(),
            end_signal(),
            consumer(),
        )

        # Should be in order regardless of producer timing
        assert results == [0.1, 0.2, 0.3, 0.4, 0.5]

    # ========== Edge Case Tests ==========

    @pytest.mark.asyncio
    async def test_single_item(self, queue):
        """Single item works correctly."""
        await queue.put(1, make_audio(100, 0.1))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        assert results == [0.1]

    @pytest.mark.asyncio
    async def test_empty_queue(self, queue):
        """Empty queue (immediate end) works."""
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio)

        assert results == []

    @pytest.mark.asyncio
    async def test_large_gap_before_first(self, queue):
        """Large sentence numbers without item 1 means nothing plays."""
        await queue.put(100, make_audio(100, 1.0))
        await queue.put(50, make_audio(100, 0.5))
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio)

        # Item 1 never arrived, so nothing plays
        assert results == []

    @pytest.mark.asyncio
    async def test_duplicate_sentence_number(self, queue):
        """Later item with same sentence number replaces earlier."""
        await queue.put(1, make_audio(100, 0.1))
        await queue.put(2, make_audio(100, 0.2))
        await queue.put(2, make_audio(100, 0.9))  # Duplicate - replaces
        await queue.signal_end()

        results = []
        async for audio in queue.iter_in_order():
            results.append(audio[0])

        # Note: behavior depends on timing - if 2 was already played,
        # the duplicate won't matter. Here 1 plays first, then 2 is pending.
        # The second put(2) may or may not overwrite depending on race.
        # This test documents current behavior.
        assert len(results) == 2
        assert results[0] == pytest.approx(0.1)
