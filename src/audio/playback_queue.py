"""Ordered playback queue for parallel TTS synthesis.

This module provides an ordered queue that handles out-of-order audio arrivals
from parallel TTS tasks. Audio chunks are numbered and played back in order
regardless of synthesis completion order.

Critical pattern from voice_session.py:852-923 - MUST be preserved:
- pending dict for out-of-order handling
- next_to_play counter
- None sentinel signals end of stream
- Empty audio allowed (doesn't block ordering)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """Item in the playback queue."""

    sentence_num: int
    audio: np.ndarray


class OrderedPlaybackQueue:
    """Queue that delivers audio chunks in sentence order.

    Handles out-of-order arrivals from parallel TTS synthesis.
    Uses a pending dict to buffer items that arrive before their turn.

    Example:
        queue = OrderedPlaybackQueue()

        # Producer (TTS tasks)
        await queue.put(2, audio_2)  # Arrives first
        await queue.put(1, audio_1)  # Arrives second
        await queue.put(3, audio_3)
        await queue.signal_end()

        # Consumer (playback)
        async for audio in queue.iter_in_order():
            play_audio(audio)  # Plays 1, 2, 3 in order
    """

    def __init__(self):
        """Initialize the ordered playback queue."""
        self._queue: asyncio.Queue[Optional[QueueItem]] = asyncio.Queue()
        self._pending: dict[int, np.ndarray] = {}
        self._next_to_play = 1
        self._ended = False
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    async def put(self, sentence_num: int, audio: np.ndarray) -> None:
        """Add audio to the queue with its sentence number.

        Args:
            sentence_num: 1-based sentence number for ordering
            audio: Audio data as float32 numpy array (may be empty)

        Note:
            Empty audio is allowed and won't block ordering.
            This is critical for error recovery - if TTS fails,
            queue empty audio so subsequent sentences can play.
        """
        if self._ended or self._cancelled:
            logger.warning(f"Queue ended/cancelled, ignoring sentence {sentence_num}")
            return

        await self._queue.put(QueueItem(sentence_num=sentence_num, audio=audio))
        logger.debug(f"[QUEUE] Added sentence {sentence_num}, {len(audio)} samples")

    async def signal_end(self) -> None:
        """Signal that no more items will be added.

        Must be called after all TTS tasks complete.
        Consumer will drain pending items and then stop.
        """
        if not self._ended:
            self._ended = True
            await self._queue.put(None)
            logger.debug("[QUEUE] End signal sent")

    def cancel(self) -> None:
        """Cancel the queue, unblocking any waiters.

        After cancel, iter_in_order() will stop yielding.
        """
        self._cancelled = True
        self._cancel_event.set()
        logger.debug("[QUEUE] Cancelled")

    async def iter_in_order(self) -> AsyncIterator[np.ndarray]:
        """Iterate over audio chunks in sentence order.

        Yields audio in order (1, 2, 3, ...) regardless of arrival order.
        Handles out-of-order arrivals via internal pending buffer.

        Yields:
            Audio chunks as float32 numpy arrays, in sentence order.
            Empty arrays are yielded (caller should skip playback for those).
        """
        while not self._cancelled:
            try:
                # Wait for next item with cancellation support
                get_task = asyncio.create_task(self._queue.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())

                done, pending_tasks = await asyncio.wait(
                    [get_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                if self._cancelled:
                    break

                if get_task in done:
                    item = get_task.result()
                else:
                    continue

                # None signals end of stream
                if item is None:
                    # Drain any remaining pending items in order
                    while self._next_to_play in self._pending:
                        audio = self._pending.pop(self._next_to_play)
                        logger.debug(
                            f"[QUEUE] Draining pending sentence {self._next_to_play}"
                        )
                        yield audio
                        self._next_to_play += 1
                    logger.debug("[QUEUE] End of stream reached")
                    break

                # Buffer this item
                self._pending[item.sentence_num] = item.audio
                logger.debug(
                    f"[QUEUE] Buffered sentence {item.sentence_num}, "
                    f"next_to_play={self._next_to_play}"
                )

                # Yield all consecutive available sentences
                while self._next_to_play in self._pending:
                    audio = self._pending.pop(self._next_to_play)
                    logger.debug(f"[QUEUE] Yielding sentence {self._next_to_play}")
                    yield audio
                    self._next_to_play += 1

            except asyncio.CancelledError:
                logger.debug("[QUEUE] Iterator cancelled")
                break
            except Exception as e:
                logger.error(f"[QUEUE] Error in iter_in_order: {e}")
                break

    def reset(self) -> None:
        """Reset the queue for reuse.

        Clears all state and allows new items to be added.
        """
        self._queue = asyncio.Queue()
        self._pending.clear()
        self._next_to_play = 1
        self._ended = False
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    @property
    def pending_count(self) -> int:
        """Number of items buffered waiting for earlier sentences."""
        return len(self._pending)

    @property
    def next_expected(self) -> int:
        """Next sentence number expected to be played."""
        return self._next_to_play
