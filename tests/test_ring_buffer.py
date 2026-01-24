"""Tests for ring buffer."""

import numpy as np
import pytest

from src.audio.ring_buffer import RingBuffer


class TestRingBuffer:
    """Tests for RingBuffer class."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)
        assert buffer.sample_rate == 16000
        assert buffer.max_samples == 80000
        assert buffer.is_empty
        assert len(buffer) == 0

    def test_write_read(self):
        """Test basic write and read."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)

        # Write audio
        audio = np.random.randn(1600).astype(np.float32)
        buffer.write(audio)

        assert not buffer.is_empty
        assert len(buffer) == 1600
        assert buffer.duration_s == pytest.approx(0.1, rel=1e-3)

        # Read back
        result = buffer.read()
        assert len(result) == 1600
        np.testing.assert_array_almost_equal(result, audio)

    def test_write_int16(self):
        """Test writing int16 audio."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)

        audio = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        buffer.write(audio)

        result = buffer.read()
        assert result.dtype == np.float32
        assert np.max(result) <= 1.0
        assert np.min(result) >= -1.0

    def test_overflow(self):
        """Test buffer overflow handling."""
        buffer = RingBuffer(max_duration_s=1.0, sample_rate=16000)

        # Write more than capacity
        for _ in range(20):
            audio = np.random.randn(1600).astype(np.float32)
            buffer.write(audio)

        # Should be capped at max
        assert len(buffer) <= buffer.max_samples
        assert buffer.duration_s <= 1.0

    def test_read_num_samples(self):
        """Test reading specific number of samples."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)

        audio = np.random.randn(3200).astype(np.float32)
        buffer.write(audio)

        result = buffer.read(num_samples=1600)
        assert len(result) == 1600
        # Should return last 1600 samples
        np.testing.assert_array_almost_equal(result, audio[-1600:])

    def test_read_and_clear(self):
        """Test reading and clearing."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)

        audio = np.random.randn(1600).astype(np.float32)
        buffer.write(audio)

        result = buffer.read_and_clear()
        assert len(result) == 1600
        assert buffer.is_empty
        assert len(buffer) == 0

    def test_clear(self):
        """Test clearing buffer."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)

        audio = np.random.randn(1600).astype(np.float32)
        buffer.write(audio)

        buffer.clear()
        assert buffer.is_empty
        assert len(buffer) == 0

    def test_empty_read(self):
        """Test reading from empty buffer."""
        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)

        result = buffer.read()
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading

        buffer = RingBuffer(max_duration_s=5.0, sample_rate=16000)
        errors = []

        def writer():
            try:
                for _ in range(100):
                    audio = np.random.randn(160).astype(np.float32)
                    buffer.write(audio)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = buffer.read()
                    _ = len(buffer)
                    _ = buffer.duration_s
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
