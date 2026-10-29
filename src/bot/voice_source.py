"""Discord audio sources for TTS playback.

Converts TTS output (24kHz mono) to Discord format (48kHz stereo).
"""

import io
import logging

import discord
import numpy as np

from ..audio import resample_24k_mono_to_48k_stereo

logger = logging.getLogger(__name__)


# Discord expects 20ms frames at 48kHz stereo s16le
# 48000 * 0.02 = 960 samples per channel
# 960 * 2 channels = 1920 samples
# 1920 * 2 bytes (16-bit) = 3840 bytes per frame
FRAME_SIZE = 3840


class TTSAudioSource(discord.AudioSource):
    """Audio source for playing TTS output through Discord.

    Takes Inworld TTS output (24kHz mono float32) and converts it to
    Discord's expected format (48kHz stereo s16le PCM).
    """

    def __init__(self, audio_24k: np.ndarray):
        """Initialize TTS audio source.

        Args:
            audio_24k: Audio as float32 numpy array (24kHz mono, -1 to 1)
        """
        super().__init__()

        # Convert to Discord format
        audio_bytes = resample_24k_mono_to_48k_stereo(audio_24k)

        # Create byte stream for reading
        self._stream = io.BytesIO(audio_bytes)
        self._total_bytes = len(audio_bytes)

        logger.debug(
            f"TTSAudioSource: {len(audio_24k)} samples at 24kHz "
            f"-> {self._total_bytes} bytes at 48kHz stereo"
        )

    def read(self) -> bytes:
        """Read 20ms of audio.

        Returns:
            3840 bytes of PCM data, or empty bytes if finished
        """
        data = self._stream.read(FRAME_SIZE)

        # Pad with silence if we have partial frame at the end
        if 0 < len(data) < FRAME_SIZE:
            data = data + b"\x00" * (FRAME_SIZE - len(data))

        return data

    def is_opus(self) -> bool:
        """Whether this source is opus encoded."""
        return False

    def cleanup(self) -> None:
        """Clean up resources."""
        self._stream.close()

    @property
    def progress(self) -> float:
        """Playback progress as percentage (0-100)."""
        if self._total_bytes == 0:
            return 100.0
        return (self._stream.tell() / self._total_bytes) * 100


class SilenceSource(discord.AudioSource):
    """Audio source that generates silence.

    Useful for keeping the voice connection active or
    as a placeholder between TTS playbacks.
    """

    def __init__(self, duration_ms: int = 100):
        """Initialize silence source.

        Args:
            duration_ms: Duration of silence in milliseconds
        """
        super().__init__()

        # Calculate number of frames needed
        frames_needed = max(1, duration_ms // 20)
        self._frames_remaining = frames_needed
        self._silence_frame = b"\x00" * FRAME_SIZE

    def read(self) -> bytes:
        """Read 20ms of silence.

        Returns:
            3840 bytes of silence, or empty bytes if finished
        """
        if self._frames_remaining > 0:
            self._frames_remaining -= 1
            return self._silence_frame
        return b""

    def is_opus(self) -> bool:
        """Whether this source is opus encoded."""
        return False

    def cleanup(self) -> None:
        """Clean up resources."""
        pass
