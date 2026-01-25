"""Discord bot module."""

from .discord_bot import ClaudeBot
from .discord_service import DiscordService
from .speaker_manager import SpeakerManager, SpeakerInfo
from .voice_session import VoiceSession
from .voice_sink import StreamingSink
from .voice_source import TTSAudioSource, SilenceSource

__all__ = [
    "ClaudeBot",
    "DiscordService",
    "SpeakerManager",
    "SpeakerInfo",
    "VoiceSession",
    "StreamingSink",
    "TTSAudioSource",
    "SilenceSource",
]
