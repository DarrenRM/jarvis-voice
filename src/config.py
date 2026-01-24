"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Discord
    discord_token: str = Field(..., alias="DISCORD_TOKEN")
    discord_guild_id: Optional[str] = Field(default=None, alias="DISCORD_GUILD_ID")

    # Wake Word Detection
    wake_word_method: str = Field(default="openwakeword", alias="WAKE_WORD_METHOD")
    wake_word_confidence: float = Field(default=0.5, alias="WAKE_WORD_CONFIDENCE")
    wake_word_model: str = Field(default="hey_jarvis", alias="WAKE_WORD_MODEL")
    wake_word_phrase: str = Field(default="hey claude", alias="WAKE_WORD_PHRASE")

    # STT (Whisper)
    whisper_model: str = Field(default="medium", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cuda", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="float16", alias="WHISPER_COMPUTE_TYPE")

    # TTS (Inworld)
    inworld_api_key: Optional[str] = Field(default=None, alias="INWORLD_API_KEY")
    tts_model: str = Field(default="inworld-tts-1.5-max", alias="TTS_MODEL")
    tts_device: str = Field(default="cuda:0", alias="TTS_DEVICE")
    tts_voice: str = Field(default="alex", alias="TTS_VOICE")

    # Claude LLM
    claude_model: str = Field(default="haiku", alias="CLAUDE_MODEL")  # haiku, sonnet, opus

    # Latency Settings
    vad_sensitivity: float = Field(default=0.5, alias="VAD_SENSITIVITY")
    speech_pad_ms: int = Field(default=300, alias="SPEECH_PAD_MS")
    max_question_duration: float = Field(default=30.0, alias="MAX_QUESTION_DURATION")
    claude_timeout: float = Field(default=30.0, alias="CLAUDE_TIMEOUT")

    # Audio
    sample_rate: int = Field(default=16000, alias="SAMPLE_RATE")
    chunk_duration_ms: int = Field(default=30, alias="CHUNK_DURATION_MS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_latency_metrics: bool = Field(default=True, alias="LOG_LATENCY_METRICS")

    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in samples."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def bytes_per_chunk(self) -> int:
        """Calculate bytes per chunk (16-bit mono)."""
        return self.chunk_size * 2


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
