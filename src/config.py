"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Dict, Optional


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

    # Auto-join voice channel on startup
    auto_join_voice: bool = Field(default=True, alias="AUTO_JOIN_VOICE")
    auto_join_channel: Optional[str] = Field(default=None, alias="AUTO_JOIN_CHANNEL")

    # Wake Word Detection
    wake_word_method: str = Field(default="openwakeword", alias="WAKE_WORD_METHOD")
    wake_word_confidence: float = Field(default=0.5, alias="WAKE_WORD_CONFIDENCE")
    wake_word_model: str = Field(default="hey_jarvis", alias="WAKE_WORD_MODEL")
    wake_word_phrase: str = Field(default="hey claude", alias="WAKE_WORD_PHRASE")

    # STT - Groq Whisper only
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")

    # TTS (Inworld)
    inworld_api_key: Optional[str] = Field(default=None, alias="INWORLD_API_KEY")
    tts_model: str = Field(default="inworld-tts-1-max", alias="TTS_MODEL")
    tts_device: str = Field(default="cuda:0", alias="TTS_DEVICE")
    tts_voice: str = Field(default="alex", alias="TTS_VOICE")
    tts_temperature: float = Field(default=1.1, alias="TTS_TEMPERATURE")  # 0.0-2.0, higher = more expressive
    tts_speed: float = Field(default=1.0, alias="TTS_SPEED")  # 0.5-1.5, talking speed multiplier

    # LLM (Cerebras, OpenAI-compatible API)
    cerebras_api_key: Optional[str] = Field(default=None, alias="CEREBRAS_API_KEY")
    cerebras_model: str = Field(default="qwen-3.8-27b", alias="CEREBRAS_MODEL")
    # none | low | medium | high. Reasoning never reaches TTS or history.
    cerebras_reasoning_effort: str = Field(default="high", alias="CEREBRAS_REASONING_EFFORT")
    # Output budget per model turn; reasoning tokens count against it
    llm_max_completion_tokens: int = Field(default=16384, alias="LLM_MAX_COMPLETION_TOKENS")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    # Voice conversation history kept per session (oldest whole exchanges dropped first)
    llm_history_max_messages: int = Field(default=60, alias="LLM_HISTORY_MAX_MESSAGES")

    # Web lookup: ddgs metasearch + local page reader (no keys, no hosted services)
    web_tools_enabled: bool = Field(default=True, alias="WEB_TOOLS_ENABLED")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")
    # Text handed to the model per page is cut at this length
    web_extract_max_chars: int = Field(default=15000, alias="WEB_EXTRACT_MAX_CHARS")

    # Deep research via OpenAI (Codex/ChatGPT subscription sign-in). Not wired up yet.
    deep_research_enabled: bool = Field(default=False, alias="DEEP_RESEARCH_ENABLED")

    # Image Generation (Nano Banana / Gemini)
    image_generation_enabled: bool = Field(default=False, alias="IMAGE_GENERATION_ENABLED")
    nanobanana_api_key: Optional[str] = Field(default=None, alias="NANOBANANA_API_KEY")
    image_channel_name: str = Field(default="general", alias="IMAGE_CHANNEL_NAME")

    # Latency Settings
    vad_sensitivity: float = Field(default=0.5, alias="VAD_SENSITIVITY")
    speech_pad_ms: int = Field(default=300, alias="SPEECH_PAD_MS")
    min_silence_ms: int = Field(default=500, alias="MIN_SILENCE_MS")  # Silence to trigger speech end
    max_question_duration: float = Field(default=30.0, alias="MAX_QUESTION_DURATION")

    # Speaking Detection (multi-user awareness)
    speaking_detection_enabled: bool = Field(default=False, alias="SPEAKING_DETECTION_ENABLED")
    speaking_energy_threshold: float = Field(default=0.01, alias="SPEAKING_ENERGY_THRESHOLD")
    speaking_silence_debounce_ms: int = Field(default=3000, alias="SPEAKING_SILENCE_DEBOUNCE_MS")

    # Speaker Isolation (filter audio from non-wake-word speakers)
    speaker_isolation_enabled: bool = Field(default=True, alias="SPEAKER_ISOLATION_ENABLED")

    # Voice Channel Greetings
    greeting_enabled: bool = Field(default=True, alias="GREETING_ENABLED")
    greeting_timeout_seconds: float = Field(default=10.0, alias="GREETING_TIMEOUT_SECONDS")
    greeting_cooldown_seconds: float = Field(default=300.0, alias="GREETING_COOLDOWN_SECONDS")  # 5 minutes

    # Keep-alive (prevent Discord disconnect after ~1 hour idle)
    keep_alive_enabled: bool = Field(default=True, alias="KEEP_ALIVE_ENABLED")
    keep_alive_interval_seconds: float = Field(default=3000.0, alias="KEEP_ALIVE_INTERVAL_SECONDS")  # 50 minutes
    keep_alive_prompt: str = Field(
        default=(
            "You've been sitting in silence for almost an hour. "
            "Discord will disconnect you if you don't say something soon. "
            "Generate a brief, slightly annoyed or sarcastic comment about being legally required to speak "
            "or Discord will kick you out. Keep it to 1-2 sentences max."
        ),
        alias="KEEP_ALIVE_PROMPT"
    )

    # Dynamic Thinking Phrases (context-aware acknowledgments)
    dynamic_thinking_enabled: bool = Field(default=False, alias="DYNAMIC_THINKING_ENABLED")

    # User name mappings (Discord user ID -> friendly name)
    # Can be set via USER_NAMES env var as JSON: '{"123456": "Name"}'
    # Users not listed fall back to their Discord display name.
    user_names: Dict[int, str] = Field(default_factory=dict, alias="USER_NAMES")

    # Optional per-user roasts the assistant may rarely work into answers.
    # Set via USER_ROASTS env var as JSON: '{"Name": "is always late."}'
    user_roasts: Dict[str, str] = Field(default_factory=dict, alias="USER_ROASTS")

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
