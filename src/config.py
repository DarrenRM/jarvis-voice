"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
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
    wake_word_confidence: float = Field(default=0.5, alias="WAKE_WORD_CONFIDENCE")
    wake_word_model: str = Field(default="hey_jarvis", alias="WAKE_WORD_MODEL")

    # STT - Groq Whisper only
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")

    # TTS (Inworld)
    inworld_api_key: Optional[str] = Field(default=None, alias="INWORLD_API_KEY")
    tts_model: str = Field(default="inworld-tts-2-flash", alias="TTS_MODEL")
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

    # Web lookup. Backend "auto" uses TinyFish (hosted search + page reading, free tier)
    # when TINYFISH_API_KEY is set, else local ddgs metasearch + Trafilatura reader.
    # "local" forces the local path. TinyFish failures fall back to local per request.
    web_tools_enabled: bool = Field(default=True, alias="WEB_TOOLS_ENABLED")
    web_backend: str = Field(default="auto", alias="WEB_BACKEND")
    tinyfish_api_key: Optional[str] = Field(default=None, alias="TINYFISH_API_KEY")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")
    # Text handed to the model per page is cut at this length
    web_extract_max_chars: int = Field(default=15000, alias="WEB_EXTRACT_MAX_CHARS")

    # Image Generation (Nano Banana / Gemini)
    image_generation_enabled: bool = Field(default=False, alias="IMAGE_GENERATION_ENABLED")
    nanobanana_api_key: Optional[str] = Field(default=None, alias="NANOBANANA_API_KEY")
    image_channel_name: str = Field(default="general", alias="IMAGE_CHANNEL_NAME")

    # Latency Settings
    vad_sensitivity: float = Field(default=0.5, alias="VAD_SENSITIVITY")
    speech_pad_ms: int = Field(default=300, alias="SPEECH_PAD_MS")
    # Quiet that ends a question (VAD silence, or Discord no longer sending audio). Long
    # enough for "um... so..." pauses; speech that comes after it still isn't lost
    # (VoiceSession starts over when the speaker carries on before the answer plays).
    min_silence_ms: int = Field(default=2000, alias="MIN_SILENCE_MS")
    max_question_duration: float = Field(default=30.0, alias="MAX_QUESTION_DURATION")

    # Follow-up without the wake word: only from the person the bot just answered, and only
    # if a quick check says they're talking to the bot. After the bot asks a question, a
    # reply counts; after a plain answer, only something that needs a reply does (a new
    # question or request, or a correction; not "okay" or "thanks").
    followup_enabled: bool = Field(default=True, alias="FOLLOWUP_ENABLED")
    followup_window_seconds: float = Field(default=10.0, alias="FOLLOWUP_WINDOW_SECONDS")
    followup_after_answer: bool = Field(default=True, alias="FOLLOWUP_AFTER_ANSWER")
    followup_after_answer_seconds: float = Field(
        default=4.0, alias="FOLLOWUP_AFTER_ANSWER_SECONDS"
    )
    # Follow-up windows only count down while the room is quiet. When someone else stops
    # talking, the speaker gets at least this long to start their reply...
    followup_grace_seconds: float = Field(default=3.0, alias="FOLLOWUP_GRACE_SECONDS")
    # ...and a window never stays open longer than this
    followup_max_wait_seconds: float = Field(default=30.0, alias="FOLLOWUP_MAX_WAIT_SECONDS")
    # TypeSafe Jev makes that check when its key is set (calibrated, ~0.1s); else Qwen
    typesafe_api_key: Optional[str] = Field(default=None, alias="TYPESAFE_API_KEY")
    # Reference answers (recipes, steps, stats) are also posted to IMAGE_CHANNEL_NAME:
    # Jev decides, Qwen writes the post. Needs TYPESAFE_API_KEY and CEREBRAS_API_KEY.
    chat_posts_enabled: bool = Field(default=True, alias="CHAT_POSTS_ENABLED")

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

    # Live monitor (src/monitor): posts problems seen in real voice turns to a text
    # channel with check/cross reactions. The owner's check means "something went wrong
    # here" (not that the post's diagnosis is right) and adds the turn to
    # MONITOR_APPROVED_FILE; a cross means "no problem, or it didn't disrupt the flow"
    # and adds it to MONITOR_DISMISSED_FILE (fix only if it costs nothing). Off by default.
    monitor_enabled: bool = Field(default=False, alias="MONITOR_ENABLED")
    monitor_channel_name: str = Field(default="bot-monitor", alias="MONITOR_CHANNEL_NAME")
    # Only this Discord user's reactions count (empty = nobody can approve)
    monitor_owner_id: Optional[int] = Field(default=None, alias="MONITOR_OWNER_ID")
    monitor_approved_file: str = Field(
        default="docs/approved-fixes.md", alias="MONITOR_APPROVED_FILE"
    )
    monitor_dismissed_file: str = Field(
        default="docs/dismissed-findings.md", alias="MONITOR_DISMISSED_FILE"
    )
    # Keep real audio for the IRL test set (src/monitor/recordings.py): every question
    # as Whisper got it, and clips around wake word hits and near misses. Local only.
    monitor_save_audio: bool = Field(default=False, alias="MONITOR_SAVE_AUDIO")
    monitor_audio_dir: str = Field(default="recordings", alias="MONITOR_AUDIO_DIR")
    monitor_audio_max_mb: float = Field(default=1000.0, alias="MONITOR_AUDIO_MAX_MB")
    # Near misses scoring below this aren't kept (everyday speech scores ~0.05-0.2)
    monitor_wake_clip_min_score: float = Field(default=0.2, alias="MONITOR_WAKE_CLIP_MIN_SCORE")
    # Re-transcribe the bot's answer audio with Groq Whisper to catch TTS that doesn't
    # match its text (one extra Groq call per answered turn)
    monitor_retranscribe: bool = Field(default=False, alias="MONITOR_RETRANSCRIBE")
    # Same problem again within this many minutes: the existing post's count goes up
    monitor_dedupe_minutes: float = Field(default=30.0, alias="MONITOR_DEDUPE_MINUTES")
    # Thresholds in seconds. Turn timings count from the end of the MIN_SILENCE_MS pause,
    # not the speaker's last word (good turns: opening phrase ~0.1-0.2s after it, answer
    # ~4s after it, ~0.3s between sentences)
    monitor_slow_first_audio_seconds: float = Field(
        default=3.0, alias="MONITOR_SLOW_FIRST_AUDIO_SECONDS"
    )
    monitor_slow_answer_seconds: float = Field(default=8.0, alias="MONITOR_SLOW_ANSWER_SECONDS")
    monitor_slow_answer_web_seconds: float = Field(
        default=20.0, alias="MONITOR_SLOW_ANSWER_WEB_SECONDS"
    )
    monitor_long_gap_seconds: float = Field(default=2.5, alias="MONITOR_LONG_GAP_SECONDS")
    monitor_hang_seconds: float = Field(default=30.0, alias="MONITOR_HANG_SECONDS")

    @field_validator("monitor_owner_id", mode="before")
    @classmethod
    def _blank_owner_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
