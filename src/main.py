"""Discord Voice Claude Bot - Entry Point."""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings, Settings
from src.bot import ClaudeBot
from src.detection import VAD, WakeWordDetector
from src.stt import Transcriber
from src.llm import ClaudeClient
from src.tts import TTSSynthesizer

logger = logging.getLogger(__name__)


def preload_components(settings: Settings) -> dict:
    """Pre-load ML components for faster voice session startup.

    Args:
        settings: Application settings

    Returns:
        Dict of component instances
    """
    components = {}

    logger.info("Pre-loading ML components...")

    # Wake word detector
    logger.info("Loading wake word detector...")
    components["wake_detector"] = WakeWordDetector(
        model_name=settings.wake_word_model,
        threshold=settings.wake_word_confidence,
        sample_rate=16000,
    )

    # VAD
    logger.info("Loading VAD...")
    components["vad"] = VAD(
        threshold=settings.vad_sensitivity,
        sample_rate=16000,
        speech_pad_ms=settings.speech_pad_ms,
    )

    # Transcriber (only load local Whisper if using it as STT provider)
    if settings.stt_provider == "whisper":
        logger.info("Loading local Whisper transcriber...")
        components["transcriber"] = Transcriber(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    else:
        logger.info(f"Using {settings.stt_provider} for STT - skipping local Whisper model")

    # Claude client
    logger.info(f"Setting up Claude client (model: {settings.claude_model})...")
    components["claude"] = ClaudeClient(model=settings.claude_model)

    # TTS (Inworld)
    logger.info("Loading TTS (Inworld)...")
    components["tts"] = TTSSynthesizer(
        model=settings.tts_model,
        device=settings.tts_device,
        voice=settings.tts_voice,
        api_key=settings.inworld_api_key,
    )

    logger.info("All components loaded!")
    return components


def setup_logging(level: str = "INFO") -> None:
    """Configure logging to both console and file.

    Args:
        level: Log level string
    """
    log_level = getattr(logging, level.upper())
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create logs directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "bot.log"

    # Clear previous log file for fresh start
    if log_file.exists():
        log_file.unlink()

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info(f"Logging to file: {log_file}")


def check_assets() -> None:
    """Ensure required assets exist."""
    assets_dir = Path(__file__).parent.parent / "assets" / "audio"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ack_file = assets_dir / "yes.wav"
    if not ack_file.exists():
        logging.warning(
            f"Acknowledgment audio not found: {ack_file}\n"
            "TTS will be used for acknowledgments (slower).\n"
            "Record a 'Yes?' audio clip and save it as assets/audio/yes.wav"
        )


async def run_bot(settings: Settings, components: Optional[dict] = None) -> None:
    """Run bot with graceful shutdown handling.

    Args:
        settings: Application settings
        components: Pre-loaded ML components (optional)
    """
    # Create bot with pre-loaded components
    if components:
        bot = ClaudeBot(
            settings,
            wake_detector=components.get("wake_detector"),
            vad=components.get("vad"),
            transcriber=components.get("transcriber"),
            claude=components.get("claude"),
            tts=components.get("tts"),
        )
    else:
        bot = ClaudeBot(settings)

    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()

    # On Windows, only SIGINT works reliably
    if sys.platform == "win32":
        # For Windows, use signal.signal for SIGINT
        signal.signal(signal.SIGINT, lambda s, f: shutdown_event.set())
        signal.signal(signal.SIGTERM, lambda s, f: shutdown_event.set())
        # SIGBREAK is sent by taskkill on Windows
        try:
            signal.signal(signal.SIGBREAK, lambda s, f: shutdown_event.set())
        except AttributeError:
            pass  # SIGBREAK not available on all platforms
    else:
        # Unix-like systems
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    # Start the bot
    async with bot:
        await bot.login(settings.discord_token)

        # Create task for bot connection
        connect_task = asyncio.create_task(bot.connect())

        # Wait for either shutdown signal or bot disconnect
        shutdown_wait = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [connect_task, shutdown_wait],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # If shutdown was requested, close gracefully
        if shutdown_event.is_set():
            logger.info("Closing bot gracefully...")
            # bot.close() is called by async context manager exit

    logger.info("Bot shutdown complete")


def main() -> None:
    """Main entry point."""
    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        print("Make sure .env file exists with DISCORD_TOKEN")
        sys.exit(1)

    # Setup
    setup_logging(settings.log_level)
    check_assets()

    logger.info("Starting Discord Voice Claude Bot")
    logger.info(f"Wake word: {settings.wake_word_model.replace('_', ' ')}")
    logger.info(f"Whisper model: {settings.whisper_model}")
    logger.info(f"TTS voice: {settings.tts_voice}")

    # Pre-load ML components for faster voice session startup
    try:
        components = preload_components(settings)
    except Exception as e:
        logger.error(f"Failed to load components: {e}")
        logger.info("Components will be loaded on-demand when joining voice")
        components = None

    logger.info("Press Ctrl+C to shutdown gracefully")

    try:
        asyncio.run(run_bot(settings, components))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
