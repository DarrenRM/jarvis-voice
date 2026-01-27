"""Discord Voice Claude Bot - Entry Point."""

import asyncio
import logging
import queue
import signal
import sys
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings, Settings
from src.bot import ClaudeBot
from src.detection import VAD
from src.tts import TTSSynthesizer
from src.audio import SAMPLE_RATE_PROCESSING

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

    # Note: Wake word detectors are now created per-user in StreamingSink
    # for accurate speaker attribution

    # VAD
    logger.info("Loading VAD...")
    components["vad"] = VAD(
        threshold=settings.vad_sensitivity,
        sample_rate=SAMPLE_RATE_PROCESSING,
        speech_pad_ms=settings.speech_pad_ms,
        min_silence_duration_ms=settings.min_silence_ms,
    )

    # STT uses Groq Whisper (initialized in VoiceSession when API key is available)
    logger.info("STT will use Groq Whisper (initialized when joining voice)")

    # LLM client - NOT pre-created here because it needs the image callback
    # which requires access to voice_client (created in VoiceSession)
    logger.info(f"LLM client ({settings.llm_provider}) will be created when joining voice")

    # TTS (Inworld)
    logger.info("Loading TTS (Inworld)...")
    components["tts"] = TTSSynthesizer(
        model=settings.tts_model,
        device=settings.tts_device,
        voice=settings.tts_voice,
        api_key=settings.inworld_api_key,
        temperature=settings.tts_temperature,
        speed=settings.tts_speed,
    )

    logger.info("All components loaded!")
    return components


def setup_logging(level: str = "INFO") -> QueueListener:
    """Configure async logging to prevent event loop blocking.

    Uses QueueHandler + QueueListener so log calls are non-blocking.
    Actual I/O happens in a background thread.

    Args:
        level: Log level string

    Returns:
        QueueListener that must be stopped on shutdown
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

    # Create actual handlers (will run in background thread)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Queue-based handler (non-blocking - writes to memory queue)
    log_queue: queue.Queue = queue.Queue(-1)  # Unbounded
    queue_handler = QueueHandler(log_queue)

    # Root logger only gets the non-blocking queue handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(queue_handler)

    # Background thread processes queue and writes to console/file
    listener = QueueListener(
        log_queue,
        console_handler,
        file_handler,
        respect_handler_level=True,
    )
    listener.start()

    # Reduce noise from libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info(f"Logging to file: {log_file} (async queue-based)")

    return listener


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
    # Note: Wake word detectors are now created per-user in StreamingSink
    if components:
        bot = ClaudeBot(
            settings,
            vad=components.get("vad"),
            llm=components.get("llm"),
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

    # Setup async logging (returns listener for cleanup)
    log_listener = setup_logging(settings.log_level)
    check_assets()

    logger.info("Starting Discord Voice Claude Bot")
    logger.info(f"Wake word: {settings.wake_word_model.replace('_', ' ')}")
    logger.info(f"STT: Groq Whisper")
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
    finally:
        # Stop log listener to flush remaining messages
        log_listener.stop()


if __name__ == "__main__":
    main()
