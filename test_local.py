"""Local testing for voice pipeline without Discord."""

import asyncio
import logging
import random
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.detection import VAD, WakeWordDetector
from src.stt import Transcriber
from src.llm import ClaudeClient
from src.tts import TTSSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Enable debug for our modules (helps debug Inworld/Claude issues)
logging.getLogger("src.tts.synthesizer").setLevel(logging.DEBUG)
logging.getLogger("src.llm.claude_client").setLevel(logging.DEBUG)


class LocalVoiceTest:
    """Test voice pipeline locally with microphone/speakers."""

    def __init__(self):
        self.settings = get_settings()
        self.sample_rate = self.settings.sample_rate
        self.running = False

        # Components
        self.wake_detector = None
        self.vad = None
        self.transcriber = None
        self.claude = None
        self.tts = None

        # State
        self.state = "idle"  # idle, listening, processing
        self.audio_buffer = []
        self.wake_detected = asyncio.Event()
        self._loop = None  # Event loop reference for thread-safe callback
        self._speech_started = False  # Track if user started speaking
        self._ack_task = None  # Track ack task for cancellation
        self._followup_window = 6.0  # Seconds to wait for follow-up without wake word

        # Thinking phrases (randomly selected)
        self._thinking_phrases = [
            "One second...",
            "One moment...",
            "Let me check...",
            "Hmm, let me see...",
            "Just a moment...",
        ]

    def setup(self):
        """Initialize all components."""
        logger.info("Setting up components...")

        # Wake word detector
        logger.info("Loading wake word detector...")
        self.wake_detector = WakeWordDetector(
            model_name=self.settings.wake_word_model,
            threshold=self.settings.wake_word_confidence,
            sample_rate=self.sample_rate,
        )
        self.wake_detector.set_callback(self._on_wake_word)

        # VAD
        logger.info("Loading VAD...")
        self.vad = VAD(
            threshold=self.settings.vad_sensitivity,
            sample_rate=self.sample_rate,
            speech_pad_ms=self.settings.speech_pad_ms,
        )

        # Transcriber
        logger.info("Loading Whisper...")
        self.transcriber = Transcriber(
            model_size=self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
        )

        # Claude client
        logger.info("Setting up Claude client...")
        self.claude = ClaudeClient()

        # TTS (Inworld)
        logger.info("Loading TTS...")
        self.tts = TTSSynthesizer(
            model=self.settings.tts_model,
            device=self.settings.tts_device,
            voice=self.settings.tts_voice,
            api_key=self.settings.inworld_api_key,
        )

        logger.info("All components ready!")

    def _on_wake_word(self):
        """Called when wake word detected (from audio thread)."""
        if self.state == "idle" and self._loop:
            self.state = "waking"  # Prevent multiple triggers
            # Thread-safe event set
            self._loop.call_soon_threadsafe(self.wake_detected.set)

    def _audio_callback(self, indata, frames, time, status):
        """Called for each audio chunk from microphone."""
        if status:
            logger.warning(f"Audio status: {status}")

        # Convert to float32
        audio = indata[:, 0].astype(np.float32)

        if self.state == "idle":
            # Check for wake word
            self.wake_detector.process(audio)

        elif self.state == "listening":
            # Buffer audio and check for end of speech
            self.audio_buffer.append(audio.copy())
            endpoint = self.vad.detect_endpoint(audio)

            # Cancel ack if user started speaking
            if endpoint == "start" and not self._speech_started:
                self._speech_started = True
                if self._ack_task and not self._ack_task.done():
                    self._ack_task.cancel()
                    logger.debug("Cancelled ack - user started speaking")

            if endpoint == "end":
                self.state = "processing"

    async def play_audio(self, audio: np.ndarray, sample_rate: int = 24000):
        """Play audio through speakers."""
        if len(audio) == 0:
            return

        # Play synchronously in executor to not block
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: sd.play(audio, sample_rate, blocking=True)
        )

    async def _play_ack(self):
        """Play acknowledgment in background (cancelled if user starts speaking)."""
        try:
            # Synthesize first
            ack_audio = await self.tts.synthesize("Yes?")

            # Check if user started speaking during synthesis
            if self._speech_started:
                logger.debug("Skipping ack playback - user already speaking")
                return

            await self.play_audio(ack_audio, self.tts.sample_rate)
        except asyncio.CancelledError:
            logger.debug("Ack playback cancelled")
        except Exception as e:
            logger.warning(f"Failed to play acknowledgment: {e}")

    async def play_file(self, path: Path):
        """Play audio file."""
        audio, sr = sf.read(path)
        await self.play_audio(audio, sr)

    async def run(self):
        """Main loop."""
        self.setup()
        self.running = True
        self._loop = asyncio.get_running_loop()

        logger.info("")
        logger.info("=" * 50)
        logger.info("LOCAL VOICE TEST READY")
        logger.info(f"Say '{self.settings.wake_word_model.replace('_', ' ')}' to activate")
        logger.info("Press Ctrl+C to exit")
        logger.info("=" * 50)
        logger.info("")

        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(self.sample_rate * 0.03),  # 30ms chunks
            callback=self._audio_callback,
        )

        try:
            with stream:
                while self.running:
                    # Wait for wake word
                    self.state = "idle"
                    self.wake_detected.clear()
                    self.wake_detector.reset()  # Clear any buffered audio
                    logger.info("Listening for wake word...")

                    await self.wake_detected.wait()

                    logger.info("Wake word detected!")

                    # Start listening IMMEDIATELY
                    self.state = "listening"
                    self.audio_buffer = []
                    self.vad.reset()
                    self._speech_started = False  # Reset speech detection
                    logger.info("Listening... (speak now)")

                    # Play ack in background (will be cancelled if user starts speaking)
                    self._ack_task = asyncio.create_task(self._play_ack())

                    # Wait for speech to end (with timeout)
                    timeout = 10.0  # seconds
                    start_time = asyncio.get_event_loop().time()
                    while self.state == "listening":
                        await asyncio.sleep(0.1)
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            logger.warning("Listening timeout")
                            break

                    if not self.audio_buffer:
                        logger.info("No audio captured")
                        continue

                    # Transcribe
                    logger.info("Transcribing...")
                    audio = np.concatenate(self.audio_buffer)

                    # Run transcription in executor
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(
                        None,
                        self.transcriber.transcribe,
                        audio,
                    )

                    if not text.strip():
                        logger.info("No speech detected")
                        continue

                    logger.info(f"You said: {text}")

                    # Play thinking phrase
                    phrase = random.choice(self._thinking_phrases)
                    confirm_audio = await self.tts.synthesize(phrase)
                    await self.play_audio(confirm_audio, self.tts.sample_rate)

                    # Get Claude response
                    logger.info("Getting Claude response...")
                    response_text = ""
                    async for chunk in self.claude.chat_stream(text):
                        response_text += chunk

                    logger.info(f"Claude: {response_text}")

                    # Synthesize and play response
                    logger.info("Synthesizing response...")
                    response_audio = await self.tts.synthesize(response_text)
                    await self.play_audio(response_audio, self.tts.sample_rate)

                    # Follow-up window - listen without wake word
                    while True:
                        logger.info(f"Listening for follow-up ({self._followup_window}s window)...")
                        self.state = "listening"
                        self.audio_buffer = []
                        self.vad.reset()
                        self._speech_started = False

                        # Wait for speech with shorter timeout
                        followup_start = asyncio.get_event_loop().time()
                        speech_detected = False

                        while self.state == "listening":
                            await asyncio.sleep(0.1)
                            elapsed = asyncio.get_event_loop().time() - followup_start

                            # If user started speaking, give them full time to finish
                            if self._speech_started:
                                speech_detected = True
                                # Reset timeout to allow full utterance
                                if elapsed > self._followup_window:
                                    continue  # Keep listening while they're speaking

                            if elapsed > self._followup_window and not self._speech_started:
                                # No speech in follow-up window
                                break

                        if not speech_detected or not self.audio_buffer:
                            logger.info("No follow-up detected, returning to wake word mode")
                            break

                        # Process follow-up
                        logger.info("Processing follow-up...")
                        audio = np.concatenate(self.audio_buffer)
                        loop = asyncio.get_running_loop()
                        text = await loop.run_in_executor(
                            None,
                            self.transcriber.transcribe,
                            audio,
                        )

                        if not text.strip():
                            logger.info("No speech in follow-up")
                            break

                        logger.info(f"You said: {text}")

                        # Play thinking phrase for follow-ups too
                        phrase = random.choice(self._thinking_phrases)
                        confirm_audio = await self.tts.synthesize(phrase)
                        await self.play_audio(confirm_audio, self.tts.sample_rate)

                        logger.info("Getting Claude response...")
                        response_text = ""
                        async for chunk in self.claude.chat_stream(text):
                            response_text += chunk

                        logger.info(f"Claude: {response_text}")

                        response_audio = await self.tts.synthesize(response_text)
                        await self.play_audio(response_audio, self.tts.sample_rate)
                        # Loop continues for another follow-up

                    logger.info("Ready for next wake word.")
                    logger.info("")

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False


async def main():
    """Entry point."""
    test = LocalVoiceTest()
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
