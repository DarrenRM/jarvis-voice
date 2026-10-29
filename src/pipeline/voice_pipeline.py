"""Full voice processing pipeline with thinking phrase support.

Pipeline: Audio -> Transcription -> Response -> TTS -> Playback

Features:
- Thinking phrase runs in parallel with LLM
- Dynamic thinking phrases (context-aware based on request)
- Playback waits for thinking before first sentence
- Cancellation at any stage
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

import numpy as np

from .base import PipelineContext
from .response_stage import ResponseStage
from .transcription_stage import TranscriptionResult, TranscriptionStage
from .tts_stage import TTSStage, TTSSynthesizerProtocol

if TYPE_CHECKING:
    from ..session.thinking_phrase_generator import ThinkingPhraseGenerator

logger = logging.getLogger(__name__)


class OpeningPhrase:
    """A short phrase ("Sure, Sam...") played as soon as speech ends.

    Synthesis and playback start immediately in a background task, so the
    phrase is heard while transcription is still running instead of after it.
    """

    def __init__(
        self,
        synthesize: Callable[[], Awaitable[np.ndarray]],
        play_audio: Callable[[np.ndarray], Awaitable[None]],
        should_play: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
        """Start the phrase.

        Args:
            synthesize: Returns the phrase audio
            play_audio: Async function to play audio
            should_play: Optional check (e.g. room is quiet); False skips the phrase
        """
        self.playing = False
        # Set when should_play said no (someone is still talking)
        self.skipped = False
        self._task = asyncio.create_task(self._run(synthesize, play_audio, should_play))

    async def _run(self, synthesize, play_audio, should_play) -> None:
        try:
            if should_play is not None and not await should_play():
                self.skipped = True
                logger.info("Pipeline: Skipping opening phrase - someone still speaking")
                return
            audio = await synthesize()
            if audio is None or len(audio) == 0:
                return
            self.playing = True
            await play_audio(audio)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Pipeline: Opening phrase failed: {e}")

    async def wait(self) -> None:
        """Wait until the phrase has played (or was skipped)."""
        await self._task

    def cancel(self) -> None:
        self._task.cancel()

    def cancel_if_not_playing(self) -> bool:
        """Cancel unless already audible. Returns True if it was cancelled."""
        if self.playing or self._task.done():
            return False
        self._task.cancel()
        return True


class VoicePipeline:
    """Full voice processing pipeline with thinking phrase support.

    Composes TranscriptionStage, ResponseStage, and TTSStage into
    a complete voice processing pipeline.

    Features:
    - Thinking phrase runs in parallel with LLM (doesn't block)
    - Playback waits for thinking phrase before first sentence
    - Cancellation propagates through all stages
    - Wake-word-only detection triggers follow-up flow

    Example:
        pipeline = VoicePipeline(transcription, response, tts)

        result = await pipeline.process(
            audio=audio_data,
            context=PipelineContext(speaker_name="Alice"),
            play_audio=player.play,
            wait_for_silence=silence_detector.wait,
            thinking_phrase="Let me think about that..."
        )

        if result == "wake_word_only":
            # Handle wake-word-only case
            pass
    """

    def __init__(
        self,
        transcription: TranscriptionStage,
        response: ResponseStage,
        tts: TTSStage,
    ):
        """Initialize voice pipeline.

        Args:
            transcription: Stage for STT and wake word stripping
            response: Stage for LLM response generation
            tts: Stage for parallel TTS synthesis
        """
        self._transcription = transcription
        self._response = response
        self._tts = tts
        self._cancelled = False
        self._thinking_task: Optional[asyncio.Task] = None
        self._opening: Optional[OpeningPhrase] = None
        # Optional turn monitor (src.monitor.TurnRecorder); only notified, never awaited
        self.observer = None
        # Optional src.llm.chat_post.ChatPoster: reference answers also go to chat
        self.poster = None
        # The reply itself has started playing (not just the opening/thinking phrase)
        self.answer_started = False

    async def process(
        self,
        audio: np.ndarray,
        context: PipelineContext,
        play_audio: Callable[[np.ndarray], Awaitable[None]],
        wait_for_silence: Callable[[], Awaitable[bool]],
        thinking_phrase: Optional[str] = None,
        thinking_audio: Optional[np.ndarray] = None,
        thinking_generator: Optional["ThinkingPhraseGenerator"] = None,
        tts_synthesizer: Optional[TTSSynthesizerProtocol] = None,
        affirmation_audio: Optional[np.ndarray] = None,
        opening: Optional[OpeningPhrase] = None,
        accept: Optional[Callable[[TranscriptionResult], Awaitable[bool]]] = None,
    ) -> Optional[str]:
        """Run full voice pipeline.

        Args:
            audio: Input audio from voice activity detection
            context: Pipeline context with speaker info
            play_audio: Async function to play audio chunks
            wait_for_silence: Async function that waits for silence
            thinking_phrase: Optional text for thinking phrase (for logging, static mode)
            thinking_audio: Optional pre-synthesized thinking phrase audio (static mode)
            thinking_generator: Optional generator for dynamic context-aware phrases
            tts_synthesizer: Optional TTS for synthesizing dynamic phrases
            affirmation_audio: Pre-synthesized affirmation (dynamic mode, plays immediately)
            opening: Phrase already started at speech end (affirmation in dynamic
                mode, the thinking phrase in static mode). Cancelled on an empty
                transcription if not yet audible; left to the caller on wake-word-only.
            accept: Checked after transcription; False drops the turn without a reply
                (follow-ups without the wake word that weren't meant for the bot).

        Returns:
            None normally
            "wake_word_only" if only wake word detected
            "cancelled" if cancelled during processing
            "empty" if no speech detected
            "not_addressed" if accept() turned the speech down
        """
        self._cancelled = False
        self._thinking_task = None
        self._opening = opening
        self.answer_started = False

        # Stage 1: Transcription
        logger.debug("Pipeline: Starting transcription")
        try:
            transcription_result = await self._transcription.process(audio, context)
        except asyncio.CancelledError:
            logger.debug("Pipeline: Transcription cancelled")
            if opening is not None:
                opening.cancel()
            return "cancelled"
        except Exception:
            if opening is not None:
                opening.cancel_if_not_playing()
            raise

        if self.observer is not None:
            self.observer.transcribed(transcription_result.raw_text, transcription_result.text)

        # Handle special cases
        if transcription_result.is_empty:
            logger.debug("Pipeline: Empty transcription, skipping")
            if opening is not None:
                opening.cancel_if_not_playing()
            return "empty"

        if transcription_result.wake_word_only:
            logger.debug("Pipeline: Wake word only detected")
            return "wake_word_only"

        text = transcription_result.text
        logger.info(f"Pipeline: Transcribed: {text[:100]}...")

        if accept is not None and not await accept(transcription_result):
            if opening is not None:
                opening.cancel_if_not_playing()
            return "not_addressed"

        if self._cancelled:
            return "cancelled"

        # Stage 2: Start thinking phrase in parallel with LLM
        thinking_done = asyncio.Event()

        if thinking_generator and tts_synthesizer:
            # Dynamic mode: play affirmation immediately, then generate context-aware phrase
            self._thinking_task = asyncio.create_task(
                self._generate_and_play_dynamic_thinking(
                    thinking_generator,
                    tts_synthesizer,
                    text,
                    play_audio,
                    thinking_done,
                    affirmation_audio=affirmation_audio,
                    opening=opening,
                )
            )
            logger.debug("Pipeline: Started dynamic thinking phrase generation")
        elif opening is not None:
            # Static mode: the thinking phrase is already playing
            self._thinking_task = asyncio.create_task(
                self._wait_for_opening(opening, thinking_done)
            )
        elif thinking_audio is not None and len(thinking_audio) > 0:
            # Static mode: use pre-synthesized audio
            self._thinking_task = asyncio.create_task(
                self._play_thinking_phrase(thinking_audio, play_audio, thinking_done)
            )
            logger.debug(f"Pipeline: Started static thinking phrase: {thinking_phrase}")
        else:
            thinking_done.set()  # No thinking phrase, skip waiting

        # Stage 3: Response generation with TTS
        try:
            # Create a wrapper callback that waits for thinking phrase first
            waited_for_thinking = [False]  # Use list for closure mutability

            async def gated_play_audio(audio_chunk: np.ndarray) -> None:
                """Play audio, waiting for thinking phrase first."""
                if not waited_for_thinking[0]:
                    # Wait for thinking phrase to finish
                    await thinking_done.wait()
                    waited_for_thinking[0] = True

                    # Wait for silence before first response sentence
                    await wait_for_silence()

                if self._cancelled:
                    return  # cancelled while it waited (cancel() opens the gate above)
                self.answer_started = True
                if self.observer is not None:
                    self.observer.answer_clip_next()
                await play_audio(audio_chunk)

            # Stream response through TTS
            response_stream = self._response.process_stream(text, context)
            await self._tts.process_stream(response_stream, context, gated_play_audio)

        except asyncio.CancelledError:
            logger.debug("Pipeline: Response/TTS cancelled")
            await self.cancel()
            return "cancelled"
        finally:
            if self.observer is not None:
                self.observer.replied(getattr(self._response, "last_turn", None))

        if self.poster is not None and not self._cancelled:
            # A follow-up without the wake word may rest on the previous turn's lookup
            await self._offer_post(text, play_audio, followup=accept is not None)

        return None

    async def _offer_post(
        self, text: str, play_audio: Callable[[np.ndarray], Awaitable[None]],
        followup: bool = False,
    ) -> None:
        """After the answer: if it's worth having in writing, say so and post it."""
        try:
            earlier = ""
            if followup and hasattr(self._response, "earlier_lookups"):
                earlier = self._response.earlier_lookups()
            material = await self.poster.check(
                text, getattr(self._response, "last_turn", None), earlier
            )
            if material is None:
                return
            self.poster.write_and_post(material)  # written while the line plays
            audio = await self.poster.announcement()
            if audio is not None and len(audio) > 0:
                await play_audio(audio)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Pipeline: Chat post check failed: {e}")

    async def _play_thinking_phrase(
        self,
        audio: np.ndarray,
        play_audio: Callable[[np.ndarray], Awaitable[None]],
        done_event: asyncio.Event,
    ) -> None:
        """Play thinking phrase audio.

        Args:
            audio: Pre-synthesized thinking phrase audio
            play_audio: Async function to play audio
            done_event: Event to signal when done
        """
        try:
            await play_audio(audio)
        except asyncio.CancelledError:
            logger.debug("Thinking phrase cancelled")
        except Exception as e:
            logger.error(f"Error playing thinking phrase: {e}")
        finally:
            done_event.set()

    async def _wait_for_opening(self, opening: OpeningPhrase, done_event: asyncio.Event) -> None:
        try:
            await opening.wait()
        except asyncio.CancelledError:
            logger.debug("Opening phrase cancelled")
        finally:
            done_event.set()

    async def _generate_and_play_dynamic_thinking(
        self,
        generator: "ThinkingPhraseGenerator",
        tts: TTSSynthesizerProtocol,
        transcribed_text: str,
        play_audio: Callable[[np.ndarray], Awaitable[None]],
        done_event: asyncio.Event,
        affirmation_audio: Optional[np.ndarray] = None,
        opening: Optional[OpeningPhrase] = None,
    ) -> None:
        """Play affirmation while generating dynamic phrase in parallel.

        Parallel approach to minimize gap:
        1. Start LLM generation for dynamic phrase
        2. Play affirmation while generation runs
        3. Wait for generation, synthesize, and play dynamic phrase

        Args:
            generator: ThinkingPhraseGenerator for dynamic phrase creation
            tts: TTS synthesizer for audio generation
            transcribed_text: The user's transcribed request
            play_audio: Async function to play audio
            done_event: Event to signal when done
            affirmation_audio: Pre-synthesized affirmation to play immediately
            opening: Affirmation already started at speech end (used instead of
                affirmation_audio)
        """
        generate_task = None
        try:
            # Start dynamic phrase generation immediately (runs in background)
            generate_task = asyncio.create_task(generator.generate(transcribed_text))

            # Play affirmation while generation runs in parallel
            if opening is not None:
                await opening.wait()
                if opening.skipped:
                    # Someone is still talking; no thinking phrase either
                    return
            elif affirmation_audio is not None and len(affirmation_audio) > 0:
                await play_audio(affirmation_audio)
                logger.debug("Pipeline: Played affirmation phrase")

            # Wait for dynamic phrase generation to complete
            phrase = await generate_task
            logger.debug(f"Pipeline: Generated dynamic phrase: {phrase}")

            # Check if LLM decided to skip (simple request, no thinking needed)
            if phrase.upper().strip() == "SKIP":
                logger.info("Pipeline: Skipping dynamic phrase (simple request)")
            else:
                # Synthesize and play the phrase
                audio = await tts.synthesize(phrase)

                if audio is not None and len(audio) > 0:
                    await play_audio(audio)
                    logger.debug("Pipeline: Played dynamic thinking phrase")
                else:
                    logger.warning("Pipeline: TTS returned empty audio for dynamic phrase")

        except asyncio.CancelledError:
            logger.debug("Pipeline: Dynamic thinking phrase cancelled")
        except Exception as e:
            logger.warning(f"Pipeline: Dynamic thinking phrase failed: {e}, skipping")
        finally:
            if generate_task is not None and not generate_task.done():
                generate_task.cancel()
            done_event.set()

    def forget_last_turn(self) -> bool:
        """Drop the latest exchange from the conversation history (see ResponseStage)."""
        forget = getattr(self._response, "forget_last_turn", None)
        return bool(forget and forget())

    async def cancel(self) -> None:
        """Cancel all ongoing processing.

        Propagates cancellation to all stages.
        """
        self._cancelled = True

        if self._opening is not None:
            self._opening.cancel()

        # Cancel thinking phrase if running
        if self._thinking_task and not self._thinking_task.done():
            self._thinking_task.cancel()
            try:
                await self._thinking_task
            except asyncio.CancelledError:
                pass

        # Cancel pipeline stages
        await self._transcription.cancel()
        await self._tts.cancel()

        logger.debug("Pipeline: Cancelled all stages")

    def reset_conversation(self) -> None:
        """Reset conversation state for new session."""
        self._response.reset_conversation()
        logger.debug("Pipeline: Conversation reset")
