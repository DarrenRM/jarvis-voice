"""Cerebras LLM client (OpenAI-compatible Chat Completions API).

One streaming request per model turn: text is yielded as it arrives so TTS can
start speaking, and any tool calls are accumulated and returned at the end of
the turn. Reasoning tokens arrive in a separate ``reasoning`` field and are
never yielded, so they can't leak into speech.
"""

import logging
import time
from typing import AsyncIterator, Optional, Union

from .chat_orchestrator import ToolCall, TurnResult

logger = logging.getLogger(__name__)

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
# Longest silence allowed between stream chunks before the turn is abandoned
STREAM_STALL_SECONDS = 15.0


class CerebrasClient:
    """Streams chat completions from Cerebras (implements LLMClientProtocol)."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-3.8-27b",
        reasoning_effort: str = "high",
        max_completion_tokens: int = 16384,
        timeout: float = 60.0,
        base_url: str = CEREBRAS_BASE_URL,
    ):
        """Initialize Cerebras client.

        Args:
            api_key: Cerebras API key
            model: Model ID (e.g. qwen-3.8-27b)
            reasoning_effort: none, low, medium or high
            max_completion_tokens: Output budget per turn. Reasoning tokens count
                against this, so it must be far above the spoken answer length.
            timeout: Request timeout in seconds
            base_url: API base URL
        """
        import httpx
        from openai import AsyncOpenAI

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            # A turn normally streams continuously (reasoning included) and ends in
            # under 5s. A stream that stalls after its headers must fail fast so the
            # user hears the error reply instead of silence.
            timeout=httpx.Timeout(timeout, connect=5.0, read=min(timeout, STREAM_STALL_SECONDS)),
            max_retries=2,
        )

    def _request_kwargs(self, messages: list[dict], tools: Optional[list[dict]]) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            # Drop reasoning from earlier assistant turns instead of re-sending it
            "extra_body": {"clear_thinking": True},
        }
        # Never send tool_choice without tools - API v2 rejects it
        if tools:
            kwargs["tools"] = tools
        return kwargs

    async def stream_turn(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[Union[str, TurnResult]]:
        """Stream one model turn.

        Yields:
            Text chunks (str) as they arrive, then exactly one TurnResult.
        """
        t0 = time.perf_counter()
        stream = await self._client.chat.completions.create(
            stream=True,
            stream_options={"include_usage": True},
            **self._request_kwargs(messages, tools),
        )

        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish_reason: Optional[str] = None
        first_text_at: Optional[float] = None
        reasoning_chars = 0
        usage = None

        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            # Final usage chunk can arrive with no choices
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning = (delta.model_extra or {}).get("reasoning")
            if reasoning:
                reasoning_chars += len(reasoning)

            if delta.content:
                if first_text_at is None:
                    first_text_at = time.perf_counter() - t0
                content_parts.append(delta.content)
                yield delta.content

            for tc in delta.tool_calls or []:
                entry = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function and tc.function.name:
                    entry["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    entry["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if finish_reason == "length":
            logger.warning(
                f"Cerebras hit max_completion_tokens={self.max_completion_tokens} "
                f"(reasoning chars: {reasoning_chars})"
            )

        logger.info(
            f"[TIMING] LLM turn: first_text={first_text_at and round(first_text_at, 2)}s "
            f"total={time.perf_counter() - t0:.2f}s reasoning_chars={reasoning_chars} "
            f"tool_calls={len(calls)} finish={finish_reason}"
            + (f" tokens_in={usage.prompt_tokens} tokens_out={usage.completion_tokens}"
               if usage else "")
        )

        yield TurnResult(
            content="".join(content_parts),
            tool_calls=[
                ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                for _, c in sorted(calls.items())
            ],
            finish_reason=finish_reason,
        )

    async def complete(self, messages: list[dict]) -> str:
        """Non-streaming completion without tools (greetings, keep-alive)."""
        response = await self._client.chat.completions.create(
            **self._request_kwargs(messages, tools=None)
        )
        return (response.choices[0].message.content or "").strip()

    async def warm_up(self) -> None:
        """Open the connection with a one-token request.

        The first call after startup takes ~2.4s (TLS + first-request setup) versus
        ~0.2s after; doing it at startup keeps that off the first spoken answer.
        """
        t0 = time.perf_counter()
        try:
            await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1,
                reasoning_effort="none",
            )
            logger.info(f"[WARMUP] Cerebras ready in {time.perf_counter() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"[WARMUP] Cerebras warm-up failed: {e}")

    async def close(self) -> None:
        await self._client.close()
