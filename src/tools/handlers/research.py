"""Research tool handler for deep research via Claude delegation."""

import asyncio
import logging
from typing import Any, Optional, Protocol

from ..base import ToolDefinition, ToolResult, ToolHandler

logger = logging.getLogger(__name__)


class ClaudeService(Protocol):
    """Protocol for Claude delegation service."""

    async def chat(self, message: str, use_tools: bool = True) -> str:
        """Send message and get complete response."""
        ...


class DeepResearchHandler(ToolHandler):
    """Handler for deep research via Claude delegation.

    Delegates complex research tasks to Claude, which has access to
    web search tools and better reasoning capabilities.
    """

    # Context threshold for determining if we need Claude to research
    CONTEXT_THRESHOLD = 100

    # Timeout multipliers (applied to base_timeout)
    # With tools (web search): needs more time for multiple searches
    # Without tools (synthesis): faster, just needs to analyze provided context
    TIMEOUT_MULTIPLIER_WITH_TOOLS = 3.0
    TIMEOUT_MULTIPLIER_WITHOUT_TOOLS = 1.5

    def __init__(
        self,
        claude_service: Optional[ClaudeService] = None,
        base_timeout: float = 30.0,
    ):
        """Initialize deep research handler.

        Args:
            claude_service: Claude client for delegation
            base_timeout: Base timeout in seconds (from CLAUDE_TIMEOUT setting).
                         Multiplied by 3x for tool use, 1.5x for synthesis.
        """
        self._claude_service = claude_service
        self._base_timeout = base_timeout

    @property
    def name(self) -> str:
        return "deep_research"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="deep_research",
            description="Delegate complex research to a more capable model. Use for questions that need deeper analysis, multiple searches, or synthesis of information. This takes longer but produces better results.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The research question or task to investigate",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional pre-gathered context (search results, etc.) to help with the research",
                    },
                },
                "required": ["task"],
            },
        )

    def set_service(self, service: ClaudeService) -> None:
        """Set the Claude service for delegation."""
        self._claude_service = service

    def build_delegation_prompt(
        self,
        original_request: str,
        gathered_context: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Build prompt for Claude delegation.

        Args:
            original_request: What to analyze/research
            gathered_context: Pre-gathered search results or context
            conversation_history: Recent conversation for context

        Returns:
            Formatted prompt for Claude
        """
        has_context = gathered_context and len(gathered_context.strip()) > self.CONTEXT_THRESHOLD

        if has_context:
            # Claude has context to synthesize - no tools needed
            truncated = gathered_context[:4000] if len(gathered_context) > 4000 else gathered_context
            prompt = f"""You are synthesizing research results for a voice assistant.
Your response will be spoken aloud, so keep it concise and clear.

TASK: {original_request}

GATHERED RESEARCH (analyze and synthesize this):
{truncated}

GUIDELINES:
1. Synthesize the gathered information into a clear answer
2. Be concise - 2-4 sentences max for voice output
3. Focus on the most important facts
4. No preamble like "Based on the research..." - just give the answer
5. Use emotion tags like [happy], [thoughtful] at sentence starts for TTS"""
        else:
            # Claude needs to research - has web tools enabled
            prompt = f"""You are doing deep research for a voice assistant.
Use WebSearch and WebFetch tools to find comprehensive information.
Your response will be spoken aloud, so keep it concise.

TASK: {original_request}

GUIDELINES:
1. Search multiple sources to build a complete picture
2. After researching, provide a concise 2-4 sentence summary
3. Focus on the most important facts the user needs
4. No preamble - just give the answer after researching
5. Use emotion tags like [happy], [thoughtful] at sentence starts for TTS"""

        # Add conversation history context
        if conversation_history:
            history_text = "\n".join(
                f"- {msg['role'].title()}: {msg['content'][:200]}..."
                if len(msg.get('content', '')) > 200 else f"- {msg['role'].title()}: {msg.get('content', '')}"
                for msg in conversation_history
            )
            prompt += f"""

RECENT CONVERSATION (for context, may not be relevant):
{history_text}"""

        return prompt

    async def execute(
        self,
        args: dict[str, Any],
        conversation_history: Optional[list[dict]] = None,
    ) -> ToolResult:
        """Execute deep research via Claude delegation.

        Args:
            args: Must contain "task", optionally "context"
            conversation_history: Recent conversation context (passed separately)

        Returns:
            ToolResult with research result or error
        """
        if not self._claude_service:
            return ToolResult(
                success=False,
                output="",
                error="Deep research not available - Claude Code SDK not configured",
            )

        task = args.get("task", "").strip()
        if not task:
            return ToolResult(
                success=False,
                output="",
                error="Research task cannot be empty",
            )

        gathered_context = args.get("context", "")

        # Build delegation prompt
        prompt = self.build_delegation_prompt(
            original_request=task,
            gathered_context=gathered_context,
            conversation_history=conversation_history,
        )

        # Determine if Claude needs to use tools (research) or just synthesize
        has_context = len(gathered_context.strip()) > self.CONTEXT_THRESHOLD
        use_tools = not has_context
        multiplier = self.TIMEOUT_MULTIPLIER_WITH_TOOLS if use_tools else self.TIMEOUT_MULTIPLIER_WITHOUT_TOOLS
        timeout_secs = self._base_timeout * multiplier

        logger.info(f"Deep research: use_tools={use_tools}, context_len={len(gathered_context)}")

        try:
            response = await asyncio.wait_for(
                self._claude_service.chat(prompt, use_tools=use_tools),
                timeout=timeout_secs,
            )

            logger.info(f"Deep research completed: {response[:100]}...")

            return ToolResult(
                success=True,
                output=f"Deep research result:\n{response}",
                metadata={
                    "task": task,
                    "used_tools": use_tools,
                    "context_provided": has_context,
                    "response_length": len(response),
                },
            )

        except asyncio.TimeoutError:
            logger.warning(f"Deep research timed out after {timeout_secs}s")
            return ToolResult(
                success=False,
                output="",
                error="Deep research took too long. Try a simpler question.",
                metadata={
                    "task": task,
                    "timeout": timeout_secs,
                },
            )
        except Exception as e:
            logger.error(f"Deep research error: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Deep research failed: {str(e)}",
                metadata={"task": task},
            )
