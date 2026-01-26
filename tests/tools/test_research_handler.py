"""Tests for research tool handler."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from src.tools.handlers.research import DeepResearchHandler


class MockClaudeService:
    """Mock Claude service for testing."""

    def __init__(self, response: str = "Research complete. Found relevant information."):
        self.response = response
        self.chat = AsyncMock(return_value=response)


class TestDeepResearchHandler:
    """Tests for DeepResearchHandler."""

    @pytest.fixture
    def claude_service(self):
        return MockClaudeService()

    @pytest.fixture
    def handler(self, claude_service):
        return DeepResearchHandler(claude_service=claude_service)

    @pytest.fixture
    def handler_no_service(self):
        return DeepResearchHandler()

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "deep_research"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "deep_research"
        assert len(defn.description) > 0
        assert "task" in defn.parameters["properties"]
        assert "context" in defn.parameters["properties"]
        assert "task" in defn.parameters["required"]

    @pytest.mark.asyncio
    async def test_successful_research_without_context(self, handler, claude_service):
        """Research without context uses tools."""
        result = await handler.execute({"task": "What are the best hiking trails in Colorado?"})

        assert result.success is True
        assert "Deep research result:" in result.output
        assert "Research complete" in result.output
        claude_service.chat.assert_called_once()
        # Without context, should use tools
        call_kwargs = claude_service.chat.call_args
        assert call_kwargs[1]["use_tools"] is True

    @pytest.mark.asyncio
    async def test_successful_research_with_context(self, handler, claude_service):
        """Research with context does not use tools (synthesis only)."""
        context = "Here are search results: " + "x" * 200  # Over 100 chars
        result = await handler.execute({
            "task": "Summarize these findings",
            "context": context,
        })

        assert result.success is True
        claude_service.chat.assert_called_once()
        # With context, should NOT use tools
        call_kwargs = claude_service.chat.call_args
        assert call_kwargs[1]["use_tools"] is False

    @pytest.mark.asyncio
    async def test_empty_task_fails(self, handler):
        """Empty task returns error."""
        result = await handler.execute({"task": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_task_fails(self, handler):
        """Whitespace-only task returns error."""
        result = await handler.execute({"task": "   "})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_task_fails(self, handler):
        """Missing task returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_service_fails(self, handler_no_service):
        """No Claude service returns not configured error."""
        result = await handler_no_service.execute({"task": "Research something"})

        assert result.success is False
        assert "Claude Code SDK" in result.error

    @pytest.mark.asyncio
    async def test_timeout_handled(self, handler, claude_service):
        """Timeout is handled gracefully."""
        async def slow_chat(*args, **kwargs):
            await asyncio.sleep(100)  # Will timeout
            return "result"

        claude_service.chat.side_effect = slow_chat

        # Set shorter timeout for test
        handler.TIMEOUT_WITH_TOOLS = 0.1

        result = await handler.execute({"task": "Research something"})

        assert result.success is False
        assert "too long" in result.error.lower()
        assert result.metadata.get("timeout") == 0.1

    @pytest.mark.asyncio
    async def test_exception_handled(self, handler, claude_service):
        """Exceptions are handled gracefully."""
        claude_service.chat.side_effect = Exception("Claude API error")

        result = await handler.execute({"task": "Research something"})

        assert result.success is False
        assert "Claude API error" in result.error

    @pytest.mark.asyncio
    async def test_metadata_includes_research_info(self, handler):
        """Result metadata includes research information."""
        result = await handler.execute({"task": "Research quantum computing"})

        assert result.metadata["task"] == "Research quantum computing"
        assert result.metadata["used_tools"] is True
        assert result.metadata["context_provided"] is False
        assert result.metadata["response_length"] > 0

    def test_set_service(self, handler_no_service, claude_service):
        """Can set service after construction."""
        handler_no_service.set_service(claude_service)
        assert handler_no_service._claude_service is claude_service


class TestBuildDelegationPrompt:
    """Tests for build_delegation_prompt method."""

    @pytest.fixture
    def handler(self):
        return DeepResearchHandler()

    def test_prompt_without_context(self, handler):
        """Prompt without context includes research instructions."""
        prompt = handler.build_delegation_prompt(
            original_request="What is quantum computing?",
            gathered_context="",
        )

        assert "TASK: What is quantum computing?" in prompt
        assert "WebSearch" in prompt  # Should mention tools
        assert "WebFetch" in prompt
        assert "deep research" in prompt.lower()

    def test_prompt_with_context(self, handler):
        """Prompt with context includes synthesis instructions."""
        context = "Search results: Quantum computing uses qubits... " + "x" * 200
        prompt = handler.build_delegation_prompt(
            original_request="Summarize quantum computing",
            gathered_context=context,
        )

        assert "TASK: Summarize quantum computing" in prompt
        assert "GATHERED RESEARCH" in prompt
        assert "synthesiz" in prompt.lower()
        # Should NOT mention tools for synthesis
        assert "WebSearch" not in prompt

    def test_context_truncated_at_4000_chars(self, handler):
        """Long context is truncated to 4000 chars."""
        long_context = "Q" * 5000  # Use 'Q' to avoid conflict with other text
        prompt = handler.build_delegation_prompt(
            original_request="Summarize",
            gathered_context=long_context,
        )

        # Context should be truncated to 4000 chars
        context_in_prompt = prompt.count("Q")
        assert context_in_prompt == 4000

    def test_conversation_history_included(self, handler):
        """Conversation history is added to prompt."""
        history = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language"},
        ]
        prompt = handler.build_delegation_prompt(
            original_request="Continue explaining",
            gathered_context="",
            conversation_history=history,
        )

        assert "RECENT CONVERSATION" in prompt
        assert "User: Tell me about Python" in prompt
        assert "Assistant: Python is a programming language" in prompt

    def test_long_history_messages_truncated(self, handler):
        """Long history messages are truncated with ellipsis."""
        long_message = "a" * 300
        history = [{"role": "user", "content": long_message}]
        prompt = handler.build_delegation_prompt(
            original_request="Test",
            gathered_context="",
            conversation_history=history,
        )

        assert "..." in prompt  # Truncation marker
        assert long_message not in prompt  # Full message not present

    def test_emotion_tags_mentioned(self, handler):
        """Prompt mentions emotion tags for TTS."""
        prompt = handler.build_delegation_prompt(
            original_request="Test",
            gathered_context="",
        )

        assert "emotion tags" in prompt.lower()
        assert "[happy]" in prompt or "[thoughtful]" in prompt

    def test_voice_output_guidelines(self, handler):
        """Prompt includes voice output guidelines."""
        prompt = handler.build_delegation_prompt(
            original_request="Test",
            gathered_context="",
        )

        assert "spoken aloud" in prompt.lower()
        assert "concise" in prompt.lower()


class TestContextThreshold:
    """Tests for context threshold logic."""

    @pytest.fixture
    def claude_service(self):
        return MockClaudeService()

    @pytest.fixture
    def handler(self, claude_service):
        return DeepResearchHandler(claude_service=claude_service)

    @pytest.mark.asyncio
    async def test_short_context_uses_tools(self, handler, claude_service):
        """Context under 100 chars triggers tool use."""
        result = await handler.execute({
            "task": "Research this",
            "context": "Short context",  # Under 100 chars
        })

        assert result.success is True
        call_kwargs = claude_service.chat.call_args
        assert call_kwargs[1]["use_tools"] is True

    @pytest.mark.asyncio
    async def test_long_context_skips_tools(self, handler, claude_service):
        """Context over 100 chars skips tool use."""
        result = await handler.execute({
            "task": "Summarize this",
            "context": "x" * 101,  # Over 100 chars
        })

        assert result.success is True
        call_kwargs = claude_service.chat.call_args
        assert call_kwargs[1]["use_tools"] is False

    @pytest.mark.asyncio
    async def test_exactly_100_chars_uses_tools(self, handler, claude_service):
        """Context at exactly 100 chars uses tools (threshold is >100)."""
        result = await handler.execute({
            "task": "Research this",
            "context": "x" * 100,  # Exactly 100 chars
        })

        assert result.success is True
        call_kwargs = claude_service.chat.call_args
        assert call_kwargs[1]["use_tools"] is True

    @pytest.mark.asyncio
    async def test_whitespace_context_uses_tools(self, handler, claude_service):
        """Whitespace-heavy context that strips to <100 uses tools."""
        result = await handler.execute({
            "task": "Research this",
            "context": "   " * 50 + "short",  # Lots of whitespace
        })

        assert result.success is True
        call_kwargs = claude_service.chat.call_args
        # After strip, "short" is <100 chars
        assert call_kwargs[1]["use_tools"] is True
