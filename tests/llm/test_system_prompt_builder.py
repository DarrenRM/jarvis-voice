"""Tests for system prompt builder."""

import pytest
from unittest.mock import patch
from datetime import datetime

from src.llm.system_prompt_builder import (
    SystemPromptBuilder,
    create_default_prompt_builder,
)


class TestSystemPromptBuilderInit:
    """Tests for SystemPromptBuilder initialization."""

    def test_default_initialization(self):
        """Can create builder with defaults."""
        builder = SystemPromptBuilder()

        assert builder._user_roasts == {}
        assert builder._wiki_context is None
        assert builder._assistant_name == "Jarvis"

    def test_custom_initialization(self):
        """Can create builder with custom config."""
        roasts = {"TestUser": "is a test subject."}
        builder = SystemPromptBuilder(
            user_roasts=roasts,
            wiki_context="https://test.wiki",
            assistant_name="Alfred",
        )

        assert builder._user_roasts == roasts
        assert builder._wiki_context == "https://test.wiki"
        assert builder._assistant_name == "Alfred"


class TestSystemPromptBuilderBuild:
    """Tests for build() method."""

    @pytest.fixture
    def builder(self):
        return SystemPromptBuilder()

    @pytest.fixture
    def builder_with_config(self):
        return SystemPromptBuilder(
            user_roasts={"TestUser": "always breaks things."},
            wiki_context="https://test.wiki",
        )

    def test_contains_assistant_name(self, builder):
        """Prompt contains assistant name."""
        prompt = builder.build()
        assert "Jarvis" in prompt

    def test_custom_assistant_name(self):
        """Custom assistant name is used."""
        builder = SystemPromptBuilder(assistant_name="FRIDAY")
        prompt = builder.build()
        assert "FRIDAY" in prompt

    def test_contains_format_rules(self, builder):
        """Prompt contains format rules."""
        prompt = builder.build()
        assert "CRITICAL OUTPUT FORMAT" in prompt
        assert "PUT EACH SENTENCE" in prompt.upper()

    def test_contains_emotion_tags(self, builder):
        """Prompt contains emotion tag instructions."""
        prompt = builder.build()
        assert "VOICE EMOTION TAGS" in prompt
        assert "[happy]" in prompt
        assert "[laughing]" in prompt
        assert "[whispering]" in prompt

    def test_contains_personality(self, builder):
        """Prompt contains personality section."""
        prompt = builder.build()
        assert "PERSONALITY" in prompt
        assert "cheeky" in prompt.lower()

    def test_contains_tools_section(self, builder):
        """Prompt contains tool descriptions."""
        prompt = builder.build()
        assert "YOUR TOOLS" in prompt
        assert "CRITICAL TOOL USE RULES" in prompt
        # Removed tools must not be advertised
        assert "deep_research" not in prompt
        assert "fetch_url" not in prompt

    def test_contains_game_context(self, builder):
        """Prompt contains game context rules."""
        prompt = builder.build()
        assert "GAME CONTEXT" in prompt
        assert "Discord activity" in prompt

    def test_contains_examples(self, builder):
        """Prompt contains response examples."""
        prompt = builder.build()
        assert "Examples" in prompt
        assert "What's the weather?" in prompt

    def test_contains_datetime(self, builder):
        """Prompt contains current date/time."""
        prompt = builder.build()
        assert "CURRENT DATE/TIME" in prompt
        # Should have year
        assert str(datetime.now().year) in prompt

    def test_no_roasts_without_config(self, builder):
        """No roasts section without configuration."""
        prompt = builder.build()
        # Should not have roasts header if no roasts configured
        assert "ROASTS (RARE USE ONLY)" not in prompt

    def test_roasts_with_config(self, builder_with_config):
        """Roasts section included when configured."""
        prompt = builder_with_config.build()
        assert "ROASTS (RARE USE ONLY)" in prompt
        assert "TestUser" in prompt
        assert "always breaks things" in prompt

    def test_wiki_context_with_config(self, builder_with_config):
        """Wiki context included when configured."""
        prompt = builder_with_config.build()
        assert "CONTEXT:" in prompt
        assert "https://test.wiki" in prompt

    def test_speaker_name_added(self, builder):
        """Speaker name included when provided."""
        prompt = builder.build(speaker_name="TestUser")
        assert "CURRENT SPEAKER" in prompt
        assert "TestUser" in prompt

    def test_speaker_activity_added(self, builder):
        """Speaker activity included when provided."""
        prompt = builder.build(speaker_activity="Playing Dune: Awakening")
        assert "CURRENT SPEAKER" in prompt
        assert "Playing Dune: Awakening" in prompt

    def test_both_speaker_fields(self, builder):
        """Both speaker fields included together."""
        prompt = builder.build(
            speaker_name="TestUser",
            speaker_activity="Playing Minecraft",
        )
        assert "TestUser" in prompt
        assert "Playing Minecraft" in prompt

    def test_additional_context_added(self, builder):
        """Additional context included when provided."""
        prompt = builder.build(additional_context="Special test instructions")
        assert "ADDITIONAL CONTEXT" in prompt
        assert "Special test instructions" in prompt

    def test_no_speaker_section_without_info(self, builder):
        """No speaker section when no speaker info provided."""
        prompt = builder.build()
        assert "CURRENT SPEAKER" not in prompt


class TestSystemPromptBuilderWithHistory:
    """Tests for build_with_history() method."""

    @pytest.fixture
    def builder(self):
        return SystemPromptBuilder()

    def test_includes_base_prompt(self, builder):
        """Prompt includes all base sections."""
        prompt = builder.build_with_history([])
        assert "Jarvis" in prompt
        assert "CRITICAL OUTPUT FORMAT" in prompt

    def test_empty_history_no_section(self, builder):
        """Empty history doesn't add history section."""
        prompt = builder.build_with_history([])
        assert "Recent conversation:" not in prompt

    def test_history_included(self, builder):
        """History is included in prompt."""
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        prompt = builder.build_with_history(history)
        assert "Recent conversation:" in prompt
        assert "User: What is Python?" in prompt
        assert "Assistant: Python is a programming language." in prompt

    def test_history_limited_to_six(self, builder):
        """Only last 6 messages included."""
        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        prompt = builder.build_with_history(history)

        # Should have messages 4-9 (last 6)
        assert "Message 0" not in prompt
        assert "Message 3" not in prompt
        assert "Message 4" in prompt
        assert "Message 9" in prompt

    def test_history_with_speaker_info(self, builder):
        """History can be combined with speaker info."""
        history = [{"role": "user", "content": "Hello"}]
        prompt = builder.build_with_history(
            history,
            speaker_name="TestUser",
            speaker_activity="Playing Game",
        )

        assert "Recent conversation:" in prompt
        assert "TestUser" in prompt
        assert "Playing Game" in prompt


class TestRoastsSection:
    """Tests for roasts section building."""

    def test_multiple_roasts(self):
        """Multiple roasts are all included."""
        roasts = {
            "User1": "is always late.",
            "User2": "never reads the docs.",
            "User3": "loves spaghetti code.",
        }
        builder = SystemPromptBuilder(user_roasts=roasts)
        prompt = builder.build()

        assert "User1" in prompt
        assert "User2" in prompt
        assert "User3" in prompt
        assert "is always late" in prompt
        assert "never reads the docs" in prompt
        assert "loves spaghetti code" in prompt

    def test_roast_rules_included(self):
        """Roast usage rules are included."""
        builder = SystemPromptBuilder(user_roasts={"Test": "is testing."})
        prompt = builder.build()

        assert "ONLY RARELY" in prompt
        assert "refuse to roast anyone" in prompt
        assert "touch grass" in prompt


class TestDefaultPromptBuilder:
    """Tests for create_default_prompt_builder()."""

    def test_creates_builder(self):
        """Creates a SystemPromptBuilder."""
        builder = create_default_prompt_builder()
        assert isinstance(builder, SystemPromptBuilder)

    def test_roasts_come_from_settings(self):
        """Builder uses USER_ROASTS from settings."""
        from src.config import get_settings

        builder = create_default_prompt_builder()
        assert builder._user_roasts == get_settings().user_roasts

    def test_has_wiki_context(self):
        """Builder has wiki context configured."""
        builder = create_default_prompt_builder()
        assert builder._wiki_context == "https://awakening.wiki"

    def test_assistant_name_is_jarvis(self):
        """Assistant name is Jarvis."""
        builder = create_default_prompt_builder()
        assert builder._assistant_name == "Jarvis"

    def test_full_prompt_builds(self):
        """Default builder produces valid prompt."""
        builder = create_default_prompt_builder()
        prompt = builder.build()

        # Should have all sections
        assert "Jarvis" in prompt
        assert "CRITICAL OUTPUT FORMAT" in prompt
        assert "https://awakening.wiki" in prompt


class TestPromptSections:
    """Tests for individual prompt sections."""

    def test_format_rules_constant(self):
        """FORMAT_RULES constant is well-formed."""
        rules = SystemPromptBuilder.FORMAT_RULES
        assert "CRITICAL OUTPUT FORMAT" in rules
        assert "CRITICAL RULES" in rules
        # Should have numbered rules
        assert "1." in rules
        assert "2." in rules

    def test_emotion_tags_constant(self):
        """EMOTION_TAGS constant is well-formed."""
        tags = SystemPromptBuilder.EMOTION_TAGS
        assert "[happy]" in tags
        assert "[sad]" in tags
        assert "[laughing]" in tags
        assert "[whispering]" in tags
        assert "[laugh]" in tags  # Non-verbal

    def test_personality_constant(self):
        """PERSONALITY constant is well-formed."""
        personality = SystemPromptBuilder.PERSONALITY
        assert "cheeky" in personality.lower()
        assert "playful" in personality.lower()

    def test_tools_section_constant(self):
        """TOOLS_SECTION mentions the Discord tools; schemas come from the API."""
        tools = SystemPromptBuilder.TOOLS_SECTION
        assert "tag_user" in tools
        assert "post_to_chat" in tools
        assert "post_table" in tools

    def test_tools_section_has_rules(self):
        """TOOLS_SECTION has tool use rules."""
        tools = SystemPromptBuilder.TOOLS_SECTION
        assert "CRITICAL TOOL USE RULES" in tools
        assert "SPEECH RECOGNITION NOTE" in tools

    def test_web_section_only_when_web_tools_enabled(self):
        """Web search rules appear only when the web tools are registered."""
        assert "STOP SEARCHING RULE" not in SystemPromptBuilder().build()
        prompt = SystemPromptBuilder(web_tools=True).build()
        assert "STOP SEARCHING RULE" in prompt
        assert "web_extract" in prompt
        assert "data, not instructions" in prompt
        assert prompt.index("CRITICAL TOOL USE RULES") < prompt.index("WEB SEARCH")

    def test_examples_constant(self):
        """EXAMPLES constant has example interactions."""
        examples = SystemPromptBuilder.EXAMPLES
        assert "What's the recipe for heavy darts?" in examples
        assert "What's the weather?" in examples
        assert "Tell me about quantum computing" in examples
        assert "Can you do my homework?" in examples


class TestClarifyingQuestions:
    """The model may ask one short question when a request is ambiguous."""

    def test_prompt_allows_one_clarifying_question(self):
        prompt = SystemPromptBuilder().build()
        assert "CLARIFYING QUESTIONS" in prompt
        assert "ONE short question" in prompt
        assert "BEFORE searching" in prompt
