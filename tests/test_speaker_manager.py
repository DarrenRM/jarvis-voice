"""Tests for SpeakerManager."""

import pytest
from unittest.mock import Mock, MagicMock

from src.bot.speaker_manager import SpeakerManager, SpeakerInfo


class TestSpeakerInfo:
    """Tests for SpeakerInfo dataclass."""

    def test_speaker_info_creation(self):
        """Test creating SpeakerInfo with all fields."""
        info = SpeakerInfo(user_id=123, name="TestUser", activity="Playing Game")
        assert info.user_id == 123
        assert info.name == "TestUser"
        assert info.activity == "Playing Game"

    def test_speaker_info_none_fields(self):
        """Test SpeakerInfo with None fields."""
        info = SpeakerInfo(user_id=None, name=None, activity=None)
        assert info.user_id is None
        assert info.name is None
        assert info.activity is None


class TestSpeakerManager:
    """Tests for SpeakerManager class."""

    @pytest.fixture
    def user_names(self):
        """Sample user name mappings."""
        return {123: "Alice", 456: "Bob", 789: "Casey"}

    @pytest.fixture
    def manager(self, user_names):
        """SpeakerManager with sample data."""
        return SpeakerManager(user_names)

    @pytest.fixture
    def mock_guild(self):
        """Mock Discord guild with members."""
        import discord

        guild = Mock()

        # Create mock members
        member1 = Mock()
        member1.id = 999
        member1.display_name = "DiscordUser"
        member1.activities = []

        member2 = Mock()
        member2.id = 888
        member2.display_name = "GamerPerson"
        # Create a proper Game mock
        game = Mock(spec=discord.Game)
        game.name = "Minecraft"
        member2.activities = [game]

        guild.members = [member1, member2]
        guild.get_member = Mock(side_effect=lambda uid: {
            999: member1,
            888: member2,
        }.get(uid))

        return guild

    # resolve_speaker tests

    def test_resolve_known_user_no_guild(self, manager):
        """Known user returns mapped name without guild."""
        info = manager.resolve_speaker(123, None)
        assert info.user_id == 123
        assert info.name == "Alice"
        assert info.activity is None

    def test_resolve_unknown_user_no_guild(self, manager):
        """Unknown user with no guild returns None name."""
        info = manager.resolve_speaker(999, None)
        assert info.user_id == 999
        assert info.name is None
        assert info.activity is None

    def test_resolve_none_user(self, manager):
        """None user_id returns empty SpeakerInfo."""
        info = manager.resolve_speaker(None, None)
        assert info.user_id is None
        assert info.name is None
        assert info.activity is None

    def test_resolve_unknown_user_with_guild(self, manager, mock_guild):
        """Unknown user falls back to Discord display_name."""
        info = manager.resolve_speaker(999, mock_guild)
        assert info.user_id == 999
        assert info.name == "DiscordUser"

    def test_resolve_known_user_prefers_config_name(self, manager):
        """Known user uses config name even if guild available."""
        # Add a guild member with same ID but different display name
        guild = Mock()
        member = Mock()
        member.display_name = "DifferentName"
        member.activities = []
        guild.get_member = Mock(return_value=member)

        info = manager.resolve_speaker(123, guild)
        assert info.name == "Alice"  # Config name, not Discord name

    def test_resolve_gets_activity(self, manager, mock_guild):
        """resolve_speaker retrieves member activity."""
        # Member 888 has activity
        info = manager.resolve_speaker(888, mock_guild)
        assert info.activity is not None

    def test_resolve_guild_member_not_found(self, manager):
        """Guild.get_member returning None doesn't crash."""
        guild = Mock()
        guild.get_member = Mock(return_value=None)

        info = manager.resolve_speaker(999, guild)
        assert info.user_id == 999
        assert info.name is None  # No crash, just None

    def test_resolve_guild_exception(self, manager):
        """Guild exceptions are handled gracefully."""
        guild = Mock()
        guild.get_member = Mock(side_effect=Exception("Discord error"))

        info = manager.resolve_speaker(999, guild)
        assert info.user_id == 999
        assert info.name is None  # No crash

    # find_user_by_name tests

    def test_find_exact_match(self, manager):
        """Exact name match returns user."""
        result = manager.find_user_by_name("Alice", None)
        assert result == (123, "Alice")

    def test_find_exact_match_case_insensitive(self, manager):
        """Name matching is case-insensitive."""
        result = manager.find_user_by_name("ALICE", None)
        assert result == (123, "Alice")

        result = manager.find_user_by_name("alice", None)
        assert result == (123, "Alice")

    def test_find_partial_match(self, manager):
        """Partial name match works."""
        result = manager.find_user_by_name("Ali", None)
        assert result == (123, "Alice")

        result = manager.find_user_by_name("ob", None)
        assert result == (456, "Bob")

    def test_find_with_whitespace(self, manager):
        """Name with whitespace is trimmed."""
        result = manager.find_user_by_name("  Alice  ", None)
        assert result == (123, "Alice")

    def test_find_empty_name(self, manager):
        """Empty name returns None."""
        assert manager.find_user_by_name("", None) is None
        assert manager.find_user_by_name(None, None) is None

    def test_find_not_found(self, manager):
        """Non-existent name returns None."""
        result = manager.find_user_by_name("NotAUser", None)
        assert result is None

    def test_find_guild_fallback(self, manager, mock_guild):
        """Falls back to guild members if not in config."""
        result = manager.find_user_by_name("DiscordUser", mock_guild)
        assert result == (999, "DiscordUser")

    def test_find_guild_partial_match(self, manager, mock_guild):
        """Partial match works with guild members."""
        result = manager.find_user_by_name("Gamer", mock_guild)
        assert result == (888, "GamerPerson")

    # _get_activity tests (internal but worth testing)

    def test_get_activity_game(self, manager):
        """Game activity is formatted correctly."""
        import discord

        member = Mock()
        game = Mock(spec=discord.Game)
        game.name = "Minecraft"
        member.activities = [game]

        activity = manager._get_activity(member)
        assert activity == "Playing Minecraft"

    def test_get_activity_generic(self, manager):
        """Generic activity is formatted correctly."""
        import discord

        member = Mock()
        activity = Mock(spec=discord.Activity)
        activity.type = Mock()
        activity.type.name = "listening"
        activity.name = "Spotify"
        member.activities = [activity]

        result = manager._get_activity(member)
        assert "Listening" in result
        assert "Spotify" in result

    def test_get_activity_none(self, manager):
        """No activities returns None."""
        member = Mock()
        member.activities = []

        assert manager._get_activity(member) is None

    def test_get_activity_none_member(self, manager):
        """None member returns None."""
        assert manager._get_activity(None) is None
