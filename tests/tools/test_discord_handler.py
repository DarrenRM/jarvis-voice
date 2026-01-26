"""Tests for Discord tool handlers."""

import pytest
from unittest.mock import AsyncMock

from src.tools.handlers.discord import (
    TagUserHandler,
    PostMessageHandler,
    PostTableHandler,
)


class TestTagUserHandler:
    """Tests for TagUserHandler."""

    @pytest.fixture
    def callback(self):
        return AsyncMock(return_value="Successfully tagged @user")

    @pytest.fixture
    def handler(self, callback):
        return TagUserHandler(on_user_tagged=callback)

    @pytest.fixture
    def handler_no_callback(self):
        return TagUserHandler()

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "tag_user"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "tag_user"
        assert len(defn.description) > 0
        assert "target_name" in defn.parameters["properties"]
        assert "target_name" in defn.parameters["required"]

    @pytest.mark.asyncio
    async def test_successful_tag(self, handler, callback):
        """Successful tag returns result."""
        result = await handler.execute({
            "target_name": "john",
            "message": "Check this out!"
        })

        assert result.success is True
        assert "tagged" in result.output.lower()
        callback.assert_called_once_with("john", "Check this out!")

    @pytest.mark.asyncio
    async def test_tag_without_message(self, handler, callback):
        """Can tag without message."""
        result = await handler.execute({"target_name": "john"})

        assert result.success is True
        callback.assert_called_once_with("john", "")

    @pytest.mark.asyncio
    async def test_empty_target_fails(self, handler):
        """Empty target name returns error."""
        result = await handler.execute({"target_name": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_target_fails(self, handler):
        """Missing target_name returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_callback_fails(self, handler_no_callback):
        """No callback returns not connected error."""
        result = await handler_no_callback.execute({"target_name": "john"})

        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_callback_error_handled(self, handler, callback):
        """Callback exceptions are handled gracefully."""
        callback.side_effect = Exception("Discord API error")

        result = await handler.execute({"target_name": "john"})

        assert result.success is False
        assert "Discord API error" in result.error

    def test_set_callback(self, handler_no_callback):
        """Can set callback after construction."""
        callback = AsyncMock(return_value="Tagged!")
        handler_no_callback.set_callback(callback)

        assert handler_no_callback._on_user_tagged is callback

    @pytest.mark.asyncio
    async def test_metadata_includes_params(self, handler):
        """Result metadata includes parameters."""
        result = await handler.execute({
            "target_name": "john",
            "message": "Hello!"
        })

        assert result.metadata["target_name"] == "john"
        assert result.metadata["message"] == "Hello!"


class TestPostMessageHandler:
    """Tests for PostMessageHandler."""

    @pytest.fixture
    def callback(self):
        return AsyncMock(return_value="Message posted successfully")

    @pytest.fixture
    def handler(self, callback):
        return PostMessageHandler(on_message_posted=callback)

    @pytest.fixture
    def handler_no_callback(self):
        return PostMessageHandler()

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "post_to_chat"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "post_to_chat"
        assert len(defn.description) > 0
        assert "message" in defn.parameters["properties"]
        assert "message" in defn.parameters["required"]

    @pytest.mark.asyncio
    async def test_successful_post(self, handler, callback):
        """Successful post returns result."""
        result = await handler.execute({"message": "Hello, world!"})

        assert result.success is True
        assert "posted" in result.output.lower()
        callback.assert_called_once_with("Hello, world!")

    @pytest.mark.asyncio
    async def test_empty_message_fails(self, handler):
        """Empty message returns error."""
        result = await handler.execute({"message": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_message_fails(self, handler):
        """Whitespace-only message returns error."""
        result = await handler.execute({"message": "   "})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_message_fails(self, handler):
        """Missing message returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_callback_fails(self, handler_no_callback):
        """No callback returns not connected error."""
        result = await handler_no_callback.execute({"message": "Hello"})

        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_callback_error_handled(self, handler, callback):
        """Callback exceptions are handled gracefully."""
        callback.side_effect = Exception("Rate limited")

        result = await handler.execute({"message": "Hello"})

        assert result.success is False
        assert "Rate limited" in result.error

    @pytest.mark.asyncio
    async def test_metadata_includes_length(self, handler):
        """Result metadata includes message length."""
        result = await handler.execute({"message": "Hello!"})

        assert result.metadata["message_length"] == 6


class TestPostTableHandler:
    """Tests for PostTableHandler."""

    @pytest.fixture
    def callback(self):
        return AsyncMock(return_value="Table posted successfully")

    @pytest.fixture
    def handler(self, callback):
        return PostTableHandler(on_message_posted=callback)

    @pytest.fixture
    def handler_no_callback(self):
        return PostTableHandler()

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "post_table"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "post_table"
        assert len(defn.description) > 0
        assert "headers" in defn.parameters["properties"]
        assert "rows" in defn.parameters["properties"]
        assert "title" in defn.parameters["properties"]
        assert "headers" in defn.parameters["required"]
        assert "rows" in defn.parameters["required"]

    @pytest.mark.asyncio
    async def test_successful_post(self, handler, callback):
        """Successful table post returns result."""
        result = await handler.execute({
            "headers": ["Name", "Score"],
            "rows": [["Alice", "95"], ["Bob", "87"]],
        })

        assert result.success is True
        assert "posted" in result.output.lower()
        # Verify callback was called with formatted table
        call_args = callback.call_args[0][0]
        assert "```" in call_args  # Code block
        assert "Name" in call_args
        assert "Alice" in call_args

    @pytest.mark.asyncio
    async def test_table_with_title(self, handler, callback):
        """Table with title includes title in output."""
        result = await handler.execute({
            "title": "Scores",
            "headers": ["Name", "Score"],
            "rows": [["Alice", "95"]],
        })

        assert result.success is True
        call_args = callback.call_args[0][0]
        assert "**Scores**" in call_args

    @pytest.mark.asyncio
    async def test_empty_headers_fails(self, handler):
        """Empty headers returns error."""
        result = await handler.execute({
            "headers": [],
            "rows": [["Alice", "95"]],
        })

        assert result.success is False
        assert "headers" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_rows_fails(self, handler):
        """Empty rows returns error."""
        result = await handler.execute({
            "headers": ["Name", "Score"],
            "rows": [],
        })

        assert result.success is False
        assert "rows" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_headers_fails(self, handler):
        """Missing headers returns error."""
        result = await handler.execute({
            "rows": [["Alice", "95"]],
        })

        assert result.success is False

    @pytest.mark.asyncio
    async def test_missing_rows_fails(self, handler):
        """Missing rows returns error."""
        result = await handler.execute({
            "headers": ["Name", "Score"],
        })

        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_callback_fails(self, handler_no_callback):
        """No callback returns not connected error."""
        result = await handler_no_callback.execute({
            "headers": ["Name"],
            "rows": [["Alice"]],
        })

        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_metadata_includes_dimensions(self, handler):
        """Result metadata includes table dimensions."""
        result = await handler.execute({
            "title": "Test",
            "headers": ["A", "B", "C"],
            "rows": [["1", "2", "3"], ["4", "5", "6"]],
        })

        assert result.metadata["title"] == "Test"
        assert result.metadata["num_columns"] == 3
        assert result.metadata["num_rows"] == 2


class TestFormatTable:
    """Tests for static format_table method."""

    def test_basic_table(self):
        """Basic table formatting."""
        result = PostTableHandler.format_table(
            headers=["Name", "Score"],
            rows=[["Alice", "95"], ["Bob", "87"]],
        )

        assert "```" in result
        assert "Name" in result
        assert "Score" in result
        assert "Alice" in result
        assert "95" in result
        assert "Bob" in result
        assert "---" in result  # Separator

    def test_table_with_title(self):
        """Table with title."""
        result = PostTableHandler.format_table(
            headers=["Name"],
            rows=[["Alice"]],
            title="Players",
        )

        assert "**Players**" in result
        assert "```" in result

    def test_empty_table(self):
        """Empty table returns message."""
        result = PostTableHandler.format_table(headers=[], rows=[])
        assert result == "Empty table"

        result = PostTableHandler.format_table(headers=["A"], rows=[])
        assert result == "Empty table"

        result = PostTableHandler.format_table(headers=[], rows=[["a"]])
        assert result == "Empty table"

    def test_column_width_auto_adjusts(self):
        """Column width adjusts to content."""
        result = PostTableHandler.format_table(
            headers=["Short", "VeryLongHeaderName"],
            rows=[["x", "y"]],
        )

        # Long header should make column wider
        lines = result.split("\n")
        header_line = [l for l in lines if "Short" in l][0]
        assert "VeryLongHeaderName" in header_line

    def test_minimum_column_width(self):
        """Columns have minimum width."""
        result = PostTableHandler.format_table(
            headers=["A"],
            rows=[["x"]],
        )

        # Even short content should have some padding
        assert "|" in result

    def test_rows_with_different_lengths(self):
        """Handles rows with fewer columns gracefully."""
        result = PostTableHandler.format_table(
            headers=["A", "B", "C"],
            rows=[["1"], ["2", "3"]],  # Shorter rows
        )

        # Should not crash
        assert "```" in result
        assert "A" in result

    def test_centered_content(self):
        """Content is centered in cells."""
        result = PostTableHandler.format_table(
            headers=["Test"],
            rows=[["X"]],
        )

        # "X" should be padded with spaces on both sides
        lines = result.split("\n")
        data_line = [l for l in lines if "X" in l][0]
        # Find X and check it's not at edge
        x_pos = data_line.find("X")
        assert data_line[x_pos - 1] == " "  # Space before
