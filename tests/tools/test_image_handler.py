"""Tests for image tool handlers."""

import pytest
from unittest.mock import AsyncMock

from src.tools.handlers.image import GenerateImageHandler, EditImageHandler


class MockImageService:
    """Mock image service for testing."""

    def __init__(self, has_chat_value: bool = False):
        self._has_chat = has_chat_value
        self.generate = AsyncMock(return_value=(b"image_bytes", "image/png"))
        self.edit = AsyncMock(return_value=(b"edited_bytes", "image/png"))

    def has_chat(self) -> bool:
        return self._has_chat

    def set_has_chat(self, value: bool) -> None:
        self._has_chat = value


class TestGenerateImageHandler:
    """Tests for GenerateImageHandler."""

    @pytest.fixture
    def image_service(self):
        return MockImageService()

    @pytest.fixture
    def callback(self):
        return AsyncMock(return_value="Image posted to #general")

    @pytest.fixture
    def handler(self, image_service, callback):
        return GenerateImageHandler(
            image_service=image_service,
            on_image_generated=callback,
        )

    @pytest.fixture
    def handler_no_service(self, callback):
        return GenerateImageHandler(on_image_generated=callback)

    @pytest.fixture
    def handler_no_callback(self, image_service):
        return GenerateImageHandler(image_service=image_service)

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "generate_image"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "generate_image"
        assert len(defn.description) > 0
        assert "prompt" in defn.parameters["properties"]
        assert "prompt" in defn.parameters["required"]

    @pytest.mark.asyncio
    async def test_successful_generation(self, handler, image_service, callback):
        """Successful generation returns result with edit hint."""
        result = await handler.execute({"prompt": "A cat wearing a top hat"})

        assert result.success is True
        assert "Image posted" in result.output
        assert "edit_image" in result.output  # Hint about editing
        image_service.generate.assert_called_once_with("A cat wearing a top hat")
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_receives_image_data(self, handler, callback):
        """Callback receives image bytes, prompt, and mime type."""
        await handler.execute({"prompt": "A dog"})

        callback.assert_called_once_with(b"image_bytes", "A dog", "image/png")

    @pytest.mark.asyncio
    async def test_empty_prompt_fails(self, handler):
        """Empty prompt returns error."""
        result = await handler.execute({"prompt": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_prompt_fails(self, handler):
        """Whitespace-only prompt returns error."""
        result = await handler.execute({"prompt": "   "})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_prompt_fails(self, handler):
        """Missing prompt returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_service_fails(self, handler_no_service):
        """No image service returns not configured error."""
        result = await handler_no_service.execute({"prompt": "A cat"})

        assert result.success is False
        assert "NANOBANANA_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_no_callback_fails(self, handler_no_callback):
        """No callback returns no channel error."""
        result = await handler_no_callback.execute({"prompt": "A cat"})

        assert result.success is False
        assert "no channel" in result.error.lower()

    @pytest.mark.asyncio
    async def test_generation_error_handled(self, handler, image_service):
        """Generation exceptions are handled gracefully."""
        image_service.generate.side_effect = Exception("API rate limit")

        result = await handler.execute({"prompt": "A cat"})

        assert result.success is False
        assert "API rate limit" in result.error

    @pytest.mark.asyncio
    async def test_metadata_includes_image_info(self, handler):
        """Result metadata includes image information."""
        result = await handler.execute({"prompt": "A cat"})

        assert result.metadata["prompt"] == "A cat"
        assert result.metadata["size_bytes"] == len(b"image_bytes")
        assert result.metadata["mime_type"] == "image/png"

    def test_set_service(self, handler_no_service, image_service):
        """Can set service after construction."""
        handler_no_service.set_service(image_service)
        assert handler_no_service._image_service is image_service

    def test_set_callback(self, handler_no_callback, callback):
        """Can set callback after construction."""
        handler_no_callback.set_callback(callback)
        assert handler_no_callback._on_image_generated is callback


class TestEditImageHandler:
    """Tests for EditImageHandler."""

    @pytest.fixture
    def image_service(self):
        # Start with has_chat=True (image was generated)
        return MockImageService(has_chat_value=True)

    @pytest.fixture
    def image_service_no_chat(self):
        # No previous image
        return MockImageService(has_chat_value=False)

    @pytest.fixture
    def callback(self):
        return AsyncMock(return_value="Edited image posted to #general")

    @pytest.fixture
    def handler(self, image_service, callback):
        return EditImageHandler(
            image_service=image_service,
            on_image_generated=callback,
        )

    @pytest.fixture
    def handler_no_chat(self, image_service_no_chat, callback):
        return EditImageHandler(
            image_service=image_service_no_chat,
            on_image_generated=callback,
        )

    @pytest.fixture
    def handler_no_service(self, callback):
        return EditImageHandler(on_image_generated=callback)

    @pytest.fixture
    def handler_no_callback(self, image_service):
        return EditImageHandler(image_service=image_service)

    def test_name(self, handler):
        """Handler has correct name."""
        assert handler.name == "edit_image"

    def test_definition_has_required_fields(self, handler):
        """Definition contains required fields."""
        defn = handler.definition
        assert defn.name == "edit_image"
        assert len(defn.description) > 0
        assert "instruction" in defn.parameters["properties"]
        assert "instruction" in defn.parameters["required"]

    @pytest.mark.asyncio
    async def test_successful_edit(self, handler, image_service, callback):
        """Successful edit returns result with edit continuation hint."""
        result = await handler.execute({"instruction": "Add silly hats"})

        assert result.success is True
        assert "posted" in result.output.lower()
        assert "edit_image" in result.output  # Can continue editing
        image_service.edit.assert_called_once_with("Add silly hats")
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_receives_edited_data(self, handler, callback):
        """Callback receives edited image with labeled prompt."""
        await handler.execute({"instruction": "Add hats"})

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == b"edited_bytes"
        assert "Edited:" in call_args[1]
        assert "Add hats" in call_args[1]

    @pytest.mark.asyncio
    async def test_edit_without_previous_image_fails(self, handler_no_chat):
        """Edit without previous image returns helpful error."""
        result = await handler_no_chat.execute({"instruction": "Add hats"})

        assert result.success is False
        assert "no previous image" in result.error.lower()
        assert "generate" in result.error.lower()  # Suggests what to do

    @pytest.mark.asyncio
    async def test_empty_instruction_fails(self, handler):
        """Empty instruction returns error."""
        result = await handler.execute({"instruction": ""})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_instruction_fails(self, handler):
        """Whitespace-only instruction returns error."""
        result = await handler.execute({"instruction": "   "})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_instruction_fails(self, handler):
        """Missing instruction returns error."""
        result = await handler.execute({})

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_service_fails(self, handler_no_service):
        """No image service returns not configured error."""
        result = await handler_no_service.execute({"instruction": "Add hats"})

        assert result.success is False
        assert "NANOBANANA_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_no_callback_fails(self, handler_no_callback):
        """No callback returns no channel error."""
        result = await handler_no_callback.execute({"instruction": "Add hats"})

        assert result.success is False
        assert "no channel" in result.error.lower()

    @pytest.mark.asyncio
    async def test_edit_error_handled(self, handler, image_service):
        """Edit exceptions are handled gracefully."""
        image_service.edit.side_effect = Exception("Content blocked by safety filter")

        result = await handler.execute({"instruction": "Add hats"})

        assert result.success is False
        assert "Content blocked" in result.error

    @pytest.mark.asyncio
    async def test_metadata_includes_edit_info(self, handler):
        """Result metadata includes edit information."""
        result = await handler.execute({"instruction": "Add hats"})

        assert result.metadata["instruction"] == "Add hats"
        assert result.metadata["size_bytes"] == len(b"edited_bytes")
        assert result.metadata["mime_type"] == "image/png"


class TestImageHandlerIntegration:
    """Integration tests for image handlers working together."""

    @pytest.fixture
    def image_service(self):
        return MockImageService(has_chat_value=False)

    @pytest.fixture
    def callback(self):
        return AsyncMock(return_value="Image posted")

    @pytest.fixture
    def generate_handler(self, image_service, callback):
        return GenerateImageHandler(image_service, callback)

    @pytest.fixture
    def edit_handler(self, image_service, callback):
        return EditImageHandler(image_service, callback)

    @pytest.mark.asyncio
    async def test_edit_fails_before_generate(self, edit_handler):
        """Edit fails when no image has been generated."""
        result = await edit_handler.execute({"instruction": "Add hats"})
        assert result.success is False
        assert "no previous image" in result.error.lower()

    @pytest.mark.asyncio
    async def test_edit_works_after_generate(self, generate_handler, edit_handler, image_service):
        """Edit works after an image has been generated."""
        # Generate first
        gen_result = await generate_handler.execute({"prompt": "A cat"})
        assert gen_result.success is True

        # Simulate that generate() created a chat session
        image_service.set_has_chat(True)

        # Now edit should work
        edit_result = await edit_handler.execute({"instruction": "Add hats"})
        assert edit_result.success is True

    @pytest.mark.asyncio
    async def test_multiple_edits(self, generate_handler, edit_handler, image_service):
        """Can perform multiple edits after generation."""
        # Generate
        await generate_handler.execute({"prompt": "A cat"})
        image_service.set_has_chat(True)

        # First edit
        result1 = await edit_handler.execute({"instruction": "Add a hat"})
        assert result1.success is True

        # Second edit
        result2 = await edit_handler.execute({"instruction": "Make the hat blue"})
        assert result2.success is True

        # Edit was called twice
        assert image_service.edit.call_count == 2
