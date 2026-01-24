"""Tests for conversation state machine."""

import time
import pytest

from src.bot.state_machine import ConversationState, StateMachine


class TestStateMachine:
    """Tests for StateMachine class."""

    def test_init(self):
        """Test initial state."""
        sm = StateMachine()
        assert sm.state == ConversationState.IDLE

    def test_valid_transition(self):
        """Test valid state transitions."""
        sm = StateMachine()

        # IDLE -> ACKNOWLEDGED
        assert sm.transition_to(ConversationState.ACKNOWLEDGED, "wake_word")
        assert sm.state == ConversationState.ACKNOWLEDGED

        # ACKNOWLEDGED -> LISTENING
        assert sm.transition_to(ConversationState.LISTENING, "ack_done")
        assert sm.state == ConversationState.LISTENING

        # LISTENING -> PROCESSING
        assert sm.transition_to(ConversationState.PROCESSING, "speech_ended")
        assert sm.state == ConversationState.PROCESSING

        # PROCESSING -> SPEAKING
        assert sm.transition_to(ConversationState.SPEAKING, "claude_responding")
        assert sm.state == ConversationState.SPEAKING

        # SPEAKING -> IDLE
        assert sm.transition_to(ConversationState.IDLE, "done")
        assert sm.state == ConversationState.IDLE

    def test_invalid_transition(self):
        """Test invalid state transitions."""
        sm = StateMachine()

        # Cannot go from IDLE directly to SPEAKING
        assert not sm.transition_to(ConversationState.SPEAKING, "invalid")
        assert sm.state == ConversationState.IDLE

        # Cannot go from IDLE directly to PROCESSING
        assert not sm.transition_to(ConversationState.PROCESSING, "invalid")
        assert sm.state == ConversationState.IDLE

    def test_interrupt_transition(self):
        """Test interrupt during speaking."""
        sm = StateMachine()

        # Get to SPEAKING state
        sm.transition_to(ConversationState.ACKNOWLEDGED, "wake")
        sm.transition_to(ConversationState.LISTENING, "ack")
        sm.transition_to(ConversationState.PROCESSING, "speech")
        sm.transition_to(ConversationState.SPEAKING, "claude")

        # Interrupt -> back to LISTENING
        assert sm.transition_to(ConversationState.LISTENING, "interrupted")
        assert sm.state == ConversationState.LISTENING

    def test_time_in_state(self):
        """Test time tracking in state."""
        sm = StateMachine()

        time.sleep(0.1)
        assert sm.time_in_state >= 0.1

        sm.transition_to(ConversationState.ACKNOWLEDGED, "wake")
        assert sm.time_in_state < 0.1

    def test_timeout(self):
        """Test state timeout detection."""
        sm = StateMachine()
        sm.transition_to(ConversationState.ACKNOWLEDGED, "wake")

        # Mock timeout by modifying internal state
        sm._state_entered_at = time.time() - 10.0  # 10 seconds ago

        assert sm.check_timeout()
        assert sm.state == ConversationState.IDLE

    def test_no_timeout_idle(self):
        """Test IDLE has no timeout."""
        sm = StateMachine()
        sm._state_entered_at = time.time() - 1000.0  # Long time ago

        assert not sm.check_timeout()
        assert sm.state == ConversationState.IDLE

    def test_reset(self):
        """Test state reset."""
        sm = StateMachine()

        sm.transition_to(ConversationState.ACKNOWLEDGED, "wake")
        sm.transition_to(ConversationState.LISTENING, "ack")

        sm.reset()
        assert sm.state == ConversationState.IDLE

    def test_callback(self):
        """Test state entry callbacks."""
        sm = StateMachine()
        callback_called = []

        def on_listening():
            callback_called.append("listening")

        sm.on_state(ConversationState.LISTENING, on_listening)

        sm.transition_to(ConversationState.ACKNOWLEDGED, "wake")
        assert len(callback_called) == 0

        sm.transition_to(ConversationState.LISTENING, "ack")
        assert callback_called == ["listening"]

    def test_history(self):
        """Test transition history."""
        sm = StateMachine()

        sm.transition_to(ConversationState.ACKNOWLEDGED, "wake")
        sm.transition_to(ConversationState.LISTENING, "ack")

        assert len(sm._history) == 2
        assert sm._history[0].from_state == ConversationState.IDLE
        assert sm._history[0].to_state == ConversationState.ACKNOWLEDGED
        assert sm._history[0].reason == "wake"
