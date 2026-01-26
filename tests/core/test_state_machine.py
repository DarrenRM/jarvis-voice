"""Tests for SessionStateMachine."""

import pytest
from src.core.state_machine import (
    SessionState,
    SessionStateMachine,
    TransitionContext,
    InvalidTransitionError,
    VALID_TRANSITIONS,
)


class TestSessionState:
    """Tests for SessionState enum."""

    def test_states_exist(self):
        """All expected states exist."""
        assert SessionState.IDLE is not None
        assert SessionState.LISTENING is not None
        assert SessionState.PROCESSING is not None

    def test_states_are_unique(self):
        """Each state has a unique value."""
        values = [s.value for s in SessionState]
        assert len(values) == len(set(values))


class TestValidTransitions:
    """Tests for VALID_TRANSITIONS mapping."""

    def test_idle_can_transition_to_listening(self):
        assert SessionState.LISTENING in VALID_TRANSITIONS[SessionState.IDLE]

    def test_idle_cannot_transition_to_processing(self):
        assert SessionState.PROCESSING not in VALID_TRANSITIONS[SessionState.IDLE]

    def test_listening_can_transition_to_processing(self):
        assert SessionState.PROCESSING in VALID_TRANSITIONS[SessionState.LISTENING]

    def test_listening_can_transition_to_idle(self):
        assert SessionState.IDLE in VALID_TRANSITIONS[SessionState.LISTENING]

    def test_processing_can_transition_to_idle(self):
        assert SessionState.IDLE in VALID_TRANSITIONS[SessionState.PROCESSING]

    def test_processing_can_transition_to_listening_for_barge_in(self):
        assert SessionState.LISTENING in VALID_TRANSITIONS[SessionState.PROCESSING]


class TestSessionStateMachine:
    """Tests for SessionStateMachine class."""

    @pytest.fixture
    def sm(self):
        """Fresh state machine instance."""
        return SessionStateMachine()

    def test_initial_state_is_idle(self, sm):
        """State machine starts in IDLE state."""
        assert sm.state == SessionState.IDLE

    def test_valid_transition_succeeds(self, sm):
        """Valid transition changes state."""
        ctx = sm.transition_to(SessionState.LISTENING, trigger="wake_word")
        assert sm.state == SessionState.LISTENING
        assert ctx.from_state == SessionState.IDLE
        assert ctx.to_state == SessionState.LISTENING
        assert ctx.trigger == "wake_word"

    def test_invalid_transition_raises_error(self, sm):
        """Invalid transition raises InvalidTransitionError."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition_to(SessionState.PROCESSING)

        assert exc_info.value.from_state == SessionState.IDLE
        assert exc_info.value.to_state == SessionState.PROCESSING
        assert "IDLE" in str(exc_info.value)
        assert "PROCESSING" in str(exc_info.value)

    def test_invalid_transition_does_not_change_state(self, sm):
        """State unchanged after invalid transition attempt."""
        try:
            sm.transition_to(SessionState.PROCESSING)
        except InvalidTransitionError:
            pass

        assert sm.state == SessionState.IDLE

    def test_transition_with_speaker_id(self, sm):
        """Transition can include speaker_id."""
        ctx = sm.transition_to(
            SessionState.LISTENING,
            trigger="wake_word",
            speaker_id=12345,
        )
        assert ctx.speaker_id == 12345

    def test_transition_with_metadata(self, sm):
        """Transition can include arbitrary metadata."""
        ctx = sm.transition_to(
            SessionState.LISTENING,
            trigger="wake_word",
            custom_data="test",
            count=42,
        )
        assert ctx.metadata["custom_data"] == "test"
        assert ctx.metadata["count"] == 42

    def test_can_transition_to_returns_true_for_valid(self, sm):
        """can_transition_to returns True for valid transitions."""
        assert sm.can_transition_to(SessionState.LISTENING) is True

    def test_can_transition_to_returns_false_for_invalid(self, sm):
        """can_transition_to returns False for invalid transitions."""
        assert sm.can_transition_to(SessionState.PROCESSING) is False

    def test_reset_returns_to_idle(self, sm):
        """reset() returns state to IDLE."""
        sm.transition_to(SessionState.LISTENING, trigger="test")
        sm.transition_to(SessionState.PROCESSING, trigger="test")
        sm.reset()
        assert sm.state == SessionState.IDLE


class TestStateMachineHooks:
    """Tests for state machine hooks."""

    @pytest.fixture
    def sm(self):
        return SessionStateMachine()

    def test_on_enter_hook_called(self, sm):
        """on_enter hook is called when entering state."""
        called = []

        def hook(ctx):
            called.append(("enter", ctx.to_state))

        sm.on_enter(SessionState.LISTENING, hook)
        sm.transition_to(SessionState.LISTENING, trigger="test")

        assert ("enter", SessionState.LISTENING) in called

    def test_on_exit_hook_called(self, sm):
        """on_exit hook is called when exiting state."""
        called = []

        def hook(ctx):
            called.append(("exit", ctx.from_state))

        sm.on_exit(SessionState.IDLE, hook)
        sm.transition_to(SessionState.LISTENING, trigger="test")

        assert ("exit", SessionState.IDLE) in called

    def test_on_transition_hook_called(self, sm):
        """on_transition hook is called on any transition."""
        called = []

        def hook(ctx):
            called.append(("transition", ctx.from_state, ctx.to_state))

        sm.on_transition(hook)
        sm.transition_to(SessionState.LISTENING, trigger="test")

        assert ("transition", SessionState.IDLE, SessionState.LISTENING) in called

    def test_hook_order_exit_then_enter(self, sm):
        """Exit hooks run before enter hooks."""
        order = []

        sm.on_exit(SessionState.IDLE, lambda ctx: order.append("exit"))
        sm.on_enter(SessionState.LISTENING, lambda ctx: order.append("enter"))

        sm.transition_to(SessionState.LISTENING, trigger="test")

        assert order == ["exit", "enter"]

    def test_multiple_hooks_all_called(self, sm):
        """Multiple hooks for same state all called."""
        called = []

        sm.on_enter(SessionState.LISTENING, lambda ctx: called.append("hook1"))
        sm.on_enter(SessionState.LISTENING, lambda ctx: called.append("hook2"))

        sm.transition_to(SessionState.LISTENING, trigger="test")

        assert "hook1" in called
        assert "hook2" in called

    def test_hook_receives_context(self, sm):
        """Hook receives full TransitionContext."""
        received_ctx = []

        def hook(ctx):
            received_ctx.append(ctx)

        sm.on_enter(SessionState.LISTENING, hook)
        sm.transition_to(
            SessionState.LISTENING,
            trigger="wake_word",
            speaker_id=123,
            extra="data",
        )

        ctx = received_ctx[0]
        assert ctx.from_state == SessionState.IDLE
        assert ctx.to_state == SessionState.LISTENING
        assert ctx.trigger == "wake_word"
        assert ctx.speaker_id == 123
        assert ctx.metadata["extra"] == "data"

    def test_hook_error_does_not_prevent_transition(self, sm):
        """Error in hook doesn't prevent state transition."""

        def bad_hook(ctx):
            raise RuntimeError("Hook error")

        sm.on_enter(SessionState.LISTENING, bad_hook)

        # Should not raise, transition should still happen
        sm.transition_to(SessionState.LISTENING, trigger="test")
        assert sm.state == SessionState.LISTENING

    def test_reset_does_not_trigger_hooks(self, sm):
        """reset() does not trigger any hooks."""
        called = []

        sm.on_exit(SessionState.IDLE, lambda ctx: called.append("exit_idle"))
        sm.on_enter(SessionState.LISTENING, lambda ctx: called.append("enter_listening"))

        sm.transition_to(SessionState.LISTENING, trigger="test")
        called.clear()

        sm.on_exit(SessionState.LISTENING, lambda ctx: called.append("exit_listening"))
        sm.on_enter(SessionState.IDLE, lambda ctx: called.append("enter_idle"))

        sm.reset()

        assert called == []


class TestFullTransitionSequence:
    """Integration tests for complete state sequences."""

    @pytest.fixture
    def sm(self):
        return SessionStateMachine()

    def test_normal_conversation_flow(self, sm):
        """Test IDLE -> LISTENING -> PROCESSING -> IDLE flow."""
        assert sm.state == SessionState.IDLE

        sm.transition_to(SessionState.LISTENING, trigger="wake_word")
        assert sm.state == SessionState.LISTENING

        sm.transition_to(SessionState.PROCESSING, trigger="speech_captured")
        assert sm.state == SessionState.PROCESSING

        sm.transition_to(SessionState.IDLE, trigger="response_complete")
        assert sm.state == SessionState.IDLE

    def test_listening_timeout_flow(self, sm):
        """Test IDLE -> LISTENING -> IDLE (timeout) flow."""
        sm.transition_to(SessionState.LISTENING, trigger="wake_word")
        sm.transition_to(SessionState.IDLE, trigger="timeout")
        assert sm.state == SessionState.IDLE

    def test_barge_in_flow(self, sm):
        """Test barge-in: PROCESSING -> LISTENING (interrupt)."""
        sm.transition_to(SessionState.LISTENING, trigger="wake_word")
        sm.transition_to(SessionState.PROCESSING, trigger="speech_captured")
        sm.transition_to(SessionState.LISTENING, trigger="barge_in")
        assert sm.state == SessionState.LISTENING
