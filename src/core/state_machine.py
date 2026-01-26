"""State machine for voice session management.

Provides centralized state management with:
- Validated state transitions
- Entry/exit/transition hooks
- Thread-safe state access
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Valid states for a voice session."""

    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()


# Valid state transitions
# IDLE -> LISTENING (wake word detected)
# LISTENING -> PROCESSING (speech captured)
# LISTENING -> IDLE (timeout, no speech)
# PROCESSING -> IDLE (response complete)
# PROCESSING -> LISTENING (barge-in interrupt)
VALID_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE: {SessionState.LISTENING},
    SessionState.LISTENING: {SessionState.PROCESSING, SessionState.IDLE},
    SessionState.PROCESSING: {SessionState.IDLE, SessionState.LISTENING},
}


@dataclass
class TransitionContext:
    """Context passed to transition hooks.

    Attributes:
        from_state: The state being exited
        to_state: The state being entered
        trigger: What caused this transition (for logging/debugging)
        speaker_id: Optional user ID if speaker-related
        metadata: Additional context data
    """

    from_state: SessionState
    to_state: SessionState
    trigger: str
    speaker_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class InvalidTransitionError(Exception):
    """Raised when attempting an illegal state transition."""

    def __init__(self, from_state: SessionState, to_state: SessionState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Cannot transition from {from_state.name} to {to_state.name}"
        )


class SessionStateMachine:
    """Manages session state with validated transitions and hooks.

    Legal transitions:
        IDLE -> LISTENING (wake word detected)
        LISTENING -> PROCESSING (speech captured)
        LISTENING -> IDLE (timeout, no speech)
        PROCESSING -> IDLE (response complete)
        PROCESSING -> LISTENING (barge-in interrupt)

    Usage:
        sm = SessionStateMachine()

        # Register hooks
        sm.on_enter(SessionState.LISTENING, lambda ctx: print("Started listening"))
        sm.on_exit(SessionState.PROCESSING, lambda ctx: cleanup())

        # Transition with validation
        try:
            sm.transition_to(SessionState.LISTENING, trigger="wake_word", speaker_id=123)
        except InvalidTransitionError:
            logger.warning("Invalid transition attempted")
    """

    def __init__(self):
        self._state = SessionState.IDLE
        self._on_enter_hooks: dict[SessionState, list[Callable[[TransitionContext], None]]] = (
            defaultdict(list)
        )
        self._on_exit_hooks: dict[SessionState, list[Callable[[TransitionContext], None]]] = (
            defaultdict(list)
        )
        self._on_transition_hooks: list[Callable[[TransitionContext], None]] = []

    @property
    def state(self) -> SessionState:
        """Current state (read-only)."""
        return self._state

    def transition_to(
        self,
        target: SessionState,
        trigger: str = "manual",
        speaker_id: Optional[int] = None,
        **metadata,
    ) -> TransitionContext:
        """Transition to a new state.

        Args:
            target: Destination state
            trigger: What caused this transition (for logging/debugging)
            speaker_id: Optional user ID if this transition is speaker-related
            **metadata: Additional context for hooks

        Returns:
            TransitionContext with full transition details

        Raises:
            InvalidTransitionError: If transition is not allowed
        """
        if target not in VALID_TRANSITIONS[self._state]:
            raise InvalidTransitionError(self._state, target)

        context = TransitionContext(
            from_state=self._state,
            to_state=target,
            trigger=trigger,
            speaker_id=speaker_id,
            metadata=metadata,
        )

        logger.debug(
            f"State transition: {self._state.name} -> {target.name} "
            f"(trigger={trigger}, speaker_id={speaker_id})"
        )

        # Execute hooks
        self._run_exit_hooks(context)
        self._state = target
        self._run_enter_hooks(context)
        self._run_transition_hooks(context)

        return context

    def on_enter(
        self, state: SessionState, hook: Callable[[TransitionContext], None]
    ) -> None:
        """Register a hook to run when entering a state.

        Args:
            state: The state to watch for entry
            hook: Function to call with TransitionContext when state is entered
        """
        self._on_enter_hooks[state].append(hook)

    def on_exit(
        self, state: SessionState, hook: Callable[[TransitionContext], None]
    ) -> None:
        """Register a hook to run when exiting a state.

        Args:
            state: The state to watch for exit
            hook: Function to call with TransitionContext when state is exited
        """
        self._on_exit_hooks[state].append(hook)

    def on_transition(self, hook: Callable[[TransitionContext], None]) -> None:
        """Register a hook to run on any transition.

        Args:
            hook: Function to call with TransitionContext on any state change
        """
        self._on_transition_hooks.append(hook)

    def can_transition_to(self, target: SessionState) -> bool:
        """Check if transition to target state is valid from current state.

        Args:
            target: The state to check

        Returns:
            True if transition is allowed, False otherwise
        """
        return target in VALID_TRANSITIONS[self._state]

    def reset(self) -> None:
        """Reset to IDLE state without triggering hooks.

        Use for error recovery or cleanup scenarios where hooks
        should not be executed.
        """
        logger.debug(f"State machine reset from {self._state.name} to IDLE")
        self._state = SessionState.IDLE

    def _run_exit_hooks(self, context: TransitionContext) -> None:
        """Run all exit hooks for the current state."""
        for hook in self._on_exit_hooks[context.from_state]:
            try:
                hook(context)
            except Exception as e:
                logger.error(f"Error in exit hook for {context.from_state.name}: {e}")

    def _run_enter_hooks(self, context: TransitionContext) -> None:
        """Run all enter hooks for the new state."""
        for hook in self._on_enter_hooks[context.to_state]:
            try:
                hook(context)
            except Exception as e:
                logger.error(f"Error in enter hook for {context.to_state.name}: {e}")

    def _run_transition_hooks(self, context: TransitionContext) -> None:
        """Run all transition hooks."""
        for hook in self._on_transition_hooks:
            try:
                hook(context)
            except Exception as e:
                logger.error(f"Error in transition hook: {e}")
