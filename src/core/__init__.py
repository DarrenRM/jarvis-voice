"""Core abstractions for voice session management."""

from .state_machine import (
    SessionState,
    SessionStateMachine,
    TransitionContext,
    InvalidTransitionError,
)

__all__ = [
    "SessionState",
    "SessionStateMachine",
    "TransitionContext",
    "InvalidTransitionError",
]
