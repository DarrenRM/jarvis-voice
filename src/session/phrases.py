"""Phrase constants and selection helpers for the voice assistant.

Contains acknowledgment phrases (played when wake word detected) and
thinking phrases (played while processing a request).
"""

import random
from typing import Optional

# Acknowledgment phrases (played when wake word detected)
ACK_PHRASES: list[str] = [
    "[happy] Yes?",
    "[surprised] Hmm?",
    "What's up?",
    "[happy] I'm listening...",
    "Go ahead...",
    "[breathe] I'm busy sorry...",
    "Yea, no, yea. [clear_throat] No.",
    "Yo?",
    "Sup?",
    "[happy] I'm here!",
]

# Generic thinking phrases (used when speaker unknown)
THINKING_PHRASES: list[str] = [
    "[happy] One second...",
    "One moment...",
    "Let me check...",
    "Google it yourself... [breathe] Nah, I'll do it...",
    "Hmm, let me see...",
    "[happy] Just a moment...",
]

# Personalized thinking phrases with {name} placeholder (used when speaker is known)
THINKING_PHRASES_PERSONAL: list[str] = [
    "[happy] One second, {name}...",
    "One moment, {name}...",
    "Let me check, {name}...",
    "[laughing] Sure {name}, I'll do all the work...",
    "Google it yourself, {name}... [breathe] Nah, I'll do it...",
    "Hmm, let me see, {name}...",
    "[happy] Just a moment, {name}...",
    "Hold on, {name}...",
    "Give me a sec, {name}...",
]


def get_random_ack() -> str:
    """Get a random acknowledgment phrase.

    Returns:
        A random phrase from ACK_PHRASES.
    """
    return random.choice(ACK_PHRASES)


def get_thinking_phrase(speaker_name: Optional[str] = None) -> str:
    """Get a thinking phrase, optionally personalized with speaker name.

    If a speaker name is provided, uses personalized phrases with the name
    inserted. Otherwise, uses generic thinking phrases.

    Args:
        speaker_name: Optional name of the current speaker.

    Returns:
        A thinking phrase, personalized if name provided.
    """
    if speaker_name:
        phrase = random.choice(THINKING_PHRASES_PERSONAL)
        return phrase.format(name=speaker_name)
    return random.choice(THINKING_PHRASES)
