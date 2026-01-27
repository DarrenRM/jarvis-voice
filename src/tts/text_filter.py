"""TTS text filter to sanitize LLM output before speech synthesis.

Filters out:
- Internal LLM reasoning/thinking (chain-of-thought leakage)
- Ellipsis-only garbage fragments
- Meta-commentary about responses
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that indicate internal reasoning (case-insensitive)
REASONING_PATTERNS = [
    r"^the assistant",
    r"^we (need|should|have|must)",
    r"^(need|ensure|provide|use|skip|give)",
    r"^(probably|perhaps|maybe) due to",
    r"^(the user|user) (requested|asked|wants)",
    r"^(this|that) (is|was|will|should|needs)",
    r"^(following|use) (format|tags|pattern)",
    r"^no extra",
    r"^(each|every) sentence",
    r"^brief (answer|response)",
    r"^(format|formatting):",
    r"^(step|steps) \d+",
    r"^\[?\.+\]?$",  # Only ellipsis/dots with optional brackets
    r"^\[.*\.\.\.\s*$",  # Starts with bracket, ends with ellipsis
    r"^\]\.+$",  # ] followed by dots
    r"^chaos\.+$",  # chaos with dots
]

# Compiled patterns for efficiency
_REASONING_REGEX = [re.compile(p, re.IGNORECASE) for p in REASONING_PATTERNS]


def is_reasoning_text(text: str) -> bool:
    """Check if text appears to be internal LLM reasoning.

    Args:
        text: Text to check

    Returns:
        True if text looks like internal reasoning, False otherwise
    """
    text = text.strip()
    if not text:
        return True  # Empty text is "reasoning" (filter it out)

    # Check against reasoning patterns
    for pattern in _REASONING_REGEX:
        if pattern.search(text):
            return True

    # Filter text that's mostly punctuation/dots
    alphanumeric_count = sum(1 for c in text if c.isalnum())
    if len(text) > 3 and alphanumeric_count < len(text) * 0.3:
        return True

    return False


def filter_for_tts(text: str) -> str:
    """Filter text before sending to TTS.

    Removes lines that appear to be internal reasoning or garbage.

    Args:
        text: Raw text from LLM response

    Returns:
        Filtered text suitable for TTS
    """
    if not text:
        return ""

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_reasoning_text(line):
            logger.debug(f"[TTS-FILTER] Filtered reasoning: {line[:80]}...")
            continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def filter_chunk_for_tts(chunk: str, buffer: str = "") -> tuple[str, str]:
    """Filter a streaming chunk, handling partial lines.

    For streaming use case where we receive partial text.

    Args:
        chunk: New text chunk
        buffer: Accumulated buffer from previous chunks

    Returns:
        Tuple of (text_to_send_to_tts, new_buffer)
    """
    combined = buffer + chunk

    # If we have a newline, we can process complete lines
    if '\n' in combined:
        lines = combined.split('\n')
        # Last part might be incomplete
        complete_lines = lines[:-1]
        incomplete = lines[-1]

        # Filter complete lines
        filtered = []
        for line in complete_lines:
            line = line.strip()
            if line and not is_reasoning_text(line):
                filtered.append(line)

        text_to_send = '\n'.join(filtered)
        if text_to_send:
            text_to_send += '\n'

        return text_to_send, incomplete

    # No newline yet - accumulate in buffer
    # But if buffer gets too long without newlines, check if it's garbage
    if len(combined) > 100:
        # Check if it looks like garbage
        if is_reasoning_text(combined):
            logger.debug(f"[TTS-FILTER] Discarding long garbage chunk: {combined[:50]}...")
            return "", ""

    return "", combined
