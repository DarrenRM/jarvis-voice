"""TTS text filter to sanitize LLM output before speech synthesis.

Filters out:
- Internal LLM reasoning/thinking (chain-of-thought leakage)
- Ellipsis-only garbage fragments (ASCII and Unicode)
- Meta-commentary about responses
- Very short fragments that aren't valid speech
- Repetitive token spam
"""

import re
import logging

logger = logging.getLogger(__name__)

# Unicode ellipsis and ASCII dots pattern
ELLIPSIS_CHARS = r'[.…]'  # Both ASCII period and Unicode ellipsis U+2026

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
    r"^the answer\s*" + ELLIPSIS_CHARS,  # "The answer ..."
    r"^oops" + ELLIPSIS_CHARS,  # "Oops..."
    # GPT-OSS reasoning leakage patterns (from known bug)
    r"^the last response",
    r"^it looks" + ELLIPSIS_CHARS,
    r"^according to rules",
    r"^must (be|not|include)",
    r"^should be brief",
    r"^could add",
    r"^make sure",
    r"^thus:?$",
    r"^let's",
    r"^first[,—]",
    r"^sure!?\s*(let's|let me)",
]

# Patterns for garbage/malformed output
GARBAGE_PATTERNS = [
    r"^[\[\]\.…\?\s\*]+$",  # Only brackets, dots, ellipsis, question marks, spaces, asterisks
    r"^\[?" + ELLIPSIS_CHARS + r"+\]?$",  # Only ellipsis with optional brackets
    r"^\[[^\]]*\]\s*" + ELLIPSIS_CHARS + r"+$",  # [tag] followed by only ellipsis (no real content)
    r"^\]" + ELLIPSIS_CHARS + r"+$",  # ] followed by dots/ellipsis
    r"^chaos" + ELLIPSIS_CHARS + r"+$",  # chaos with dots
    r"^(the|we|this|that)\s*" + ELLIPSIS_CHARS + r"+$",  # Single word + ellipsis
    r"\*+" + ELLIPSIS_CHARS + r"+\*+",  # Markdown bold with ellipsis: **...** (anywhere in text)
    r"^" + ELLIPSIS_CHARS + r"+$",  # Only ellipsis/dots
    r"^\?+$",  # Only question marks
]

# Compiled patterns for efficiency
_REASONING_REGEX = [re.compile(p, re.IGNORECASE) for p in REASONING_PATTERNS]
_GARBAGE_REGEX = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]

# Minimum content length (chars) for valid speech (excludes emotion tags)
MIN_CONTENT_LENGTH = 5


def _has_word_repetition(text: str, threshold: int = 4) -> bool:
    """Check if text has excessive word repetition.

    Args:
        text: Text to check
        threshold: Number of consecutive same words to trigger

    Returns:
        True if text has word spam like "fresh fresh fresh fresh"
    """
    # Remove emotion tags first
    cleaned = re.sub(r'\[[^\]]+\]', '', text).strip().lower()
    words = cleaned.split()

    if len(words) < threshold:
        return False

    # Check for consecutive same words
    count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1

    return False


def _get_content_without_tags(text: str) -> str:
    """Remove emotion/style tags to get actual content.

    Args:
        text: Text possibly containing [tags]

    Returns:
        Text with tags removed
    """
    return re.sub(r'\[[^\]]+\]', '', text).strip()


def is_reasoning_text(text: str) -> bool:
    """Check if text appears to be internal LLM reasoning or garbage.

    Args:
        text: Text to check

    Returns:
        True if text should be filtered out, False otherwise
    """
    text = text.strip()
    if not text:
        return True  # Empty text - filter out

    # Check garbage patterns first (fastest)
    for pattern in _GARBAGE_REGEX:
        if pattern.search(text):
            return True

    # Check reasoning patterns
    for pattern in _REASONING_REGEX:
        if pattern.search(text):
            return True

    # Get content without tags
    content = _get_content_without_tags(text)

    # Filter very short fragments (unless it's just an emotion tag which is OK)
    if len(content) < MIN_CONTENT_LENGTH and len(content) > 0:
        # Allow short interjections that make sense
        if content.lower() not in {'ok', 'yes', 'no', 'hi', 'hey', 'bye', 'done', 'sure', 'wow', 'nice'}:
            return True

    # Filter text that's mostly punctuation/dots/ellipsis
    alphanumeric_count = sum(1 for c in text if c.isalnum())
    if len(text) > 3 and alphanumeric_count < len(text) * 0.3:
        return True

    # Filter word repetition spam
    if _has_word_repetition(text):
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
