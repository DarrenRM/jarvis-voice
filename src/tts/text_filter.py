"""TTS text filter to sanitize LLM output before speech synthesis.

Filters out:
- Internal LLM reasoning/thinking (chain-of-thought leakage)
- Ellipsis-only garbage fragments (ASCII and Unicode)
- Meta-commentary about responses
- Very short fragments that aren't valid speech
- Repetitive token spam

Also normalizes:
- Numbers to words (400 -> "four hundred")
- Number ranges (400-600 -> "four hundred to six hundred")
"""

import re
import logging

try:
    from num2words import num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False

logger = logging.getLogger(__name__)


# Pattern to fix missing spaces before numbers (LLM tokenization issue)
# Matches: "about2,500" -> "about 2,500", "to3,500" -> "to 3,500"
MISSING_SPACE_BEFORE_NUMBER = re.compile(
    r'([a-zA-Z])(\d+(?:,\d{3})*(?:\.\d+)?)'
)

# Number range pattern - matches various dash types between numbers
# Handles: 400-600, 400‑600 (non-breaking hyphen), 400–600 (en-dash), 400—600 (em-dash)
NUMBER_RANGE_PATTERN = re.compile(
    r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*[-‑–—]\s*(\d+(?:,\d{3})*(?:\.\d+)?)'
)

# Numbers with plus sign (e.g., "25,000+")
NUMBER_PLUS_PATTERN = re.compile(
    r'(\d+(?:,\d{3})*(?:\.\d+)?)\+'
)

# Standalone numbers (not part of a range, not followed by certain units that TTS handles well)
# Matches both comma-separated (1,234) and plain numbers (1234)
STANDALONE_NUMBER_PATTERN = re.compile(
    r'(?<![0-9.,-‑–—])(\d+(?:,\d{3})*(?:\.\d+)?)(?![0-9.,-‑–—%+])'
)


def _number_to_words(num_str: str) -> str:
    """Convert a number string to words.

    Args:
        num_str: Number as string (e.g., "400", "1,234", "3.14")

    Returns:
        Number as words (e.g., "four hundred", "one thousand two hundred thirty-four")
    """
    if not HAS_NUM2WORDS:
        return num_str

    try:
        # Remove commas for parsing
        clean_num = num_str.replace(',', '')

        # Check if it's a decimal
        if '.' in clean_num:
            num = float(clean_num)
        else:
            num = int(clean_num)

        return num2words(num)
    except (ValueError, TypeError):
        return num_str


def normalize_numbers_for_tts(text: str) -> str:
    """Normalize numbers in text for better TTS pronunciation.

    Converts:
    - Number ranges: "400-600" -> "four hundred to six hundred"
    - Numbers with plus: "25,000+" -> "twenty-five thousand plus"
    - Standalone numbers >= 100: "The answer is 400" -> "The answer is four hundred"

    Preserves:
    - Small numbers < 100: "42" stays as is (TTS handles these well)
    - Percentages: "50%" stays as is (TTS handles these)
    - Times: "3:00" stays as is
    - Dates in certain formats

    Args:
        text: Text containing numbers

    Returns:
        Text with numbers converted to words
    """
    # First, fix missing spaces before numbers (LLM tokenization issue)
    # "about2,500" -> "about 2,500"
    text = MISSING_SPACE_BEFORE_NUMBER.sub(r'\1 \2', text)

    if not HAS_NUM2WORDS:
        logger.debug("num2words not installed, skipping number normalization")
        return text

    # Handle number ranges (do this before standalone numbers)
    def replace_range(match: re.Match) -> str:
        num1 = match.group(1)
        num2 = match.group(2)
        words1 = _number_to_words(num1)
        words2 = _number_to_words(num2)
        return f"{words1} to {words2}"

    text = NUMBER_RANGE_PATTERN.sub(replace_range, text)

    # Handle numbers with plus sign (e.g., "25,000+" -> "twenty-five thousand plus")
    def replace_plus(match: re.Match) -> str:
        num_str = match.group(1)
        words = _number_to_words(num_str)
        return f"{words} plus"

    text = NUMBER_PLUS_PATTERN.sub(replace_plus, text)

    # Then handle standalone numbers (3+ digits to avoid converting small numbers unnecessarily)
    def replace_standalone(match: re.Match) -> str:
        num_str = match.group(1)
        # Only convert numbers >= 100 (smaller numbers TTS usually handles fine)
        clean_num = num_str.replace(',', '')
        try:
            if '.' not in clean_num and int(clean_num) < 100:
                return num_str
        except ValueError:
            return num_str
        return _number_to_words(num_str)

    text = STANDALONE_NUMBER_PATTERN.sub(replace_standalone, text)

    return text


# Unicode ellipsis and ASCII dots pattern
ELLIPSIS_CHARS = r'[.…]'  # Both ASCII period and Unicode ellipsis U+2026

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
_GARBAGE_REGEX = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]

# Minimum content length (chars) for valid speech (excludes emotion tags)
MIN_CONTENT_LENGTH = 5


_URL = re.compile(
    r"(?:https?://|www\.)\S+"  # scheme or www links
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|gg|io|co|dev|app|wiki|tv)/\S*",  # domain/path
    re.IGNORECASE,
)


def strip_urls_for_tts(text: str) -> str:
    """Replace links with "the link" so they're never read aloud.

    The prompt already says not to speak URLs; this is the backstop. Bare site
    names without a path ("wiki.gg") are left alone since they're speakable.
    """
    def replace(match: re.Match) -> str:
        url = match.group(0)
        trail = ""
        while url and url[-1] in ".,!?;:)]'\"":
            trail = url[-1] + trail
            url = url[:-1]
        return "the link" + trail

    return _URL.sub(replace, text)


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

    # Check garbage patterns
    for pattern in _GARBAGE_REGEX:
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
    Normalizes numbers to words for better pronunciation.

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

    result = '\n'.join(filtered_lines)

    # Normalize numbers to words for better TTS pronunciation
    result = normalize_numbers_for_tts(result)

    return result


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
