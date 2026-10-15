"""Text checks shared by the live monitor and the e2e voice harness (scripts/e2e)."""

import re

# Answers that admit a lookup failed ("couldn't find it", "my weather feeds ghosted me")
GIVE_UP_PHRASES = ("couldn't find", "could not find", "can't find", "ghosted", "no luck",
                   "not sure", "don't know", "couldn't get", "no results")
# These also turn up in ordinary chit-chat, so live they only count after a web lookup
WEAK_GIVE_UP_PHRASES = ("not sure", "don't know")

# The bot asking to hear it again usually means Whisper misheard the question
UNCLEAR_PHRASES = ("didn't catch", "did not catch", "say that again", "repeat that",
                   "not sure what you mean", "what do you mean", "misheard", "didn't quite get")

# Inworld markup words that should be performed, never read out
SPOKEN_TAG_WORDS = {"whispering", "laughing", "happy", "sad", "angry", "surprised", "fearful",
                    "disgusted", "cheerful", "sigh", "laugh", "breathe", "cough", "clear throat"}
SPOKEN_LINK_MARKERS = ("http", "www", "dot com", ".com")

_TAG = re.compile(r"\[[a-z ]+\]", re.IGNORECASE)
_ANY_TAG = re.compile(r"\[[^\]]+\]")

# Formatting the prompt forbids (voice output); found in the text sent to TTS
_TEXT_ARTIFACTS = (
    ("a link", re.compile(r"https?://|\bwww\.", re.IGNORECASE)),
    ("bold/italic markers", re.compile(r"\*\*|__|(?<!\w)\*\w")),
    ("a markdown heading", re.compile(r"(?m)^\s*#{1,6}\s")),
    ("a bullet list", re.compile(r"(?m)^\s*[-*•]\s+\S")),
    ("code formatting", re.compile(r"`")),
    ("a table", re.compile(r"(?m)^\s*\|.*\|\s*$")),
    ("a 'Sources:' line", re.compile(r"(?im)^\s*sources?:")),
)


def normalize_words(text: str) -> list[str]:
    from num2words import num2words

    text = _TAG.sub(" ", text.lower())
    text = re.sub(r"(\d+)", lambda m: " " + num2words(int(m.group(1))) + " ", text)
    text = re.sub(r"[^a-z' ]+", " ", text.replace("-", " "))
    return [w.strip("'") for w in text.split() if w.strip("'")]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def overlap(a: str, b: str) -> float:
    """Share of a's words found in b."""
    wa, wb = normalize_words(a), set(normalize_words(b))
    return sum(w in wb for w in wa) / len(wa) if wa else 0.0


def give_up_phrases(answer: str, phrases: tuple[str, ...] = GIVE_UP_PHRASES) -> list[str]:
    """Give-up phrases found in an answer."""
    answer = answer.lower().replace("’", "'")
    return [p for p in phrases if p in answer]


def unclear_phrases(answer: str) -> list[str]:
    """Phrases where the bot says it didn't understand the question."""
    return give_up_phrases(answer, UNCLEAR_PHRASES)


def spoken_artifacts(spoken: str, said: str) -> list[str]:
    """Tag words or link fragments heard in the bot's audio that its text didn't say as words.

    spoken: transcript of the bot's audio; said: the text it was given (tags allowed).
    """
    spoken = spoken.lower()
    said = _ANY_TAG.sub(" ", said).lower()
    problems = [w for w in sorted(SPOKEN_TAG_WORDS) if re.search(rf"\b{w}\b", spoken)
                and not re.search(rf"\b{w}\b", said)]
    problems += [u for u in SPOKEN_LINK_MARKERS if u in spoken]
    return problems


def text_artifacts(text: str) -> list[str]:
    """Links or markdown in text meant to be spoken."""
    return [name for name, pattern in _TEXT_ARTIFACTS if pattern.search(text)]


def strip_tags(text: str) -> str:
    """Text without [tags], for comparing with what was heard."""
    return re.sub(r"\s+", " ", _ANY_TAG.sub(" ", text)).strip()
