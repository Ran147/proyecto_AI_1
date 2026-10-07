"""Map transcribed text to a model (keyword phrases plus typo tolerance).

Pure functions: they do not know the registry; the orchestrator passes the
models in.

Matching rules:

- A keyword matches exactly when its words appear in the command as whole
  words, in order and next to each other ("car price" inside "what is the
  car price today"). Substrings of longer words never match.
- Otherwise a keyword matches approximately when every one of its words has
  a close counterpart, in the same order, so that a speech recognizer's slip
  ("avocato price") is forgiven but a different topic ("gold price") is not:
  each word must be similar on its own, and short words must be identical.
"""
import os
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

from ..models.base import BaseModel

# Minimum similarity between a keyword word and the word heard in its place.
WORD_THRESHOLD = 0.85
# Words up to this length must match exactly: one changed letter in a short
# word usually makes a different word ("car" / "bar").
SHORT_WORD = 4
# Minimum similarity to accept a mangled wake word.
WAKE_WORD_THRESHOLD = 0.75
# What speech recognizers write for the "TEC" of "JarvisTEC".
WAKE_WORD_TAILS = {"tec", "tech", "tek", "teck", "tick", "deck", "tack"}


def normalize(text: str) -> str:
    """Lowercase, strip accents and keep only letters, digits and spaces."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def strip_wake_word(text: str) -> tuple[str, bool]:
    """Remove a leading 'JarvisTEC', tolerating transcription variants.

    Accepts forms such as 'jarvis tec', 'jarvis tech' or 'yarvis tek', with
    an optional greeting before it ('hey jarvis'). Returns
    ``(text_without_wake_word, wake_word_found)``.
    """
    wake_word = normalize(os.getenv("WAKE_WORD") or "jarvistec")
    words = normalize(text).split()
    greetings = ("hey", "ok", "okay", "hi", "hello", "hola", "oye")
    if words and words[0] in greetings:
        words = words[1:]
    if not words:
        return "", False
    if words[0] == wake_word:
        return " ".join(words[1:]), True
    if words[0] == "jarvis":
        rest = words[1:]
        if rest and rest[0] in WAKE_WORD_TAILS:
            rest = rest[1:]
        return " ".join(rest), True
    # One or two leading words glued together: "yarvis tek" -> "yarvistek".
    for count in (2, 1):
        if len(words) >= count:
            head = "".join(words[:count])
            ratio = SequenceMatcher(None, head, wake_word).ratio()
            if ratio >= WAKE_WORD_THRESHOLD:
                return " ".join(words[count:]), True
    return " ".join(words), False


def _word_similarity(expected: str, heard: str) -> float:
    if expected == heard:
        return 1.0
    if min(len(expected), len(heard)) <= SHORT_WORD:
        return 0.0
    return SequenceMatcher(None, expected, heard).ratio()


def _score(keyword: str, words: list[str]) -> float:
    """Score how well ``keyword`` is said somewhere in ``words``."""
    expected = normalize(keyword).split()
    size = len(expected)
    if not expected or size > len(words):
        return 0.0
    best = 0.0
    for start in range(len(words) - size + 1):
        ratios = [
            _word_similarity(word, heard)
            for word, heard in zip(expected, words[start:start + size])
        ]
        if min(ratios) < WORD_THRESHOLD:
            continue
        mean = sum(ratios) / size
        if mean == 1.0:
            # Exact phrase: beats any approximate match; longer phrases win.
            return 1.0 + len(" ".join(expected)) / 100
        best = max(best, mean)
    return best


def match_command(
    text: str, models: Iterable[BaseModel]
) -> tuple[BaseModel | None, float]:
    """Return the best matching model for ``text`` and its score."""
    cleaned, _ = strip_wake_word(text)
    words = cleaned.split()
    best_model, best_score = None, 0.0
    for model in models:
        for keyword in model.phrases():
            score = _score(keyword, words)
            if score > best_score:
                best_model, best_score = model, score
    return best_model, best_score
