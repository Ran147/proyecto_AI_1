"""Map transcribed text to a model (keywords plus fuzzy matching).

Pure functions: they do not know the registry; the orchestrator passes the
models in.
"""
import os
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

from ..models.base import BaseModel

# Minimum similarity to accept a fuzzy keyword or wake-word match.
KEYWORD_THRESHOLD = 0.85
WAKE_WORD_THRESHOLD = 0.75


def normalize(text: str) -> str:
    """Lowercase, strip accents and keep only letters, digits and spaces."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def strip_wake_word(text: str) -> tuple[str, bool]:
    """Remove a leading 'JarvisTEC', tolerating transcription variants.

    Accepts forms such as 'jarvis tec' or 'yarvis tek'. Returns
    ``(text_without_wake_word, wake_word_found)``.
    """
    wake_word = normalize(os.getenv("WAKE_WORD", "jarvistec"))
    normalized = normalize(text)
    for variant in (wake_word, "jarvis tec", "jarvis"):
        if normalized.startswith(variant):
            return normalized[len(variant):].strip(), True
    words = normalized.split()
    # One or two leading words glued together: "yarvis tek" -> "yarvistek".
    for count in (2, 1):
        if len(words) >= count:
            head = "".join(words[:count])
            ratio = SequenceMatcher(None, head, wake_word).ratio()
            if ratio >= WAKE_WORD_THRESHOLD:
                return " ".join(words[count:]), True
    return normalized, False


def _score(keyword: str, text: str) -> float:
    normalized_keyword = normalize(keyword)
    if normalized_keyword and normalized_keyword in text:
        # Exact containment wins; longer phrases win ties.
        return 1.0 + len(normalized_keyword) / 100
    words = text.split()
    size = len(normalized_keyword.split())
    best = 0.0
    for start in range(max(1, len(words) - size + 1)):
        window = " ".join(words[start:start + size])
        ratio = SequenceMatcher(None, normalized_keyword, window).ratio()
        best = max(best, ratio)
    return best if best >= KEYWORD_THRESHOLD else 0.0


def match_command(
    text: str, models: Iterable[BaseModel]
) -> tuple[BaseModel | None, float]:
    """Return the best matching model for ``text`` and its score."""
    cleaned, _ = strip_wake_word(text)
    best_model, best_score = None, 0.0
    for model in models:
        for keyword in model.keywords:
            score = _score(keyword, cleaned)
            if score > best_score:
                best_model, best_score = model, score
    return best_model, best_score
