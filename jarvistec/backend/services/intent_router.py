"""Convierte texto transcrito en un modelo a ejecutar (palabras clave + coincidencia aproximada).

Funciones puras: no conocen el registro. El orquestador les pasa los modelos.
"""
import os
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

from ..models.base import BaseModel


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def strip_wake_word(text: str) -> tuple[str, bool]:
    """Quita 'JarvisTEC' del inicio, tolerando variantes de la transcripción
    ('jarvis tec', 'yarvis tek'...). Devuelve (texto_sin_palabra, había_palabra)."""
    wake = normalize(os.getenv("WAKE_WORD", "jarvistec"))
    t = normalize(text)
    for variant in (wake, "jarvis tec", "jarvis"):
        if t.startswith(variant):
            return t[len(variant):].strip(), True
    words = t.split()
    for n in (2, 1):  # una o dos palabras iniciales pegadas: "yarvis tek" -> "yarvistek"
        if len(words) >= n:
            head = "".join(words[:n])
            if SequenceMatcher(None, head, wake).ratio() >= 0.75:
                return " ".join(words[n:]), True
    return t, False


def _score(keyword: str, text: str) -> float:
    kw = normalize(keyword)
    if kw and kw in text:
        return 1.0 + len(kw) / 100  # frases más largas ganan
    words, n = text.split(), len(kw.split())
    best = 0.0
    for i in range(max(1, len(words) - n + 1)):
        window = " ".join(words[i:i + n])
        best = max(best, SequenceMatcher(None, kw, window).ratio())
    return best if best >= 0.85 else 0.0


def match_command(text: str, models: Iterable[BaseModel]) -> tuple[BaseModel | None, float]:
    cleaned, _ = strip_wake_word(text)
    best_model, best_score = None, 0.0
    for model in models:
        for kw in model.keywords:
            s = _score(kw, cleaned)
            if s > best_score:
                best_model, best_score = model, s
    return best_model, best_score
