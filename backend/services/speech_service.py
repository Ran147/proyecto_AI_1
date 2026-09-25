"""Speech-to-text with Google Cloud Speech-to-Text (REST v1, API key).

Single entry point: ``transcribe(audio_bytes, content_type)``.
Output: ``{"text": "...", "confidence": 0.93, "provider": "google"}``.
"""
import base64
import os

import httpx
from fastapi import HTTPException

from ..core.orchestrator import get_orchestrator

SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"
# Google accepts at most 500 hint phrases per request.
MAX_HINT_PHRASES = 500


def provider_name() -> str:
    """Return the active provider: google, mock or an explicit override."""
    choice = os.getenv("SPEECH_PROVIDER", "auto").lower()
    if choice == "auto":
        return "google" if os.getenv("GOOGLE_API_KEY") else "mock"
    return choice


def _encoding_for(content_type: str) -> str | None:
    """Map the browser recording format to a Google encoding.

    Chrome, Edge and Firefox record webm/opus. Safari records mp4, which
    Google does not accept.
    """
    content_type = (content_type or "").lower()
    if "webm" in content_type:
        return "WEBM_OPUS"
    if "ogg" in content_type:
        return "OGG_OPUS"
    # wav/flac: Google detects the encoding from the file header.
    return None


def build_config(content_type: str) -> dict:
    """Build the recognition config, including command hint phrases."""
    # Hint phrases greatly improve recognition of "JarvisTEC" and commands.
    phrases = [
        "JarvisTEC",
        "Jarvis TEC",
        *get_orchestrator().command_phrases(),
    ]
    config = {
        "languageCode": os.getenv("SPEECH_LANGUAGE", "es-CR"),
        "enableAutomaticPunctuation": False,
        "maxAlternatives": 1,
        "speechContexts": [
            {"phrases": phrases[:MAX_HINT_PHRASES], "boost": 15}
        ],
    }
    encoding = _encoding_for(content_type)
    if encoding:
        config["encoding"] = encoding
    return config


async def _call_speech(audio_bytes: bytes, content_type: str) -> dict:
    """HTTP call to Google Speech-to-Text (isolated for tests)."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "Falta GOOGLE_API_KEY en el archivo .env")
    payload = {
        "config": build_config(content_type),
        "audio": {"content": base64.b64encode(audio_bytes).decode()},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                SPEECH_URL, params={"key": key}, json=payload
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"No se pudo contactar a Google Speech-to-Text: {exc}"
        ) from exc
    return response.json()


def parse_speech_response(data: dict) -> dict:
    """Convert a raw Google Speech-to-Text response to the project format."""
    if "error" in data:
        message = data["error"].get("message", "error desconocido")
        raise HTTPException(502, f"Google Speech-to-Text: {message}")
    results = data.get("results") or []
    if not results:
        raise HTTPException(
            422,
            "No se entendió el audio. Habla más cerca del micrófono e "
            "intenta de nuevo.",
        )
    best = results[0]["alternatives"][0]
    return {
        "text": best["transcript"].strip(),
        "confidence": best.get("confidence"),
        "provider": "google",
    }


async def _transcribe_google(audio_bytes: bytes, content_type: str) -> dict:
    raw = await _call_speech(audio_bytes, content_type)
    return parse_speech_response(raw)


async def _transcribe_mock(audio_bytes: bytes, content_type: str) -> dict:
    """Simulated provider: returns MOCK_TRANSCRIPT and ignores the audio."""
    text = os.getenv("MOCK_TRANSCRIPT", "JarvisTEC precio del bitcoin")
    return {"text": text, "confidence": None, "provider": "mock"}


PROVIDERS = {"google": _transcribe_google, "mock": _transcribe_mock}


async def transcribe(audio_bytes: bytes, content_type: str = "") -> dict:
    """Transcribe audio with the active provider."""
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"SPEECH_PROVIDER inválido: {name}")
    return await PROVIDERS[name](audio_bytes, content_type)
