"""Speech-to-text on the server.

Single entry point: ``transcribe(audio_bytes, content_type)``.
Output: ``{"text": "...", "confidence": 0.93, "provider": "google"}``.

Providers:

- ``google``: Google Cloud Speech-to-Text (REST v1, needs GOOGLE_API_KEY).
- ``browser``: the server has no speech engine. The web interface transcribes
  with the browser's Web Speech API and sends only the text to
  ``POST /command``; ``POST /transcribe`` answers 503 with that explanation.
- ``mock``: canned text, only when SPEECH_PROVIDER=mock is set explicitly
  (development and tests).
"""
import base64
import os

import httpx
from fastapi import HTTPException

from ..core.orchestrator import get_orchestrator

SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"
# Google accepts at most 500 hint phrases per request.
MAX_HINT_PHRASES = 500


NO_PROVIDER_MESSAGE = (
    "No speech-to-text provider is configured on the server. The web "
    "interface transcribes in the browser and sends the text to "
    "POST /command; set GOOGLE_API_KEY to enable POST /transcribe."
)


def provider_name() -> str:
    """Return the active provider, or the explicit SPEECH_PROVIDER.

    In ``auto`` mode: google with GOOGLE_API_KEY, otherwise browser.
    """
    choice = (os.getenv("SPEECH_PROVIDER") or "auto").lower()
    if choice == "auto":
        return "google" if os.getenv("GOOGLE_API_KEY") else "browser"
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
        "languageCode": os.getenv("SPEECH_LANGUAGE") or "en-US",
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
        raise HTTPException(500, "GOOGLE_API_KEY is missing in .env")
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
            502, f"Could not reach Google Speech-to-Text: {exc}"
        ) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            502, "Google Speech-to-Text returned an unreadable response"
        ) from exc


def parse_speech_response(data: dict) -> dict:
    """Convert a raw Google Speech-to-Text response to the project format."""
    if "error" in data:
        message = data["error"].get("message", "unknown error")
        raise HTTPException(502, f"Google Speech-to-Text: {message}")
    results = data.get("results") or []
    alternatives = results[0].get("alternatives") if results else None
    if not alternatives:
        raise HTTPException(
            422,
            "I could not understand the audio. Speak closer to the "
            "microphone and try again.",
        )
    best = alternatives[0]
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
    text = os.getenv("MOCK_TRANSCRIPT") or "JarvisTEC bitcoin price"
    return {"text": text, "confidence": None, "provider": "mock"}


async def _transcribe_browser(audio_bytes: bytes, content_type: str) -> dict:
    """No engine on the server: say so instead of inventing a text."""
    raise HTTPException(503, NO_PROVIDER_MESSAGE)


PROVIDERS = {
    "google": _transcribe_google,
    "browser": _transcribe_browser,
    "mock": _transcribe_mock,
}


async def transcribe(audio_bytes: bytes, content_type: str = "") -> dict:
    """Transcribe audio with the active provider."""
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"Invalid SPEECH_PROVIDER: {name}")
    return await PROVIDERS[name](audio_bytes, content_type)
