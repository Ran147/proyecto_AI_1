"""Voz a texto con Google Cloud Speech-to-Text (API REST v1, con clave de API).

Punto único de entrada: transcribe(audio_bytes, content_type).
Salida: {"text": "...", "confidence": 0.93, "provider": "google"}
"""
import base64
import os

import httpx
from fastapi import HTTPException

from ..core.orchestrator import get_orchestrator

SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"


def provider_name() -> str:
    choice = os.getenv("SPEECH_PROVIDER", "auto").lower()
    if choice == "auto":
        return "google" if os.getenv("GOOGLE_API_KEY") else "mock"
    return choice


def _encoding_for(content_type: str) -> str | None:
    """El navegador graba webm/opus (Chrome, Edge, Firefox). Safari graba mp4, que Google no admite."""
    ct = (content_type or "").lower()
    if "webm" in ct:
        return "WEBM_OPUS"
    if "ogg" in ct:
        return "OGG_OPUS"
    return None  # wav/flac: Google lo detecta por el encabezado del archivo


def build_config(content_type: str) -> dict:
    # Frases de sugerencia: mejoran mucho el reconocimiento de "JarvisTEC" y de los comandos
    phrases = ["JarvisTEC", "Jarvis TEC", *get_orchestrator().command_phrases()]
    config = {
        "languageCode": os.getenv("SPEECH_LANGUAGE", "es-CR"),
        "enableAutomaticPunctuation": False,
        "maxAlternatives": 1,
        "speechContexts": [{"phrases": phrases[:500], "boost": 15}],
    }
    enc = _encoding_for(content_type)
    if enc:
        config["encoding"] = enc
    return config


async def _call_speech(audio_bytes: bytes, content_type: str) -> dict:
    """Llamada HTTP a Google Speech-to-Text (aislada para poder simularla en los tests)."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "Falta GOOGLE_API_KEY en el archivo .env")
    payload = {"config": build_config(content_type),
               "audio": {"content": base64.b64encode(audio_bytes).decode()}}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(SPEECH_URL, params={"key": key}, json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"No se pudo contactar a Google Speech-to-Text: {e}")
    return r.json()


def parse_speech_response(data: dict) -> dict:
    if "error" in data:
        raise HTTPException(502, f"Google Speech-to-Text: {data['error'].get('message', 'error desconocido')}")
    results = data.get("results") or []
    if not results:
        raise HTTPException(422, "No se entendió el audio. Habla más cerca del micrófono e intenta de nuevo.")
    best = results[0]["alternatives"][0]
    return {"text": best["transcript"].strip(), "confidence": best.get("confidence"), "provider": "google"}


async def _transcribe_google(audio_bytes: bytes, content_type: str) -> dict:
    return parse_speech_response(await _call_speech(audio_bytes, content_type))


async def _transcribe_mock(audio_bytes: bytes, content_type: str) -> dict:
    """Simulación: NO transcribe el audio. Devuelve MOCK_TRANSCRIPT del .env."""
    text = os.getenv("MOCK_TRANSCRIPT", "JarvisTEC precio del bitcoin")
    return {"text": text, "confidence": None, "provider": "mock"}


PROVIDERS = {"google": _transcribe_google, "mock": _transcribe_mock}


async def transcribe(audio_bytes: bytes, content_type: str = "") -> dict:
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"SPEECH_PROVIDER inválido: {name}")
    return await PROVIDERS[name](audio_bytes, content_type)
