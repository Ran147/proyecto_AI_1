"""Emotion detection from an image.

Single entry point: ``detect_emotion(image_bytes)``. To switch providers
(Azure, a local model...) add a ``_detect_<name>`` function and register it
in PROVIDERS.

Normalized output (the format the frontend expects)::

    {"emotion": "happiness", "scores": {"happiness": 0.95, ...},
     "provider": "google"}
"""
import base64
import os
import random

import httpx
from fastapi import HTTPException

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"

# Google Vision returns likelihood levels, not probabilities; they are
# mapped to a 0-1 score.
LIKELIHOOD_SCORES = {
    "UNKNOWN": 0.0,
    "VERY_UNLIKELY": 0.03,
    "UNLIKELY": 0.2,
    "POSSIBLE": 0.5,
    "LIKELY": 0.8,
    "VERY_LIKELY": 0.95,
}
# Google field -> normalized project emotion name.
VISION_FIELDS = {
    "joyLikelihood": "happiness",
    "sorrowLikelihood": "sadness",
    "angerLikelihood": "anger",
    "surpriseLikelihood": "surprise",
}
# Below "POSSIBLE" for every emotion, the face is considered neutral.
NEUTRAL_THRESHOLD = 0.5


def provider_name() -> str:
    """Return the active provider: google, mock or an explicit override."""
    choice = os.getenv("EMOTION_PROVIDER", "auto").lower()
    if choice == "auto":
        return "google" if os.getenv("GOOGLE_API_KEY") else "mock"
    return choice


async def _call_vision(image_bytes: bytes) -> dict:
    """HTTP call to Google Vision (isolated so tests can replace it)."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "Falta GOOGLE_API_KEY en el archivo .env")
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(image_bytes).decode()},
            "features": [{"type": "FACE_DETECTION", "maxResults": 1}],
        }]
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                VISION_URL, params={"key": key}, json=payload
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"No se pudo contactar a Google Vision: {exc}"
        ) from exc
    return response.json()


def _vision_error(error: dict) -> HTTPException:
    message = error.get("message", "error desconocido")
    return HTTPException(502, f"Google Vision: {message}")


def parse_vision_response(data: dict) -> dict:
    """Convert a raw Google Vision response into the project format."""
    if "error" in data:
        raise _vision_error(data["error"])
    response = (data.get("responses") or [{}])[0]
    if "error" in response:
        raise _vision_error(response["error"])
    faces = response.get("faceAnnotations") or []
    if not faces:
        raise HTTPException(
            422,
            "No se detectó ningún rostro. Mira a la cámara con buena "
            "iluminación.",
        )
    face = faces[0]
    scores = {
        name: LIKELIHOOD_SCORES.get(face.get(field, "UNKNOWN"), 0.0)
        for field, name in VISION_FIELDS.items()
    }
    top, top_score = max(scores.items(), key=lambda item: item[1])
    emotion = top if top_score >= NEUTRAL_THRESHOLD else "neutral"
    return {
        "emotion": emotion,
        "scores": scores,
        "provider": "google",
        "face_confidence": face.get("detectionConfidence"),
    }


async def _detect_google(image_bytes: bytes) -> dict:
    return parse_vision_response(await _call_vision(image_bytes))


async def _detect_mock(image_bytes: bytes) -> dict:
    """Simulated provider for work without credentials; ignores the image."""
    emotion = os.getenv("MOCK_EMOTION") or random.choice(
        ["happiness", "neutral", "sadness", "surprise", "anger"]
    )
    scores = {name: 0.05 for name in VISION_FIELDS.values()}
    if emotion in scores:
        scores[emotion] = 0.9
    return {"emotion": emotion, "scores": scores, "provider": "mock"}


PROVIDERS = {"google": _detect_google, "mock": _detect_mock}


async def detect_emotion(image_bytes: bytes) -> dict:
    """Detect the dominant emotion with the active provider."""
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"EMOTION_PROVIDER inválido: {name}")
    return await PROVIDERS[name](image_bytes)
