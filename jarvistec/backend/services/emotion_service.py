"""Detección de emociones a partir de una imagen.

Punto único de entrada: detect_emotion(image_bytes). Para cambiar de proveedor
(Azure, un modelo propio en la entrega final...) solo hay que agregar otra
función _detect_xxx y registrarla en PROVIDERS.

Salida normalizada (la que espera el frontend):
    {"emotion": "happiness", "scores": {"happiness": 0.95, ...}, "provider": "google"}
"""
import base64
import os
import random

import httpx
from fastapi import HTTPException

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"

# Google Vision no da probabilidades sino "niveles de verosimilitud". Se convierten a un puntaje 0-1.
LIKELIHOOD = {
    "UNKNOWN": 0.0, "VERY_UNLIKELY": 0.03, "UNLIKELY": 0.2,
    "POSSIBLE": 0.5, "LIKELY": 0.8, "VERY_LIKELY": 0.95,
}
# Campo de Google -> nombre normalizado del proyecto
FIELDS = {
    "joyLikelihood": "happiness", "sorrowLikelihood": "sadness",
    "angerLikelihood": "anger", "surpriseLikelihood": "surprise",
}
NEUTRAL_THRESHOLD = 0.5  # si ninguna emoción llega a "POSSIBLE", se considera neutral


def provider_name() -> str:
    choice = os.getenv("EMOTION_PROVIDER", "auto").lower()
    if choice == "auto":
        return "google" if os.getenv("GOOGLE_API_KEY") else "mock"
    return choice


async def _call_vision(image_bytes: bytes) -> dict:
    """Llamada HTTP a Google Vision (aislada para poder simularla en los tests)."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "Falta GOOGLE_API_KEY en el archivo .env")
    payload = {"requests": [{
        "image": {"content": base64.b64encode(image_bytes).decode()},
        "features": [{"type": "FACE_DETECTION", "maxResults": 1}],
    }]}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(VISION_URL, params={"key": key}, json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"No se pudo contactar a Google Vision: {e}")
    return r.json()


def parse_vision_response(data: dict) -> dict:
    """Convierte la respuesta cruda de Google Vision al formato del proyecto."""
    if "error" in data:
        raise HTTPException(502, f"Google Vision: {data['error'].get('message', 'error desconocido')}")
    resp = (data.get("responses") or [{}])[0]
    if "error" in resp:
        raise HTTPException(502, f"Google Vision: {resp['error'].get('message', 'error desconocido')}")
    faces = resp.get("faceAnnotations") or []
    if not faces:
        raise HTTPException(422, "No se detectó ningún rostro. Mira a la cámara con buena iluminación.")
    face = faces[0]
    scores = {name: LIKELIHOOD.get(face.get(field, "UNKNOWN"), 0.0) for field, name in FIELDS.items()}
    top, top_score = max(scores.items(), key=lambda kv: kv[1])
    emotion = top if top_score >= NEUTRAL_THRESHOLD else "neutral"
    return {"emotion": emotion, "scores": scores, "provider": "google",
            "face_confidence": face.get("detectionConfidence")}


async def _detect_google(image_bytes: bytes) -> dict:
    return parse_vision_response(await _call_vision(image_bytes))


async def _detect_mock(image_bytes: bytes) -> dict:
    """Simulación para desarrollar/demostrar sin credenciales. NO analiza la imagen."""
    forced = os.getenv("MOCK_EMOTION")
    emotion = forced or random.choice(["happiness", "neutral", "sadness", "surprise", "anger"])
    scores = {n: 0.05 for n in FIELDS.values()}
    if emotion in scores:
        scores[emotion] = 0.9
    return {"emotion": emotion, "scores": scores, "provider": "mock"}


PROVIDERS = {"google": _detect_google, "mock": _detect_mock}


async def detect_emotion(image_bytes: bytes) -> dict:
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"EMOTION_PROVIDER inválido: {name}")
    return await PROVIDERS[name](image_bytes)
