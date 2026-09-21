import asyncio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import emotion_service as emo, speech_service as sp
from backend.services.intent_router import strip_wake_word

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("EMOTION_PROVIDER", "auto")
    monkeypatch.setenv("SPEECH_PROVIDER", "auto")


def face(**kw):
    base = dict(joyLikelihood="VERY_UNLIKELY", sorrowLikelihood="VERY_UNLIKELY",
                angerLikelihood="VERY_UNLIKELY", surpriseLikelihood="VERY_UNLIKELY", detectionConfidence=0.98)
    base.update(kw)
    return {"responses": [{"faceAnnotations": [base]}]}


# ---------- Google Vision (formato de respuesta documentado) ----------
def test_vision_joy():
    out = emo.parse_vision_response(face(joyLikelihood="VERY_LIKELY"))
    assert out["emotion"] == "happiness" and out["scores"]["happiness"] > 0.9


def test_vision_neutral_when_nothing_likely():
    assert emo.parse_vision_response(face(joyLikelihood="UNLIKELY"))["emotion"] == "neutral"


def test_vision_no_face():
    with pytest.raises(HTTPException) as e:
        emo.parse_vision_response({"responses": [{}]})
    assert e.value.status_code == 422


def test_vision_api_error():
    with pytest.raises(HTTPException) as e:
        emo.parse_vision_response({"error": {"code": 403, "message": "API key not valid"}})
    assert e.value.status_code == 502 and "API key" in e.value.detail


# ---------- Google Speech ----------
def test_speech_ok():
    data = {"results": [{"alternatives": [{"transcript": " JarvisTEC precio del bitcoin ", "confidence": 0.92}]}]}
    assert sp.parse_speech_response(data)["text"] == "JarvisTEC precio del bitcoin"


def test_speech_empty():
    with pytest.raises(HTTPException) as e:
        sp.parse_speech_response({})
    assert e.value.status_code == 422


def test_speech_config_has_hints_and_encoding():
    cfg = sp.build_config("audio/webm;codecs=opus")
    assert cfg["encoding"] == "WEBM_OPUS" and cfg["languageCode"] == "es-CR"
    assert "JarvisTEC" in cfg["speechContexts"][0]["phrases"]
    assert "precio del bitcoin" in cfg["speechContexts"][0]["phrases"]
    assert "encoding" not in sp.build_config("audio/wav")


# ---------- Endpoints completos con Google simulado ----------
def test_emotion_endpoint_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    async def fake(_): return face(sorrowLikelihood="LIKELY")
    monkeypatch.setattr(emo, "_call_vision", fake)
    r = client.post("/emotion", files={"image": ("f.jpg", b"123", "image/jpeg")})
    assert r.status_code == 200 and r.json()["emotion"] == "sadness" and r.json()["provider"] == "google"


def test_transcribe_endpoint_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    seen = {}
    async def fake(audio, ct):
        seen["ct"] = ct
        return {"results": [{"alternatives": [{"transcript": "calidad del vino", "confidence": 0.9}]}]}
    monkeypatch.setattr(sp, "_call_speech", fake)
    r = client.post("/transcribe", files={"audio": ("a.webm", b"123", "audio/webm")})
    assert r.json()["text"] == "calidad del vino" and seen["ct"] == "audio/webm"


# ---------- Modo simulado (sin claves) ----------
def test_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_EMOTION", "anger")
    r = client.post("/emotion", files={"image": ("f.jpg", b"123", "image/jpeg")})
    assert r.json()["provider"] == "mock" and r.json()["emotion"] == "anger"
    r = client.post("/transcribe", files={"audio": ("a.webm", b"123", "audio/webm")})
    assert r.json()["provider"] == "mock"
    assert client.get("/health").json()["speech_provider"] == "mock"


def test_empty_upload_rejected():
    assert client.post("/emotion", files={"image": ("f.jpg", b"", "image/jpeg")}).status_code == 422


# ---------- Palabra de activación tolerante ----------
@pytest.mark.parametrize("text", ["Jarvis tec precio del bitcoin", "yarvis tek precio del bitcoin", "JarvisTEC, precio del bitcoin"])
def test_wake_word_variants(text):
    cleaned, had = strip_wake_word(text)
    assert had and cleaned == "precio del bitcoin"


def test_no_false_wake_word():
    assert strip_wake_word("cuentame un chiste")[1] is False
