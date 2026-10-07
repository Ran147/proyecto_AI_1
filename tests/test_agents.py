"""Emotion and speech agents, with Google calls replaced by fakes."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import emotion_service, speech_service
from backend.services.intent_router import strip_wake_word

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("EMOTION_PROVIDER", "auto")
    monkeypatch.setenv("SPEECH_PROVIDER", "auto")
    monkeypatch.delenv("SPEECH_LANGUAGE", raising=False)


def vision_face(**likelihoods):
    face = {
        "joyLikelihood": "VERY_UNLIKELY",
        "sorrowLikelihood": "VERY_UNLIKELY",
        "angerLikelihood": "VERY_UNLIKELY",
        "surpriseLikelihood": "VERY_UNLIKELY",
        "detectionConfidence": 0.98,
    }
    face.update(likelihoods)
    return {"responses": [{"faceAnnotations": [face]}]}


# ---------- Google Vision (documented response format) ----------
def test_vision_joy():
    out = emotion_service.parse_vision_response(
        vision_face(joyLikelihood="VERY_LIKELY")
    )
    assert out["emotion"] == "happiness"
    assert out["scores"]["happiness"] > 0.9


def test_vision_neutral_when_nothing_likely():
    out = emotion_service.parse_vision_response(
        vision_face(joyLikelihood="UNLIKELY")
    )
    assert out["emotion"] == "neutral"


def test_vision_no_face():
    with pytest.raises(HTTPException) as error:
        emotion_service.parse_vision_response({"responses": [{}]})
    assert error.value.status_code == 422


def test_vision_api_error():
    with pytest.raises(HTTPException) as error:
        emotion_service.parse_vision_response(
            {"error": {"code": 403, "message": "API key not valid"}}
        )
    assert error.value.status_code == 502
    assert "API key" in error.value.detail


# ---------- Google Speech-to-Text ----------
def test_speech_ok():
    data = {"results": [{"alternatives": [{
        "transcript": " JarvisTEC bitcoin price ",
        "confidence": 0.92,
    }]}]}
    out = speech_service.parse_speech_response(data)
    assert out["text"] == "JarvisTEC bitcoin price"


def test_speech_empty():
    with pytest.raises(HTTPException) as error:
        speech_service.parse_speech_response({})
    assert error.value.status_code == 422


def test_speech_config_has_hints_and_encoding():
    config = speech_service.build_config("audio/webm;codecs=opus")
    phrases = config["speechContexts"][0]["phrases"]
    assert config["encoding"] == "WEBM_OPUS"
    assert config["languageCode"] == "en-US"
    assert "JarvisTEC" in phrases
    assert "bitcoin price" in phrases
    assert "encoding" not in speech_service.build_config("audio/wav")


# ---------- Full endpoints with a simulated Google ----------
def test_emotion_endpoint_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    async def fake_vision(_):
        return vision_face(sorrowLikelihood="LIKELY")

    monkeypatch.setattr(emotion_service, "_call_vision", fake_vision)
    response = client.post(
        "/emotion", files={"image": ("f.jpg", b"123", "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["emotion"] == "sadness"
    assert response.json()["provider"] == "google"


def test_transcribe_endpoint_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    seen = {}

    async def fake_speech(audio, content_type, language):
        seen["content_type"] = content_type
        return {"results": [{"alternatives": [{
            "transcript": "wine quality",
            "confidence": 0.9,
        }]}]}

    monkeypatch.setattr(speech_service, "_call_speech", fake_speech)
    response = client.post(
        "/transcribe", files={"audio": ("a.webm", b"123", "audio/webm")}
    )
    assert response.json()["text"] == "wine quality"
    assert seen["content_type"] == "audio/webm"


# ---------- Mock mode (no keys) ----------
def test_empty_provider_setting_means_auto(monkeypatch):
    monkeypatch.setenv("EMOTION_PROVIDER", "")
    monkeypatch.setenv("SPEECH_PROVIDER", "")
    assert speech_service.provider_name() in ("local", "browser")
    assert emotion_service.provider_name() in ("local", "mock")


def test_mock_mode(monkeypatch):
    monkeypatch.setenv("EMOTION_PROVIDER", "mock")
    monkeypatch.setenv("SPEECH_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_EMOTION", "anger")
    response = client.post(
        "/emotion", files={"image": ("f.jpg", b"123", "image/jpeg")}
    )
    assert response.json()["provider"] == "mock"
    assert response.json()["emotion"] == "anger"
    response = client.post(
        "/transcribe", files={"audio": ("a.webm", b"123", "audio/webm")}
    )
    assert response.json()["provider"] == "mock"
    assert client.get("/health").json()["speech_provider"] == "mock"


def test_empty_upload_rejected():
    response = client.post(
        "/emotion", files={"image": ("f.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 422


# ---------- Tolerant wake word ----------
@pytest.mark.parametrize("text", [
    "Jarvis tec bitcoin price",
    "yarvis tek bitcoin price",
    "JarvisTEC, bitcoin price",
])
def test_wake_word_variants(text):
    cleaned, found = strip_wake_word(text)
    assert found
    assert cleaned == "bitcoin price"


def test_no_false_wake_word():
    assert strip_wake_word("cuentame un chiste")[1] is False
