"""Local speech-to-text provider (Vosk).

Tests that run the recognizer are skipped when the model folder is missing
(scripts/download_speech_models.py fetches it). The fixture
tests/data/jarvis_wine_quality.wav is a synthetic voice saying "Jarvis tech,
wine quality".
"""
import io
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import speech_service

client = TestClient(app)
FIXTURE = Path(__file__).parent / "data" / "jarvis_wine_quality.wav"
needs_model = pytest.mark.skipif(
    not speech_service.local_model_available(),
    reason="speech model missing; run scripts/download_speech_models.py",
)


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("SPEECH_PROVIDER", "auto")


def wav(samples: np.ndarray, rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def post(data: bytes, name: str = "audio.wav", kind: str = "audio/wav"):
    return client.post("/transcribe", files={"audio": (name, data, kind)})


def test_auto_uses_local_only_when_the_model_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(speech_service, "LOCAL_MODEL_DIRS", {"en": tmp_path})
    assert speech_service.provider_name() == "browser"
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    assert speech_service.provider_name() == "google"


def test_without_any_engine_the_endpoint_says_so(monkeypatch, tmp_path):
    # No key and no model: the server must not invent a text.
    monkeypatch.setattr(speech_service, "LOCAL_MODEL_DIRS", {"en": tmp_path})
    assert client.get("/health").json()["speech_provider"] == "browser"
    response = post(b"123", "a.webm", "audio/webm")
    assert response.status_code == 503
    assert "browser" in response.json()["detail"]


def test_read_wav_resamples_and_mixes_down():
    stereo = np.zeros(44100 * 2, np.int16)  # one second, two channels
    pcm = speech_service.read_wav(wav(stereo, rate=44100, channels=2))
    assert len(pcm) == 16000 * 2  # one second of 16-bit mono at 16 kHz


@needs_model
def test_health_reports_local_provider():
    assert client.get("/health").json()["speech_provider"] == "local"


@needs_model
def test_transcribes_a_spoken_command():
    response = post(FIXTURE.read_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "local"
    assert "wine quality" in body["text"]
    assert 0 < body["confidence"] <= 1
    # The transcription is a valid command for the assistant.
    command = client.post("/command", json={"text": body["text"]}).json()
    assert command["matched"] and command["model"] == "wine"


@needs_model
def test_audio_that_is_not_wav_is_rejected():
    response = post(b"not a wav file", "audio.webm", "audio/webm")
    assert response.status_code == 415
    assert "WAV" in response.json()["detail"]


@needs_model
def test_silence_is_not_understood():
    response = post(wav(np.zeros(16000, np.int16)))
    assert response.status_code == 422


@needs_model
def test_too_short_audio_is_not_understood():
    response = post(wav(np.zeros(100, np.int16)))
    assert response.status_code == 422


# ---------- Languages ----------
SILENCE = wav(np.zeros(16000, np.int16))
needs_spanish = pytest.mark.skipif(
    not speech_service.local_model_available("es"),
    reason="Spanish model missing; run scripts/download_speech_models.py",
)


def test_unknown_language_is_rejected():
    response = client.post(
        "/transcribe",
        files={"audio": ("a.wav", SILENCE, "audio/wav")},
        data={"language": "fr"},
    )
    assert response.status_code == 422
    assert "language" in response.json()["detail"]


@needs_model
def test_answer_reports_the_language():
    assert post(FIXTURE.read_bytes()).json()["language"] == "en"


@needs_spanish
def test_spanish_model_loads_and_rejects_silence():
    response = client.post(
        "/transcribe",
        files={"audio": ("a.wav", SILENCE, "audio/wav")},
        data={"language": "es"},
    )
    assert response.status_code == 422


def test_google_language_codes():
    assert speech_service.build_config("audio/wav", "es")["languageCode"] == (
        "es-US"
    )
    assert speech_service.build_config("audio/wav", "en")["languageCode"] == (
        "en-US"
    )
