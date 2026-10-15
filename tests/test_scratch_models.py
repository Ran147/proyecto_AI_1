"""Our own emotion and speech networks, trained from scratch.

They are optional: the pretrained models stay the default, and ours run only
with EMOTION_MODEL=scratch or SPEECH_PROVIDER=scratch. Tests that run a
network are skipped when its ONNX file is missing (the
notebooks in training/emotion_cnn/ and training/speech_commands/ create
them). The fixture tests/data/jarvis_three.wav is a synthetic voice (Windows
"Microsoft David") saying "Jarvis Tec, three": a voice the speech network
never heard in training.
"""
import io
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import emotion_service, speech_service

client = TestClient(app)
THREE = Path(__file__).parent / "data" / "jarvis_three.wav"
WINE = Path(__file__).parent / "data" / "jarvis_wine_quality.wav"
needs_speech = pytest.mark.skipif(
    not speech_service.scratch_model_available(),
    reason="run training/speech_commands/notebook.ipynb",
)
needs_emotion = pytest.mark.skipif(
    not emotion_service.SCRATCH_MODEL.exists()
    or not emotion_service.FACE_MODEL.exists(),
    reason="run training/emotion_cnn/notebook.ipynb",
)


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("SPEECH_PROVIDER", "auto")
    monkeypatch.setenv("EMOTION_PROVIDER", "auto")
    monkeypatch.delenv("EMOTION_MODEL", raising=False)


def wav(samples: np.ndarray, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def transcribe(data: bytes, **fields):
    return client.post(
        "/transcribe",
        files={"audio": ("audio.wav", data, "audio/wav")},
        data=fields,
    )


# --- Speech ---------------------------------------------------------------


def test_every_model_has_a_spoken_number():
    numbers = speech_service.command_numbers()
    names = client.get("/health").json()["models"]
    assert sorted(numbers.values()) == sorted(names)
    assert numbers["one"] == names[0]
    assert numbers["zero"] == names[9]


def test_mel_filterbank_covers_the_spectrum():
    bank = speech_service.mel_filterbank(16000, 512, 40, 20)
    assert bank.shape == (40, 257)
    # Every filter is a triangle that peaks close to 1.
    assert np.all(bank.max(axis=1) > 0.5)
    assert np.all(bank >= 0)


def test_unknown_engine_answers_422():
    response = transcribe(wav(np.zeros(16000)), engine="whisper")
    assert response.status_code == 422
    assert "not available" in response.json()["detail"]


@needs_speech
def test_our_model_is_only_used_when_configured(monkeypatch):
    # Default: never chosen automatically, nor offered to the interface.
    assert speech_service.provider_name() != "scratch"
    assert "scratch" not in client.get("/health").json()["speech_engines"]
    assert transcribe(THREE.read_bytes(), engine="scratch").status_code == 422
    monkeypatch.setenv("SPEECH_PROVIDER", "scratch")
    health = client.get("/health").json()
    assert health["speech_provider"] == "scratch"
    assert "scratch" in health["speech_engines"]
    assert health["speech_numbers"]["one"] == health["models"][0]


@needs_speech
def test_our_model_hears_a_number_and_runs_its_command(monkeypatch):
    monkeypatch.setenv("SPEECH_PROVIDER", "scratch")
    response = transcribe(THREE.read_bytes())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "scratch"
    assert body["heard"] == "three"
    third = speech_service.command_numbers()["three"]
    model, _ = speech_service.get_orchestrator().match(body["text"])
    assert model == third


@needs_speech
def test_our_model_rejects_silence(monkeypatch):
    monkeypatch.setenv("SPEECH_PROVIDER", "scratch")
    response = transcribe(wav(np.zeros(2 * 16000)))
    assert response.status_code == 422
    assert "number" in response.json()["detail"]


@needs_speech
@pytest.mark.skipif(
    not speech_service.local_model_available(), reason="Vosk missing"
)
def test_vosk_stays_available_per_request(monkeypatch):
    monkeypatch.setenv("SPEECH_PROVIDER", "scratch")
    response = transcribe(WINE.read_bytes(), engine="local")
    assert response.status_code == 200, response.text
    assert response.json()["provider"] == "local"
    assert "wine" in response.json()["text"]


# --- Emotion --------------------------------------------------------------


def test_unknown_emotion_model_answers_422():
    with pytest.raises(HTTPException) as error:
        emotion_service.check_classifier("deepface")
    assert error.value.status_code == 422


def test_missing_scratch_file_falls_back(monkeypatch, tmp_path):
    files = dict(emotion_service.NETWORK_FILES)
    files["scratch"] = tmp_path / "missing.onnx"
    monkeypatch.setattr(emotion_service, "NETWORK_FILES", files)
    assert "scratch" not in emotion_service.available_classifiers()
    assert emotion_service.local_classifiers("scratch") != ["scratch"]


@needs_emotion
def test_pretrained_ensemble_is_the_default(monkeypatch):
    assert emotion_service.local_classifiers() == ["ferplus", "hsemotion"]
    assert client.get("/health").json()["emotion_models"][0] == "ensemble"
    monkeypatch.setenv("EMOTION_MODEL", "scratch")
    assert emotion_service.local_classifiers() == ["scratch"]


@needs_emotion
def test_our_network_scores_every_emotion():
    model = emotion_service._get_local_model()
    face = np.full((120, 120, 3), 128, np.uint8)
    scores = model.classify(face, "scratch")
    assert set(scores) == set(emotion_service.FERPLUS_CLASSES)
    assert scores["contempt"] == 0
    assert sum(scores.values()) == pytest.approx(1, abs=0.01)


@needs_emotion
def test_endpoint_takes_the_model_field(monkeypatch):
    import cv2

    model = emotion_service._get_local_model()
    monkeypatch.setattr(
        model, "detector", type("D", (), {
            "setInputSize": lambda self, size: None,
            "detect": lambda self, image: (1, np.array(
                [[100, 60, 200, 240] + [0] * 10 + [0.95]], np.float32
            )),
        })()
    )
    image = cv2.imencode(".jpg", np.full((480, 640, 3), 120, np.uint8))[1]
    for choice in emotion_service.available_classifiers():
        response = client.post(
            "/emotion",
            files={"image": ("f.jpg", image.tobytes(), "image/jpeg")},
            data={"model": choice},
        )
        assert response.status_code == 200, response.text
        expected = emotion_service.CLASSIFIER_CHOICES[choice]
        assert response.json()["model"] == "+".join(expected)
    bad = client.post(
        "/emotion",
        files={"image": ("f.jpg", image.tobytes(), "image/jpeg")},
        data={"model": "nope"},
    )
    assert bad.status_code == 422
