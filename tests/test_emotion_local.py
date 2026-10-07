"""Local emotion provider (YuNet + FER+).

Tests that run the networks are skipped when the ONNX files are missing
(scripts/download_emotion_models.py fetches them).
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import emotion_service

client = TestClient(app)
needs_models = pytest.mark.skipif(
    not emotion_service.local_models_available(),
    reason="emotion models missing; run scripts/download_emotion_models.py",
)


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("EMOTION_PROVIDER", "auto")


def jpeg(image) -> bytes:
    import cv2

    return cv2.imencode(".jpg", image)[1].tobytes()


class FakeDetector:
    """Stands in for YuNet: always reports one face at a fixed box."""

    def __init__(self, box=(220, 120, 200, 240), confidence=0.95):
        self.row = np.zeros(15, np.float32)
        self.row[:4] = box
        self.row[14] = confidence

    def setInputSize(self, size):  # noqa: N802 (OpenCV method name)
        pass

    def detect(self, image):
        return 1, np.array([self.row])


def post(data: bytes):
    return client.post(
        "/emotion", files={"image": ("frame.jpg", data, "image/jpeg")}
    )


def test_auto_prefers_local_only_when_models_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(emotion_service, "FACE_MODEL", tmp_path / "no.onnx")
    assert emotion_service.provider_name() == "mock"
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    assert emotion_service.provider_name() == "google"


def test_explicit_local_without_models_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("EMOTION_PROVIDER", "local")
    monkeypatch.setattr(emotion_service, "FACE_MODEL", tmp_path / "no.onnx")
    monkeypatch.setattr(emotion_service, "_local_model", None)
    assert post(b"123").status_code == 503


@needs_models
def test_health_reports_local_provider():
    assert client.get("/health").json()["emotion_provider"] == "local"


@needs_models
def test_unreadable_image_rejected():
    response = post(b"this is not an image")
    assert response.status_code == 422
    assert "image" in response.json()["detail"]


@needs_models
def test_frame_without_face_rejected():
    blank = np.full((480, 640, 3), 127, np.uint8)
    response = post(jpeg(blank))
    assert response.status_code == 422
    assert "face" in response.json()["detail"]


@needs_models
def test_frame_too_thin_to_hold_a_face_rejected():
    # A 5000x2 image used to crash the resize step with a 500.
    sliver = np.full((2, 5000, 3), 127, np.uint8)
    response = post(jpeg(sliver))
    assert response.status_code == 422
    assert "face" in response.json()["detail"]


@needs_models
def test_face_returns_normalized_scores(monkeypatch):
    # The classifier runs for real on the region the fake detector reports.
    model = emotion_service._get_local_model()
    monkeypatch.setattr(model, "detector", FakeDetector())
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    response = post(jpeg(frame))
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "local"
    assert set(body["scores"]) == set(emotion_service.FERPLUS_CLASSES)
    assert abs(sum(body["scores"].values()) - 1) < 1e-2
    assert body["emotion"] == max(body["scores"], key=body["scores"].get)
    assert body["face_confidence"] == pytest.approx(0.95, abs=1e-3)


@needs_models
def test_face_box_outside_the_frame_is_rejected(monkeypatch):
    model = emotion_service._get_local_model()
    monkeypatch.setattr(
        model, "detector", FakeDetector(box=(900, 900, 50, 50))
    )
    frame = np.full((480, 640, 3), 127, np.uint8)
    assert post(jpeg(frame)).status_code == 422
