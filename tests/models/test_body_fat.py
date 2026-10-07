"""Trained ``body_fat`` model.

Tests that run the model are skipped when artifacts/body_fat.joblib is
missing (training/body_fat/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.body_fat import BodyFat

ARTIFACT = BodyFat().artifact_path()

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/body_fat/notebook.ipynb",
)
client = TestClient(app)


def predict(inputs=None):
    response = client.post(
        "/models/body_fat/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def test_registered_as_real_model():
    info = get_orchestrator().describe("body_fat")
    assert info["demo"] is False
    assert info["task_type"] == "regression"
    assert set(info["input_fields"]) == set(BodyFat.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        BodyFat,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([BodyFat]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("body_fat")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"abdomen_cm": 0},
        {"weight_kg": -5},
        {"weight_kg": 1000},
        {"abdomen_cm": "wide"},
    ):
        response = client.post(
            "/models/body_fat/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    assert 0 <= result["body_fat_percent"] <= 60
    assert result["outside_training_range"] is False
    assert result["fat_mass_kg"] == pytest.approx(
        body["inputs"]["weight_kg"] * result["body_fat_percent"] / 100,
        abs=0.1,
    )


@needs_artifact
def test_predict_partial_input_uses_defaults():
    inputs = predict({"abdomen_cm": 100})["inputs"]
    assert inputs["abdomen_cm"] == 100
    assert inputs["weight_kg"] == BodyFat.default_input["weight_kg"]


@needs_artifact
def test_larger_abdomen_means_more_body_fat():
    def body_fat(**inputs):
        return predict(inputs)["result"]["body_fat_percent"]

    # Same weight, wider abdomen: more fat.
    assert body_fat(abdomen_cm=105) > body_fat(abdomen_cm=85)
    # Same abdomen, heavier man: more lean mass, so a lower percentage.
    assert body_fat(weight_kg=95) < body_fat(weight_kg=70)


@needs_artifact
def test_extreme_input_is_flagged_and_stays_in_range():
    for inputs in (
        {"abdomen_cm": 190, "weight_kg": 60},
        {"abdomen_cm": 55, "weight_kg": 200},
    ):
        result = predict(inputs)["result"]
        assert result["outside_training_range"] is True
        assert 0 <= result["body_fat_percent"] <= 60


def test_keywords_are_the_english_voice_commands():
    assert BodyFat.keywords == [
        "body fat",
        "body fat percentage",
        "how much body fat do i have",
        "fat percentage",
    ]


def test_spoken_sentence_is_english_and_warns_outside_the_range():
    result = {
        "body_fat_percent": 21.3,
        "fat_mass_kg": 16.7,
        "typical_error_points": 3.2,
        "outside_training_range": False,
    }
    message = BodyFat().speak(result)
    assert "body fat percentage of about 21.3%" in message
    assert "16.7 kilograms of fat" in message
    assert "typical error of 3 points" in message
    assert "not a medical assessment" in message
    warning = BodyFat().speak({**result, "outside_training_range": True})
    assert "outside the range" in warning
    assert "not very reliable" in warning


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, body fat",
    "Jarvis tec body fat percentage",
    "JarvisTEC how much body fat do I have?",
    "JarvisTEC, fat percentage",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "body_fat"
    assert 0 <= body["result"]["body_fat_percent"] <= 60
    assert "body fat percentage" in body["message"]
    assert "not a medical assessment" in body["message"]
