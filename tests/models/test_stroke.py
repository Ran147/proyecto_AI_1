"""Trained ``stroke`` model.

Tests that run the model are skipped when artifacts/stroke.joblib is missing
(training/stroke/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.stroke import Stroke

ARTIFACT = Stroke().artifact_path()
META = Stroke().artifact_path(".meta.json")
LEVELS = {"low", "moderate", "high"}

needs_artifact = pytest.mark.skipif(
    not (ARTIFACT.exists() and META.exists()),
    reason=f"{ARTIFACT.name} missing; run training/stroke/notebook.ipynb",
)
client = TestClient(app)


def predict(inputs=None):
    response = client.post(
        "/models/stroke/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def test_registered_as_real_model():
    info = get_orchestrator().describe("stroke")
    assert info["demo"] is False
    assert set(info["input_fields"]) == set(Stroke.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Stroke,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / f"missing{suffix}",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Stroke]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("stroke")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"gender": "Robot"},
        {"age": -1},
        {"hypertension": 2},
        {"smoking_status": "sometimes"},
        {"bmi": 0},
    ):
        response = client.post(
            "/models/stroke/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_default_input_matches_meta():
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert Stroke.default_input == meta["default_input"]
    assert set(Stroke.default_input) == set(meta["features"])


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    assert 0 <= result["stroke_probability"] <= 1
    assert result["risk_level"] in LEVELS
    assert result["flagged"] is (
        result["stroke_probability"] >= result["threshold"]
    )
    # The typical patient (45 years old, no conditions) is not flagged.
    assert result["risk_level"] == "low"


@needs_artifact
def test_predict_partial_input_uses_defaults():
    inputs = predict({"age": 70, "smoking_status": "smokes"})["inputs"]
    assert inputs["age"] == 70
    assert inputs["smoking_status"] == "smokes"
    assert inputs["bmi"] == Stroke.default_input["bmi"]


@needs_artifact
def test_old_patient_with_conditions_has_higher_risk():
    old = predict({
        "age": 80, "hypertension": 1, "heart_disease": 1,
        "avg_glucose_level": 220,
    })["result"]
    young = predict({
        "age": 25, "ever_married": "No", "avg_glucose_level": 85, "bmi": 22,
    })["result"]
    assert old["stroke_probability"] > young["stroke_probability"]
    assert old["flagged"] is True
    assert young["risk_level"] == "low"


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, stroke risk",
    "Jarvis tec risk of a stroke",
    "JarvisTEC risk of stroke",
    "JarvisTEC, stroke",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "stroke"
    assert body["result"]["risk_level"] in LEVELS
    # Spoken as an estimate, never as a diagnosis.
    assert "I estimate" in body["message"]
    assert "not a medical diagnosis" in body["message"]


def test_each_keyword_routes_to_stroke():
    orchestrator = get_orchestrator()
    for keyword in Stroke.keywords:
        assert orchestrator.match(f"JarvisTEC {keyword}")[0] == "stroke"


def test_keywords_are_the_english_commands():
    assert Stroke.title == "Stroke risk"
    assert Stroke.keywords == [
        "stroke risk",
        "risk of a stroke",
        "risk of stroke",
        "stroke",
    ]


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_spoken_sentence_is_english(level):
    message = Stroke().speak({
        "stroke_probability": 0.1234,
        "risk_level": level,
        "flagged": True,
        "threshold": 0.1,
    })
    assert message == (
        f"I estimate a {level} risk of stroke, with a probability of "
        "about 12.3%; this is a rough estimate, not a medical "
        "diagnosis."
    )
