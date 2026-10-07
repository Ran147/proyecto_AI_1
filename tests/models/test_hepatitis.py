"""Trained ``hepatitis`` model.

Tests that run the model are skipped when artifacts/hepatitis.joblib is
missing (training/hepatitis/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.hepatitis import Hepatitis

ARTIFACT = Hepatitis().artifact_path()
CLASSES = {"blood_donor", "hepatitis", "fibrosis", "cirrhosis"}
# Lab profile with the median values of the cirrhosis patients (section 3
# of the notebook): low albumin and cholinesterase, high AST and bilirubin.
CIRRHOSIS_PROFILE = {
    "age": 56,
    "alb": 33.0,
    "alp": 80.05,
    "alt": 5.65,
    "ast": 92.9,
    "bil": 34.0,
    "che": 3.42,
    "chol": 3.87,
    "crea": 68.5,
    "ggt": 96.35,
    "prot": 70.0,
}

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/hepatitis/notebook.ipynb",
)
client = TestClient(app)


def predict(inputs=None):
    response = client.post(
        "/models/hepatitis/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def test_registered_as_real_model():
    info = get_orchestrator().describe("hepatitis")
    assert info["demo"] is False
    assert set(info["input_fields"]) == set(Hepatitis.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Hepatitis,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Hepatitis]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("hepatitis")


def test_invalid_input_rejected_before_loading():
    for inputs in ({"sex": "x"}, {"ast": -1}, {"age": 200}):
        response = client.post(
            "/models/hepatitis/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    probabilities = result["probabilities"]
    assert set(probabilities) == CLASSES
    assert abs(sum(probabilities.values()) - 1) < 1e-3
    assert result["category"] == max(probabilities, key=probabilities.get)
    # The default input is the median training patient: a healthy profile.
    assert result["category"] == "blood_donor"


@needs_artifact
def test_predict_partial_input_uses_defaults():
    inputs = predict({"sex": "f", "ast": 60.0})["inputs"]
    assert inputs["sex"] == "f"
    assert inputs["ast"] == 60.0
    assert inputs["alb"] == Hepatitis.default_input["alb"]


@needs_artifact
def test_cirrhosis_profile_is_not_a_blood_donor():
    probabilities = predict(CIRRHOSIS_PROFILE)["result"]["probabilities"]
    default = predict()["result"]["probabilities"]
    assert probabilities["blood_donor"] < default["blood_donor"]
    assert probabilities["cirrhosis"] > default["cirrhosis"]


def test_keywords_are_the_english_voice_commands():
    assert Hepatitis.keywords == [
        "hepatitis type",
        "type of hepatitis",
        "classify hepatitis",
        "hepatitis",
    ]


@pytest.mark.parametrize("text", [
    "JarvisTEC, hepatitis type",
    "Jarvis tec type of hepatitis",
    "JarvisTEC classify hepatitis",
    "JarvisTEC, hepatitis",
])
def test_voice_phrases_route_to_this_model(text):
    assert get_orchestrator().match(text)[0] == "hepatitis"


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, hepatitis type",
    "Jarvis tec type of hepatitis",
    "JarvisTEC classify hepatitis",
    "JarvisTEC, hepatitis",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "hepatitis"
    assert body["result"]["category"] in CLASSES
    assert "I estimate" in body["message"]
    assert "probability of" in body["message"]
    assert "not a medical diagnosis" in body["message"]


@needs_artifact
def test_default_spoken_sentence():
    assert predict()["message"].startswith(
        "I estimate that these lab values correspond to a blood donor "
        "without hepatitis C, with a probability of "
    )
