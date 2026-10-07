"""Trained ``wine`` model.

Tests that run the model are skipped when artifacts/wine.joblib is missing
(training/wine/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.wine import Wine

ARTIFACT = Wine().artifact_path()
CLASSES = {"low", "medium", "high"}

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/wine/notebook.ipynb",
)
client = TestClient(app)


def test_registered_as_real_model():
    info = get_orchestrator().describe("wine")
    assert info["demo"] is False
    assert set(info["input_fields"]) == set(Wine.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Wine,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Wine]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("wine")


def test_invalid_input_rejected_before_loading():
    for inputs in ({"type": "rose"}, {"ph": 20}):
        response = client.post(
            "/models/wine/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_predict_default_input():
    response = client.post("/models/wine/predict", json={})
    assert response.status_code == 200
    body = response.json()
    json.dumps(body)  # must be serializable
    result = body["result"]
    probabilities = result["probabilities"]
    assert result["quality"] in CLASSES
    assert set(probabilities) == CLASSES
    assert abs(sum(probabilities.values()) - 1) < 1e-3
    assert result["quality"] == max(probabilities, key=probabilities.get)


@needs_artifact
def test_predict_partial_input_uses_defaults():
    response = client.post(
        "/models/wine/predict",
        json={"inputs": {"type": "red", "alcohol": 13.5}},
    )
    assert response.status_code == 200
    inputs = response.json()["inputs"]
    assert inputs["type"] == "red"
    assert inputs["ph"] == Wine.default_input["ph"]


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, wine quality",
    "JarvisTEC quality of the wine",
    "Jarvis tec classify wine",
    "Jarvis tec how good is this wine",
    "JarvisTEC, wine",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "wine"
    assert body["result"]["quality"] in CLASSES
    assert "quality" in body["message"]
    assert "probability" in body["message"]


def test_spoken_wine_type_needs_the_whole_word():
    wine = Wine()
    assert wine.inputs_from_command("wine quality red") == {
        "type": "red"
    }
    assert wine.inputs_from_command("classify white wine") == {
        "type": "white"
    }
    # "preferred" and "whitewashed" only contain the words.
    assert wine.inputs_from_command("wine quality preferred") == {}
    assert wine.inputs_from_command("whitewashed wine quality") == {}


@needs_artifact
def test_voice_command_sets_spoken_wine_type():
    def spoken_type(text):
        body = client.post("/command", json={"text": text}).json()
        assert body["model"] == "wine"
        return body["inputs"]["type"]

    assert Wine.default_input["type"] == "white"
    assert spoken_type("JarvisTEC wine quality red") == "red"
    assert spoken_type("JarvisTEC wine quality preferred") == "white"
