"""Trained ``cirrhosis`` model.

Tests that run the model are skipped when artifacts/cirrhosis.joblib is
missing (training/cirrhosis/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.cirrhosis import Cirrhosis

ARTIFACT = Cirrhosis().artifact_path()
STAGES = {1, 2, 3, 4}

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/cirrhosis/notebook.ipynb",
)
client = TestClient(app)


def test_registered_as_real_model():
    info = get_orchestrator().describe("cirrhosis")
    assert info["demo"] is False
    assert set(info["input_fields"]) == set(Cirrhosis.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Cirrhosis,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Cirrhosis]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("cirrhosis")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"sex": "X"},
        {"edema": "maybe"},
        {"age": -1},
        {"albumin": 0},
    ):
        response = client.post(
            "/models/cirrhosis/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_predict_default_input():
    response = client.post("/models/cirrhosis/predict", json={})
    assert response.status_code == 200
    body = response.json()
    json.dumps(body)  # must be serializable
    result = body["result"]
    probabilities = result["probabilities"]
    assert result["stage"] in STAGES
    assert {int(stage) for stage in probabilities} == STAGES
    assert abs(sum(probabilities.values()) - 1) < 1e-3
    assert str(result["stage"]) == max(probabilities, key=probabilities.get)


@needs_artifact
def test_predict_partial_input_uses_defaults():
    response = client.post(
        "/models/cirrhosis/predict",
        json={"inputs": {"ascites": "Y", "bilirubin": 12.0}},
    )
    assert response.status_code == 200
    inputs = response.json()["inputs"]
    assert inputs["ascites"] == "Y"
    assert inputs["bilirubin"] == 12.0
    assert inputs["albumin"] == Cirrhosis.default_input["albumin"]


@needs_artifact
def test_advanced_signs_raise_the_probability_of_stage_4():
    def stage_4(**inputs):
        response = client.post(
            "/models/cirrhosis/predict", json={"inputs": inputs}
        )
        return response.json()["result"]["probabilities"]["4"]

    mild = stage_4(
        hepatomegaly="N", bilirubin=0.6, albumin=4.0, platelets=320
    )
    advanced = stage_4(
        hepatomegaly="Y", ascites="Y", spiders="Y", edema="Y",
        bilirubin=8.0, albumin=2.8, platelets=120, prothrombin=12.5,
    )
    assert advanced > mild


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, cirrhosis stage",
    "Jarvis tec stage of cirrhosis",
    "JarvisTEC type of cirrhosis",
    "JarvisTEC, cirrhosis",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "cirrhosis"
    assert body["result"]["stage"] in STAGES
    assert "I estimate" in body["message"]
    assert f"stage {body['result']['stage']} of 4" in body["message"]
    assert "probability of" in body["message"]
    assert "not a medical diagnosis" in body["message"]


def test_keywords_are_the_english_voice_commands():
    assert Cirrhosis.keywords == [
        "cirrhosis stage",
        "stage of cirrhosis",
        "type of cirrhosis",
        "cirrhosis",
    ]


@pytest.mark.parametrize("keyword", Cirrhosis.keywords)
def test_every_keyword_routes_to_this_model(keyword):
    match = get_orchestrator().match(f"JarvisTEC {keyword}")
    assert match[0] == "cirrhosis"

