"""Trained ``car_price`` model.

Tests that run the model are skipped when artifacts/car_price.joblib is
missing (training/car_price/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.car_price import CarPrice

ARTIFACT = CarPrice().artifact_path()

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/car_price/notebook.ipynb",
)
client = TestClient(app)


def predict(inputs=None):
    response = client.post(
        "/models/car_price/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def test_registered_as_real_model():
    info = get_orchestrator().describe("car_price")
    assert info["demo"] is False
    assert set(info["input_fields"]) == set(CarPrice.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        CarPrice,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([CarPrice]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("car_price")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"fuel_type": "Electric"},
        {"present_price": 0},
        {"kms_driven": -1},
    ):
        response = client.post(
            "/models/car_price/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    # A used vehicle sells below its showroom price.
    assert 0 < result["selling_price_lakh"] < body["inputs"]["present_price"]
    assert result["selling_price_inr"] == pytest.approx(
        result["selling_price_lakh"] * 100_000, abs=1000
    )


@needs_artifact
def test_predict_partial_input_uses_defaults():
    inputs = predict({"year": 2017, "fuel_type": "Diesel"})["inputs"]
    assert inputs["year"] == 2017
    assert inputs["fuel_type"] == "Diesel"
    assert inputs["present_price"] == CarPrice.default_input["present_price"]


@needs_artifact
def test_newer_and_pricier_vehicles_are_worth_more():
    def price(**inputs):
        return predict(inputs)["result"]["selling_price_lakh"]

    assert price(year=2017) > price(year=2008)
    assert price(present_price=20) > price(present_price=5)


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, car price",
    "JarvisTEC price of the car",
    "Jarvis tec price of my car",
    "JarvisTEC, used car price",
    "Jarvis tec how much is my car worth",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "car_price"
    assert body["result"]["selling_price_lakh"] > 0
    assert "Indian rupees" in body["message"]
    assert "lakh" in body["message"]
