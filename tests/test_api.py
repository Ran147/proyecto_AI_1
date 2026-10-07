"""Public HTTP API with the default model registry."""
import pytest
from fastapi.testclient import TestClient

from backend.core.orchestrator import get_orchestrator
from backend.main import app
from backend.models.bitcoin import Bitcoin

client = TestClient(app)
# Tests that run the model need its artifact.
needs_bitcoin = pytest.mark.skipif(
    not Bitcoin().artifact_path().exists(),
    reason="bitcoin.joblib missing; run training/bitcoin/notebook.ipynb",
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert {"bitcoin", "wine"} <= set(response.json()["models"])


@needs_bitcoin
def test_predict_direct():
    response = client.post(
        "/models/bitcoin/predict", json={"inputs": {"last_price": 100}}
    )
    # A next-day forecast stays close to the last price.
    assert 90 < response.json()["result"]["predicted_price"] < 110


@needs_bitcoin
def test_command_with_wake_word():
    response = client.post(
        "/command", json={"text": "JarvisTEC, bitcoin price tomorrow"}
    )
    body = response.json()
    assert body["matched"]
    assert body["model"] == "bitcoin"
    assert body["wake_word"]


def test_command_fuzzy():
    # Routing only: running `wine` needs its artifact (see test_wine.py).
    orchestrator = get_orchestrator()
    assert orchestrator.match("Jarvis tec wine qualiti")[0] == "wine"


def test_command_unknown():
    response = client.post("/command", json={"text": "tell me a joke"})
    assert response.json()["matched"] is False


def test_unknown_model_404():
    response = client.post("/models/xyz/predict", json={})
    assert response.status_code == 404
