"""Public HTTP API with the default model registry."""
from fastapi.testclient import TestClient

from backend.core.orchestrator import get_orchestrator
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert {"bitcoin", "wine"} <= set(response.json()["models"])


def test_predict_direct():
    response = client.post(
        "/models/bitcoin/predict", json={"inputs": {"last_price": 100}}
    )
    assert response.json()["result"]["predicted_price"] == 101.0


def test_command_with_wake_word():
    response = client.post(
        "/command", json={"text": "JarvisTEC, precio del bitcoin para mañana"}
    )
    body = response.json()
    assert body["matched"]
    assert body["model"] == "bitcoin"
    assert body["wake_word"]


def test_command_fuzzy():
    # Routing only: running `wine` needs its artifact (see test_wine.py).
    orchestrator = get_orchestrator()
    assert orchestrator.match("Jarvis tec calidad del vinno")[0] == "wine"


def test_command_unknown():
    response = client.post("/command", json={"text": "cuéntame un chiste"})
    assert response.json()["matched"] is False


def test_unknown_model_404():
    response = client.post("/models/xyz/predict", json={})
    assert response.status_code == 404
