from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert {"bitcoin", "vino"} <= set(r.json()["models"])


def test_predict_direct():
    r = client.post("/models/vino/predict", json={"inputs": {"alcohol": 9, "volatile_acidity": 0.9}})
    assert r.json()["result"]["quality"] == "regular"


def test_command_with_wake_word():
    r = client.post("/command", json={"text": "JarvisTEC, precio del bitcoin para mañana"})
    body = r.json()
    assert body["matched"] and body["model"] == "bitcoin" and body["wake_word"]


def test_command_fuzzy():
    r = client.post("/command", json={"text": "Jarvis tec calidad del vinno"})
    assert r.json()["model"] == "vino"


def test_command_unknown():
    assert client.post("/command", json={"text": "cuéntame un chiste"}).json()["matched"] is False


def test_unknown_model_404():
    assert client.post("/models/xyz/predict", json={}).status_code == 404
