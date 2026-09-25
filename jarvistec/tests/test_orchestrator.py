"""Registro, orquestador y su integración con la API. Sin servicios externos."""
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel as Schema

from backend.core.errors import (
    DuplicateModelError, InvalidInputError, ModelNotFoundError, RegistrationError,
)
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models import MODEL_CLASSES
from backend.models.base import BaseModel


class EchoInputs(Schema):
    x: float


class Echo(BaseModel):
    name = "eco"
    title = "Eco"
    keywords = ["repite el numero"]
    input_schema = EchoInputs
    default_input = {"x": 1.0}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"doble": inputs["x"] * 2}

    def speak(self, result: dict[str, Any]) -> str:
        return f"El doble es {result['doble']}."


class OtherEcho(Echo):
    keywords = ["otra frase"]


class Broken(BaseModel):
    name = "roto"
    keywords = ["modelo roto"]

    def load(self) -> None:
        raise FileNotFoundError("artifacts/roto.joblib")

    def predict(self, inputs):
        raise AssertionError("no debería ejecutarse")


def make(*classes) -> ModelOrchestrator:
    return ModelOrchestrator(ModelRegistry.from_classes(classes))


# ---------- registro ----------
def test_default_registry_has_isolated_demo_packages():
    orch = get_orchestrator()
    assert set(orch.names()) == {"bitcoin", "vino"}
    for cls in MODEL_CLASSES:
        # cada modelo vive en su propio paquete backend.models.<paquete>
        assert cls.__module__.count(".") == 3 and cls.__module__.startswith("backend.models.")
    assert all(info["demo"] for info in orch.list_models())


def test_duplicate_name_rejected():
    with pytest.raises(DuplicateModelError, match="eco"):
        make(Echo, OtherEcho)


def test_invalid_name_rejected():
    class Spaced(Echo):
        name = "con espacios"
    with pytest.raises(RegistrationError):
        make(Spaced)


def test_shared_keyword_rejected():
    class Clash(Echo):
        name = "otro"
    with pytest.raises(RegistrationError, match="repite el numero"):
        make(Echo, Clash)


def test_input_fields_derived_from_schema():
    info = get_orchestrator().describe("vino")
    assert info["input_fields"] == {"alcohol": "float, % de alcohol",
                                    "volatile_acidity": "float, acidez volátil"}


# ---------- ejecución ----------
def test_predict_merges_defaults_and_validates():
    orch = make(Echo)
    assert orch.predict("eco")["result"] == {"doble": 2.0}
    out = orch.predict("eco", {"x": "4"})
    assert out["inputs"] == {"x": 4.0} and out["message"] == "El doble es 8.0."


def test_predict_invalid_input():
    with pytest.raises(InvalidInputError, match="x"):
        make(Echo).predict("eco", {"x": "no es numero"})


def test_predict_unknown_model():
    with pytest.raises(ModelNotFoundError):
        make(Echo).predict("xyz")


def test_command_executes_matched_model():
    out = make(Echo, Broken).handle_command("JarvisTEC repite el numero", {"x": 3})
    assert out["matched"] and out["wake_word"] and out["model"] == "eco"
    assert out["result"] == {"doble": 6.0}


def test_command_without_match_lists_examples():
    out = make(Echo).handle_command("cuéntame un chiste")
    assert out["matched"] is False and "repite el numero" in out["message"]


# ---------- API usando el orquestador ----------
@pytest.fixture
def api():
    app.dependency_overrides[get_orchestrator] = lambda: make(Echo, Broken)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_goes_through_orchestrator(api):
    assert [m["name"] for m in api.get("/models").json()] == ["eco", "roto"]
    assert api.get("/health").json()["models"] == ["eco", "roto"]
    assert api.post("/models/eco/predict", json={"inputs": {"x": 5}}).json()["result"] == {"doble": 10.0}
    r = api.post("/command", json={"text": "Jarvis tec repite el numero"})
    assert r.json()["model"] == "eco"


def test_api_errors(api):
    assert api.get("/models/xyz").status_code == 404
    assert api.post("/models/xyz/predict", json={}).status_code == 404
    r = api.post("/models/eco/predict", json={"inputs": {"x": "abc"}})
    assert r.status_code == 422 and "Entradas inválidas" in r.json()["detail"]
    r = api.post("/models/roto/predict", json={})
    assert r.status_code == 503 and "no está disponible" in r.json()["detail"]
    assert api.post("/command", json={"text": "JarvisTEC modelo roto"}).status_code == 503
