"""Registry, orchestrator and their API integration; no external services."""
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel as Schema

from backend.core.errors import (
    DuplicateModelError,
    InvalidInputError,
    ModelNotFoundError,
    RegistrationError,
)
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models import MODEL_CLASSES
from backend.models.base import BaseModel


class EchoInputs(Schema):
    x: float


class Echo(BaseModel):
    name = "echo"
    title = "Echo"
    keywords = ["repite el numero"]
    input_schema = EchoInputs
    default_input = {"x": 1.0}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"double": inputs["x"] * 2}

    def speak(self, result: dict[str, Any]) -> str:
        return f"El doble es {result['double']}."


class OtherEcho(Echo):
    keywords = ["otra frase"]


class Broken(BaseModel):
    name = "broken"
    keywords = ["modelo roto"]

    def load(self) -> None:
        raise FileNotFoundError("artifacts/broken.joblib")

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("must not run")


def make_orchestrator(*classes: type[BaseModel]) -> ModelOrchestrator:
    return ModelOrchestrator(ModelRegistry.from_classes(classes))


# ---------- Registration ----------
def test_default_registry_has_isolated_packages():
    orchestrator = get_orchestrator()
    assert set(orchestrator.names()) == {"bitcoin", "wine"}
    for cls in MODEL_CLASSES:
        # Every model lives in its own package backend.models.<package>.
        assert cls.__module__.startswith("backend.models.")
        assert cls.__module__.count(".") == 3
    demo_flags = {m["name"]: m["demo"] for m in orchestrator.list_models()}
    assert demo_flags == {"bitcoin": True, "wine": False}


def test_duplicate_name_rejected():
    with pytest.raises(DuplicateModelError, match="echo"):
        make_orchestrator(Echo, OtherEcho)


def test_invalid_name_rejected():
    class Spaced(Echo):
        name = "with spaces"

    with pytest.raises(RegistrationError):
        make_orchestrator(Spaced)


def test_shared_keyword_rejected():
    class Clash(Echo):
        name = "other"

    with pytest.raises(RegistrationError, match="repite el numero"):
        make_orchestrator(Echo, Clash)


def test_input_fields_derived_from_schema():
    info = get_orchestrator().describe("bitcoin")
    assert info["input_fields"] == {"last_price": "float, last price in USD"}


# ---------- Execution ----------
def test_predict_merges_defaults_and_validates():
    orchestrator = make_orchestrator(Echo)
    assert orchestrator.predict("echo")["result"] == {"double": 2.0}
    output = orchestrator.predict("echo", {"x": "4"})
    assert output["inputs"] == {"x": 4.0}
    assert output["message"] == "El doble es 8.0."


def test_predict_invalid_input():
    with pytest.raises(InvalidInputError, match="x"):
        make_orchestrator(Echo).predict("echo", {"x": "not a number"})


def test_predict_unknown_model():
    with pytest.raises(ModelNotFoundError):
        make_orchestrator(Echo).predict("xyz")


def test_command_executes_matched_model():
    output = make_orchestrator(Echo, Broken).handle_command(
        "JarvisTEC repite el numero", {"x": 3}
    )
    assert output["matched"]
    assert output["wake_word"]
    assert output["model"] == "echo"
    assert output["result"] == {"double": 6.0}


def test_command_without_match_lists_examples():
    output = make_orchestrator(Echo).handle_command("cuéntame un chiste")
    assert output["matched"] is False
    assert "repite el numero" in output["message"]


# ---------- API through the orchestrator ----------
@pytest.fixture
def api():
    app.dependency_overrides[get_orchestrator] = (
        lambda: make_orchestrator(Echo, Broken)
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_goes_through_orchestrator(api):
    names = [m["name"] for m in api.get("/models").json()]
    assert names == ["echo", "broken"]
    assert api.get("/health").json()["models"] == ["echo", "broken"]
    response = api.post("/models/echo/predict", json={"inputs": {"x": 5}})
    assert response.json()["result"] == {"double": 10.0}
    response = api.post(
        "/command", json={"text": "Jarvis tec repite el numero"}
    )
    assert response.json()["model"] == "echo"


def test_api_errors(api):
    assert api.get("/models/xyz").status_code == 404
    assert api.post("/models/xyz/predict", json={}).status_code == 404

    response = api.post("/models/echo/predict", json={"inputs": {"x": "a"}})
    assert response.status_code == 422
    assert "Entradas inválidas" in response.json()["detail"]

    response = api.post("/models/broken/predict", json={})
    assert response.status_code == 503
    assert "no está disponible" in response.json()["detail"]

    response = api.post("/command", json={"text": "JarvisTEC modelo roto"})
    assert response.status_code == 503
