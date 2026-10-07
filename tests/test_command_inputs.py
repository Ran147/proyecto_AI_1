"""Inputs sent along with a command, and the options exposed to the UI."""
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel as Schema

from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.base import BaseModel


class ShapeInputs(Schema):
    shape: Literal["circle", "square"]
    size: float
    note: Literal["a", "b"] | None = None


class Shape(BaseModel):
    name = "shape"
    keywords = ["describe the shape"]
    input_schema = ShapeInputs
    default_input = {"shape": "circle", "size": 1.0, "note": None}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs


class Other(BaseModel):
    name = "other"
    keywords = ["something else"]
    default_input = {"value": 1}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs


def make_orchestrator() -> ModelOrchestrator:
    return ModelOrchestrator(ModelRegistry.from_classes([Shape, Other]))


def test_input_options_list_fixed_choices():
    info = make_orchestrator().describe("shape")
    assert info["input_options"] == {
        "shape": ["circle", "square"],
        "note": ["a", "b"],
    }
    assert make_orchestrator().describe("other")["input_options"] == {}


def test_command_uses_inputs_of_the_matched_model_only():
    output = make_orchestrator().handle_command(
        "JarvisTEC describe the shape",
        inputs_by_model={
            "shape": {"shape": "square"},
            "other": {"value": 99},
        },
    )
    assert output["model"] == "shape"
    assert output["result"] == {"shape": "square", "size": 1.0, "note": None}


def test_explicit_inputs_win_over_inputs_by_model():
    output = make_orchestrator().handle_command(
        "JarvisTEC describe the shape",
        {"size": 5},
        {"shape": {"shape": "square", "size": 2}},
    )
    assert output["result"]["size"] == 5.0
    assert output["result"]["shape"] == "square"


def test_command_endpoint_accepts_inputs_by_model():
    app.dependency_overrides[get_orchestrator] = make_orchestrator
    try:
        response = TestClient(app).post(
            "/command",
            json={
                "text": "JarvisTEC describe the shape",
                "inputs_by_model": {"shape": {"size": 3}},
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["result"]["size"] == 3.0


# ---------- Values said inside the command ----------
def run(text, **kwargs):
    return get_orchestrator().handle_command(text, **kwargs)


def needs(name):
    path = get_orchestrator()._registry.get(name).artifact_path()
    return pytest.mark.skipif(
        not path.exists(), reason=f"{path.name} missing"
    )


@needs("avocado")
def test_spoken_avocado_type_is_used():
    output = run("JarvisTEC, organic avocado price")
    assert output["inputs"]["type"] == "organic"
    assert run("JarvisTEC avocado price")["inputs"]["type"] == (
        "conventional"
    )


@needs("wine")
def test_spoken_wine_type_beats_the_form_value():
    output = run(
        "JarvisTEC red wine quality",
        inputs_by_model={"wine": {"type": "white", "alcohol": 13.0}},
    )
    assert output["inputs"]["type"] == "red"
    assert output["inputs"]["alcohol"] == 13.0


@needs("movies")
def test_spoken_movie_title_is_used():
    output = run("Jarvis, recommend me a movie like Jurassic Park")
    assert output["model"] == "movies"
    assert output["result"]["title"].startswith("Jurassic Park")
    default = run("JarvisTEC recommend me a movie")
    assert default["result"]["title"].startswith("Toy Story")
