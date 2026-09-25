"""Model catalog and direct prediction endpoints."""
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel as Schema

from ..core.orchestrator import ModelOrchestrator, get_orchestrator

router = APIRouter(prefix="/models", tags=["models"])


class PredictRequest(Schema):
    """Body of POST /models/{name}/predict."""

    inputs: dict[str, Any] = {}


@router.get("")
def list_models(
    orchestrator: ModelOrchestrator = Depends(get_orchestrator),
):
    """Return the metadata of every registered model."""
    return orchestrator.list_models()


@router.get("/{name}")
def model_info(
    name: str,
    orchestrator: ModelOrchestrator = Depends(get_orchestrator),
):
    """Return the metadata of one model."""
    return orchestrator.describe(name)


@router.post("/{name}/predict")
def predict(
    name: str,
    body: PredictRequest,
    orchestrator: ModelOrchestrator = Depends(get_orchestrator),
):
    """Run a model; missing inputs are filled from its default_input."""
    return orchestrator.predict(name, body.inputs)
