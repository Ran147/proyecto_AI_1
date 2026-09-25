from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel as Schema

from ..core.orchestrator import ModelOrchestrator, get_orchestrator

router = APIRouter(prefix="/models", tags=["modelos"])


class PredictRequest(Schema):
    inputs: dict[str, Any] = {}


@router.get("")
def list_models(orch: ModelOrchestrator = Depends(get_orchestrator)):
    return orch.list_models()


@router.get("/{name}")
def model_info(name: str, orch: ModelOrchestrator = Depends(get_orchestrator)):
    return orch.describe(name)


@router.post("/{name}/predict")
def predict(name: str, body: PredictRequest, orch: ModelOrchestrator = Depends(get_orchestrator)):
    return orch.predict(name, body.inputs)
