from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as Schema

from ..models import get_registry

router = APIRouter(prefix="/models", tags=["modelos"])


class PredictRequest(Schema):
    inputs: dict[str, Any] = {}


@router.get("")
def list_models():
    return [m.info() for m in get_registry().values()]


@router.get("/{name}")
def model_info(name: str):
    model = get_registry().get(name)
    if not model:
        raise HTTPException(404, f"Modelo '{name}' no existe")
    return model.info()


@router.post("/{name}/predict")
def predict(name: str, body: PredictRequest):
    model = get_registry().get(name)
    if not model:
        raise HTTPException(404, f"Modelo '{name}' no existe")
    try:
        return model.run(body.inputs)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(422, f"Entradas inválidas: {e}")
