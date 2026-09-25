from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel as Schema

from ..core.orchestrator import ModelOrchestrator, get_orchestrator

router = APIRouter(tags=["comandos"])


class CommandRequest(Schema):
    text: str
    inputs: dict[str, Any] = {}


@router.post("/command")
def command(body: CommandRequest, orch: ModelOrchestrator = Depends(get_orchestrator)):
    return orch.handle_command(body.text, body.inputs)
