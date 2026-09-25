"""Voice/text command endpoint."""
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel as Schema

from ..core.orchestrator import ModelOrchestrator, get_orchestrator

router = APIRouter(tags=["commands"])


class CommandRequest(Schema):
    """Body of POST /command."""

    text: str
    inputs: dict[str, Any] = {}


@router.post("/command")
def command(
    body: CommandRequest,
    orchestrator: ModelOrchestrator = Depends(get_orchestrator),
):
    """Route transcribed text to the matching model and run it."""
    return orchestrator.handle_command(body.text, body.inputs)
