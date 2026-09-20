from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel as Schema

from ..models import get_registry
from ..services.intent_router import match_command, strip_wake_word

router = APIRouter(tags=["comandos"])


class CommandRequest(Schema):
    text: str
    inputs: dict[str, Any] = {}


@router.post("/command")
def command(body: CommandRequest):
    _, had_wake = strip_wake_word(body.text)
    model, score = match_command(body.text)
    if model is None:
        return {
            "matched": False,
            "wake_word": had_wake,
            "message": "No entendí el comando. Prueba con: "
            + ", ".join(m.keywords[0] for m in get_registry().values() if m.keywords),
        }
    out = model.run(body.inputs)
    return {"matched": True, "wake_word": had_wake, "score": round(score, 2), **out}
