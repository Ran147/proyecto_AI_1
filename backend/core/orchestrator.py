"""Model orchestrator: the ONLY entry point from the backend to the models.

Routers and services never access the registry or call ``predict()``
directly; they ask the orchestrator to list, describe, run or route a
command. Lazy loading, input validation and the response format live here.

The ``message`` field and the error messages are Spanish because the
frontend shows them verbatim to the end user.
"""
import logging
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from ..models import MODEL_CLASSES
from ..models.base import BaseModel
from ..services.intent_router import (
    match_command,
    normalize,
    strip_wake_word,
)
from .errors import (
    InvalidInputError,
    ModelNotFoundError,
    ModelUnavailableError,
    RegistrationError,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """Registers, validates and runs the models of a registry."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._check_keywords()

    def _check_keywords(self) -> None:
        """Reject a phrase shared by two models: routing would be ambiguous."""
        owners: dict[str, str] = {}
        for model in self._registry:
            for keyword in model.keywords:
                key = normalize(keyword)
                if key in owners and owners[key] != model.name:
                    raise RegistrationError(
                        f"Command phrase '{keyword}' is used by "
                        f"'{owners[key]}' and '{model.name}'"
                    )
                owners[key] = model.name

    # ---------- Catalog ----------
    def names(self) -> list[str]:
        """Return the registered model names."""
        return self._registry.names()

    def list_models(self) -> list[dict[str, Any]]:
        """Return the metadata of every registered model."""
        return [model.info() for model in self._registry]

    def describe(self, name: str) -> dict[str, Any]:
        """Return the metadata of one model."""
        return self._get(name).info()

    def command_phrases(self) -> list[str]:
        """Return every voice command phrase of every model."""
        return [kw for model in self._registry for kw in model.keywords]

    # ---------- Execution ----------
    def predict(
        self, name: str, inputs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the model registered as ``name`` on ``inputs``."""
        return self._run(self._get(name), inputs)

    def match(self, text: str) -> tuple[str | None, float]:
        """Return the model a command routes to, without running it."""
        model, score = match_command(text, self._registry)
        return (model.name if model else None), score

    def handle_command(
        self, text: str, inputs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Route a transcribed command to a model and run it."""
        _, had_wake_word = strip_wake_word(text)
        model, score = match_command(text, self._registry)
        if model is None:
            examples = ", ".join(
                m.keywords[0] for m in self._registry if m.keywords
            )
            return {
                "matched": False,
                "wake_word": had_wake_word,
                "message": "No entendí el comando. Prueba con: " + examples,
            }
        output = self._run(model, inputs)
        return {
            "matched": True,
            "wake_word": had_wake_word,
            "score": round(score, 2),
            **output,
        }

    def _get(self, name: str) -> BaseModel:
        model = self._registry.get(name)
        if model is None:
            raise ModelNotFoundError(f"Modelo '{name}' no existe")
        return model

    def _run(
        self, model: BaseModel, inputs: dict[str, Any] | None
    ) -> dict[str, Any]:
        # Validate before loading: bad requests never trigger a model load.
        data = self._validate(model, {**model.default_input, **(inputs or {})})
        try:
            model.ensure_loaded()
        except Exception as exc:
            logger.exception("Could not load model '%s'", model.name)
            raise ModelUnavailableError(
                f"El modelo '{model.name}' no está disponible: {exc}"
            ) from exc
        try:
            result = model.predict(data)
        except (KeyError, ValueError, TypeError) as exc:
            raise InvalidInputError(f"Entradas inválidas: {exc}") from exc
        return {
            "model": model.name,
            "inputs": data,
            "result": result,
            "message": model.speak(result),
        }

    @staticmethod
    def _validate(model: BaseModel, data: dict[str, Any]) -> dict[str, Any]:
        if model.input_schema is None:
            return data
        try:
            return model.input_schema.model_validate(data).model_dump()
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'inputs'}: "
                f"{err['msg']}"
                for err in exc.errors()
            )
            raise InvalidInputError(f"Entradas inválidas: {detail}") from exc


@lru_cache(maxsize=None)
def get_orchestrator() -> ModelOrchestrator:
    """Shared orchestrator over MODEL_CLASSES; used as a FastAPI dependency."""
    return ModelOrchestrator(ModelRegistry.from_classes(MODEL_CLASSES))
