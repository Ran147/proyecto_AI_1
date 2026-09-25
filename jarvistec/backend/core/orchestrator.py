"""Orquestador de modelos: ÚNICO punto de acceso del backend a los modelos de ML.

Routers y servicios no tocan el registro ni llaman a predict() directamente;
piden al orquestador que liste, describa, ejecute o enrute un comando. Aquí se
concentran la carga perezosa, la validación de entradas y el formato de salida.
"""
import logging
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from ..models import MODEL_CLASSES
from ..models.base import BaseModel
from ..services.intent_router import match_command, normalize, strip_wake_word
from .errors import (
    InvalidInputError, ModelNotFoundError, ModelUnavailableError, RegistrationError,
)
from .registry import ModelRegistry

log = logging.getLogger(__name__)


class ModelOrchestrator:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._check_keywords()

    def _check_keywords(self) -> None:
        """Una misma frase en dos modelos haría ambiguo el enrutamiento de comandos."""
        owners: dict[str, str] = {}
        for model in self._registry:
            for kw in model.keywords:
                key = normalize(kw)
                if key in owners and owners[key] != model.name:
                    raise RegistrationError(
                        f"La frase de comando '{kw}' está en '{owners[key]}' y en '{model.name}'")
                owners[key] = model.name

    # ---------- catálogo ----------
    def names(self) -> list[str]:
        return self._registry.names()

    def list_models(self) -> list[dict[str, Any]]:
        return [m.info() for m in self._registry]

    def describe(self, name: str) -> dict[str, Any]:
        return self._get(name).info()

    def command_phrases(self) -> list[str]:
        return [kw for m in self._registry for kw in m.keywords]

    # ---------- ejecución ----------
    def predict(self, name: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._run(self._get(name), inputs)

    def handle_command(self, text: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        _, had_wake = strip_wake_word(text)
        model, score = match_command(text, self._registry)
        if model is None:
            examples = ", ".join(m.keywords[0] for m in self._registry if m.keywords)
            return {"matched": False, "wake_word": had_wake,
                    "message": "No entendí el comando. Prueba con: " + examples}
        out = self._run(model, inputs)
        return {"matched": True, "wake_word": had_wake, "score": round(score, 2), **out}

    def _get(self, name: str) -> BaseModel:
        model = self._registry.get(name)
        if model is None:
            raise ModelNotFoundError(f"Modelo '{name}' no existe")
        return model

    def _run(self, model: BaseModel, inputs: dict[str, Any] | None) -> dict[str, Any]:
        try:
            model.ensure_loaded()
        except Exception as e:
            log.exception("No se pudo cargar el modelo '%s'", model.name)
            raise ModelUnavailableError(f"El modelo '{model.name}' no está disponible: {e}") from e
        data = self._validate(model, {**model.default_input, **(inputs or {})})
        try:
            result = model.predict(data)
        except (KeyError, ValueError, TypeError) as e:
            raise InvalidInputError(f"Entradas inválidas: {e}") from e
        return {"model": model.name, "inputs": data, "result": result, "message": model.speak(result)}

    @staticmethod
    def _validate(model: BaseModel, data: dict[str, Any]) -> dict[str, Any]:
        if model.input_schema is None:
            return data
        try:
            return model.input_schema.model_validate(data).model_dump()
        except ValidationError as e:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'inputs'}: {err['msg']}" for err in e.errors())
            raise InvalidInputError(f"Entradas inválidas: {detail}") from e


@lru_cache(maxsize=None)
def get_orchestrator() -> ModelOrchestrator:
    """Instancia compartida con los modelos de MODEL_CLASSES. Usada como dependencia de FastAPI."""
    return ModelOrchestrator(ModelRegistry.from_classes(MODEL_CLASSES))
