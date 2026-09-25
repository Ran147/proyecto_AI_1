"""Registro de modelos: valida y guarda una instancia por `name`.

No descubre nada por su cuenta; recibe la lista explícita de clases
(backend/models/__init__.py -> MODEL_CLASSES). Los errores se lanzan al
construirlo, es decir, al iniciar la API.
"""
import re
from collections.abc import Iterable, Iterator

from ..models.base import BaseModel
from .errors import DuplicateModelError, RegistrationError

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")  # se usa en la URL /models/{name}


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}

    @classmethod
    def from_classes(cls, classes: Iterable[type[BaseModel]]) -> "ModelRegistry":
        registry = cls()
        for model_cls in classes:
            registry.register(model_cls())
        return registry

    def register(self, model: BaseModel) -> None:
        if not isinstance(model, BaseModel):
            raise RegistrationError(f"{type(model).__name__} no hereda de BaseModel")
        if not NAME_PATTERN.match(model.name or ""):
            raise RegistrationError(
                f"{type(model).__name__}: `name` inválido ({model.name!r}); "
                "usar minúsculas, dígitos, '_' o '-' y sin espacios")
        if model.name in self._models:
            other = type(self._models[model.name]).__name__
            raise DuplicateModelError(
                f"Nombre de modelo duplicado: '{model.name}' ({other} y {type(model).__name__})")
        self._models[model.name] = model

    def get(self, name: str) -> BaseModel | None:
        return self._models.get(name)

    def names(self) -> list[str]:
        return list(self._models)

    def __iter__(self) -> Iterator[BaseModel]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)
