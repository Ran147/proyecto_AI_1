"""Model registry: validates models and keeps one instance per ``name``.

The registry does not discover models by itself; it receives the explicit
list of classes in ``backend.models.MODEL_CLASSES``. Validation errors are
raised when the registry is built, that is, when the API starts.
"""
import re
from collections.abc import Iterable, Iterator

from ..models.base import BaseModel
from .errors import DuplicateModelError, RegistrationError

# Names are part of the URL /models/{name}.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ModelRegistry:
    """Ordered collection of model instances indexed by name."""

    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}

    @classmethod
    def from_classes(
        cls, classes: Iterable[type[BaseModel]]
    ) -> "ModelRegistry":
        """Instantiate and register every class in ``classes``."""
        registry = cls()
        for model_cls in classes:
            registry.register(model_cls())
        return registry

    def register(self, model: BaseModel) -> None:
        """Add ``model``; reject invalid or duplicated names."""
        model_type = type(model).__name__
        if not isinstance(model, BaseModel):
            raise RegistrationError(
                f"{model_type} does not inherit from BaseModel"
            )
        if not NAME_PATTERN.match(model.name or ""):
            raise RegistrationError(
                f"{model_type}: invalid name {model.name!r}; use lowercase "
                "letters, digits, '_' or '-' and no spaces"
            )
        if model.name in self._models:
            other = type(self._models[model.name]).__name__
            raise DuplicateModelError(
                f"Duplicate model name '{model.name}' "
                f"({other} and {model_type})"
            )
        self._models[model.name] = model

    def get(self, name: str) -> BaseModel | None:
        """Return the model registered as ``name``, or None."""
        return self._models.get(name)

    def names(self) -> list[str]:
        """Return the registered names in registration order."""
        return list(self._models)

    def __iter__(self) -> Iterator[BaseModel]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)
