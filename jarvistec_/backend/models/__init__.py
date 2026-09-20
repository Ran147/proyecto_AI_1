"""Registro automático: importa todos los módulos de esta carpeta y registra
cada subclase concreta de BaseModel."""
import importlib
import pkgutil
from .base import BaseModel

_REGISTRY: dict[str, BaseModel] = {}


def _discover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_") or mod.name == "base":
            continue
        importlib.import_module(f"{__name__}.{mod.name}")
    stack = list(BaseModel.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__abstractmethods__ or not cls.name:
            continue
        if cls.name in _REGISTRY and type(_REGISTRY[cls.name]) is not cls:
            raise RuntimeError(f"Nombre de modelo duplicado: {cls.name}")
        _REGISTRY.setdefault(cls.name, cls())


def get_registry() -> dict[str, BaseModel]:
    if not _REGISTRY:
        _discover()
    return _REGISTRY
