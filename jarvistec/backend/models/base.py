"""Interfaz común que deben cumplir TODOS los modelos de JarvisTEC.

Cada modelo vive en su propio paquete (backend/models/<nombre>/) con un
adaptador que hereda de BaseModel y un esquema de entradas. Se registra
agregando su clase a MODEL_CLASSES en backend/models/__init__.py; el
orquestador (backend/core/orchestrator.py) es quien lo carga y lo ejecuta.
"""
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel as Schema

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


class BaseModel(ABC):
    # --- Metadatos (obligatorios) ---
    name: str = ""            # identificador único, sin espacios: "vino", "bitcoin"
    title: str = ""           # nombre legible
    description: str = ""
    task_type: str = ""       # "clasificacion" | "regresion" | "serie_temporal" | "recomendacion"
    keywords: list[str] = []  # frases de comando de voz: ["precio del bitcoin", "bitcoin"]
    is_demo: bool = False     # True = lógica de juguete, NO es un modelo entrenado

    # --- Entradas ---
    input_schema: type[Schema] | None = None  # valida las entradas antes de predict()
    input_fields: dict[str, str] = {}   # si hay input_schema se deriva de él
    default_input: dict[str, Any] = {}  # se usa cuando el comando de voz no trae datos

    def __init__(self) -> None:
        self._loaded = False
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Carga el modelo entrenado (joblib, etc.) desde artifacts/. Opcional."""

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if not self._loaded:
                self.load()
                self._loaded = True

    def artifact_path(self, suffix: str = ".joblib") -> Path:
        """Ruta convencional del artefacto que genera el notebook: artifacts/<name><suffix>."""
        return ARTIFACTS_DIR / f"{self.name}{suffix}"

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Recibe un diccionario de entradas (ya validado) y devuelve un diccionario serializable."""

    def speak(self, result: dict[str, Any]) -> str:
        """Texto que Jarvis dirá/mostrará al usuario. Sobrescribir para algo amigable."""
        return str(result)

    def describe_inputs(self) -> dict[str, str]:
        if self.input_fields or self.input_schema is None:
            return self.input_fields
        fields = {}
        for field, spec in self.input_schema.model_fields.items():
            kind = getattr(spec.annotation, "__name__", str(spec.annotation))
            fields[field] = f"{kind}, {spec.description}" if spec.description else kind
        return fields

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type,
            "demo": self.is_demo,
            "keywords": self.keywords,
            "input_fields": self.describe_inputs(),
            "default_input": self.default_input,
        }
