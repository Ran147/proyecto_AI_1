"""Interfaz común que deben cumplir TODOS los modelos de JarvisTEC.

Para agregar un modelo nuevo: crear un archivo en backend/models/ con una
clase que herede de BaseModel. Se registra automáticamente al iniciar la API.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


class BaseModel(ABC):
    # --- Metadatos (obligatorios) ---
    name: str = ""            # identificador único, sin espacios: "vino", "bitcoin"
    title: str = ""           # nombre legible
    description: str = ""
    task_type: str = ""       # "clasificacion" | "regresion" | "serie_temporal" | "recomendacion"
    keywords: list[str] = []  # frases de comando de voz: ["precio del bitcoin", "bitcoin"]

    # --- Entradas ---
    input_fields: dict[str, str] = {}   # {"alcohol": "float, % de alcohol"}
    default_input: dict[str, Any] = {}  # se usa cuando el comando de voz no trae datos

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        """Carga el modelo entrenado (joblib, etc.) desde artifacts/. Opcional."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Recibe un diccionario de entradas y devuelve un diccionario serializable."""

    def speak(self, result: dict[str, Any]) -> str:
        """Texto que Jarvis dirá/mostrará al usuario. Sobrescribir para algo amigable."""
        return str(result)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type,
            "keywords": self.keywords,
            "input_fields": self.input_fields,
            "default_input": self.default_input,
        }

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ensure_loaded()
        data = {**self.default_input, **(inputs or {})}
        result = self.predict(data)
        return {"model": self.name, "inputs": data, "result": result, "message": self.speak(result)}
