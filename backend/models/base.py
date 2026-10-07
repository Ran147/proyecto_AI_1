"""Common interface that EVERY JarvisTEC model must implement.

Each model lives in its own package (backend/models/<name>/) with an adapter
that inherits from BaseModel and an input schema. It is activated by adding
its class to MODEL_CLASSES in backend/models/__init__.py; the orchestrator
(backend/core/orchestrator.py) loads and runs it.
"""
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel as Schema

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


class BaseModel(ABC):
    """Adapter between the orchestrator and one trained model."""

    # ---------- Metadata ----------
    # Unique identifier used in /models/{name}: lowercase, no spaces.
    name: str = ""
    # Display name shown in the UI.
    title: str = ""
    # Technical description for API consumers.
    description: str = ""
    # "classification" | "regression" | "time_series" | "recommendation"
    task_type: str = ""
    # Voice command phrases, lowercase, e.g. ["bitcoin price"].
    keywords: list[str] = []
    # The same commands in Spanish, e.g. ["precio del bitcoin"]. Commands are
    # understood in both languages (accents are ignored when matching); the
    # answers are in English.
    keywords_es: list[str] = []
    # True only for toy logic that is NOT a trained model.
    is_demo: bool = False

    # ---------- Inputs ----------
    # Validates inputs before predict(); input_fields is derived from it.
    input_schema: type[Schema] | None = None
    # Explicit field descriptions; only needed without an input_schema.
    input_fields: dict[str, str] = {}
    # Used when a voice command carries no data.
    default_input: dict[str, Any] = {}

    def __init__(self) -> None:
        self._loaded = False
        self._load_lock = threading.Lock()

    def phrases(self) -> list[str]:
        """Return every command phrase, English first."""
        return [*self.keywords, *self.keywords_es]

    def load(self) -> None:
        """Load the trained model (joblib, etc.) from artifacts/. Optional."""

    def ensure_loaded(self) -> None:
        """Call load() exactly once, even under concurrent requests."""
        if self._loaded:
            return
        with self._load_lock:
            if not self._loaded:
                self.load()
                self._loaded = True

    def artifact_path(self, suffix: str = ".joblib") -> Path:
        """Conventional artifact path: artifacts/<name><suffix>."""
        return ARTIFACTS_DIR / f"{self.name}{suffix}"

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Take validated inputs and return a JSON-serializable dict."""

    def inputs_from_command(self, text: str) -> dict[str, Any]:
        """Extract inputs said in a command, e.g. "organic avocado".

        ``text`` is the normalized command (lowercase, no accents, no wake
        word). Optional: by default a command carries no data.
        """
        return {}

    def speak(self, result: dict[str, Any]) -> str:
        """Return the sentence JarvisTEC says to the user."""
        return str(result)

    def describe_inputs(self) -> dict[str, str]:
        """Return ``{field: "type, description"}`` for every input."""
        if self.input_fields or self.input_schema is None:
            return self.input_fields
        fields = {}
        for field, spec in self.input_schema.model_fields.items():
            kind = getattr(spec.annotation, "__name__", str(spec.annotation))
            fields[field] = (
                f"{kind}, {spec.description}" if spec.description else kind
            )
        return fields

    def input_options(self) -> dict[str, list[Any]]:
        """Return the allowed values of the inputs limited to a fixed set."""
        if self.input_schema is None:
            return {}
        properties = self.input_schema.model_json_schema().get(
            "properties", {}
        )
        options = {}
        for field, spec in properties.items():
            # Optional fields wrap the enum in an "anyOf" list.
            for variant in [spec, *spec.get("anyOf", [])]:
                if "enum" in variant:
                    options[field] = variant["enum"]
        return options

    def info(self) -> dict[str, Any]:
        """Return the public metadata exposed by GET /models."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type,
            "demo": self.is_demo,
            "keywords": self.keywords,
            "keywords_es": self.keywords_es,
            "input_fields": self.describe_inputs(),
            "input_options": self.input_options(),
            "default_input": self.default_input,
        }
