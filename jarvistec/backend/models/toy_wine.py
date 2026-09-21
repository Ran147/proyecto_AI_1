"""MODELO DE JUGUETE (reemplazar por el real). Clasificación basada en reglas."""
from typing import Any
from .base import BaseModel


class ToyWine(BaseModel):
    name = "vino"
    title = "Calidad del vino (juguete)"
    description = "Clasificación falsa por reglas: alcohol alto y acidez volátil baja => buena."
    task_type = "clasificacion"
    keywords = ["calidad del vino", "vino", "clasificar vino"]
    input_fields = {
        "alcohol": "float, % de alcohol",
        "volatile_acidity": "float, acidez volátil",
    }
    default_input = {"alcohol": 11.5, "volatile_acidity": 0.4}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        good = float(inputs["alcohol"]) >= 11 and float(inputs["volatile_acidity"]) <= 0.6
        return {"quality": "buena" if good else "regular"}

    def speak(self, result: dict[str, Any]) -> str:
        return f"Ese vino parece ser de calidad {result['quality']}."
