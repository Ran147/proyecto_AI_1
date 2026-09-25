from typing import Any

from ..base import BaseModel
from .schema import VinoInputs


class DemoVino(BaseModel):
    name = "vino"
    title = "Calidad del vino (demo, sin entrenar)"
    description = ("DEMO, no es un modelo entrenado: regla fija "
                   "(alcohol alto y acidez volátil baja => buena).")
    task_type = "clasificacion"
    is_demo = True
    keywords = ["calidad del vino", "vino", "clasificar vino"]
    input_schema = VinoInputs
    default_input = {"alcohol": 11.5, "volatile_acidity": 0.4}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        good = float(inputs["alcohol"]) >= 11 and float(inputs["volatile_acidity"]) <= 0.6
        return {"quality": "buena" if good else "regular"}

    def speak(self, result: dict[str, Any]) -> str:
        return f"Ese vino parece ser de calidad {result['quality']}."
