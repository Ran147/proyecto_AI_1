from typing import Any

from ..base import BaseModel
from .schema import BitcoinInputs


class DemoBitcoin(BaseModel):
    name = "bitcoin"
    title = "Precio de Bitcoin (demo, sin entrenar)"
    description = "DEMO, no es un modelo entrenado: aplica un +1% al último precio. Ejemplo de serie temporal."
    task_type = "serie_temporal"
    is_demo = True
    keywords = ["precio del bitcoin", "bitcoin", "precio de bitcoin para mañana"]
    input_schema = BitcoinInputs
    default_input = {"last_price": 60000.0}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        last = float(inputs["last_price"])
        return {"predicted_price": round(last * 1.01, 2)}

    def speak(self, result: dict[str, Any]) -> str:
        return f"Estimo que el bitcoin costará unos {result['predicted_price']:,.2f} dólares mañana."
