"""MODELO DE JUGUETE (reemplazar por el real). Sirve para probar el flujo completo."""
from typing import Any
from .base import BaseModel


class ToyBitcoin(BaseModel):
    name = "bitcoin"
    title = "Precio de Bitcoin (juguete)"
    description = "Predicción falsa: aplica un +1% al último precio. Ejemplo de serie temporal."
    task_type = "serie_temporal"
    keywords = ["precio del bitcoin", "bitcoin", "precio de bitcoin para mañana"]
    input_fields = {"last_price": "float, último precio en USD"}
    default_input = {"last_price": 60000.0}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        last = float(inputs["last_price"])
        return {"predicted_price": round(last * 1.01, 2)}

    def speak(self, result: dict[str, Any]) -> str:
        return f"Estimo que el bitcoin costará unos {result['predicted_price']:,.2f} dólares mañana."
