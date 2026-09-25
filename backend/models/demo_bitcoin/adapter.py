"""Adapter of the Bitcoin demo."""
from typing import Any

from ..base import BaseModel
from .schema import BitcoinInputs


class DemoBitcoin(BaseModel):
    """Toy forecast: next-day price = last price + 1%."""

    name = "bitcoin"
    title = "Precio de Bitcoin (demo, sin entrenar)"
    description = (
        "DEMO, not a trained model: adds 1% to the last price. "
        "Time-series example."
    )
    task_type = "time_series"
    is_demo = True
    keywords = [
        "precio del bitcoin",
        "bitcoin",
        "precio de bitcoin para mañana",
    ]
    input_schema = BitcoinInputs
    default_input = {"last_price": 60000.0}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        last_price = float(inputs["last_price"])
        return {"predicted_price": round(last_price * 1.01, 2)}

    def speak(self, result: dict[str, Any]) -> str:
        price = result["predicted_price"]
        return (
            f"Estimo que el bitcoin costará unos {price:,.2f} "
            "dólares mañana."
        )
