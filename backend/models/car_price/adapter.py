"""Adapter of the car price model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import CarPriceInputs

# The dataset prices are in lakh of Indian rupees.
RUPEES_PER_LAKH = 100_000


class CarPrice(BaseModel):
    """Estimates the resale price of a used vehicle."""

    name = "car_price"
    title = "Used car price"
    description = (
        "Estimates the resale price of a used vehicle (lakh INR) from its "
        "year, showroom price, kilometers, fuel, seller and transmission. "
        "Gradient Boosting on the log price; see "
        "training/car_price/README.md for the test metrics."
    )
    task_type = "regression"
    keywords = [
        "car price",
        "price of the car",
        "price of my car",
        "used car price",
        "how much is my car worth",
    ]
    # Spanish commands; the answers stay in English.
    keywords_es = [
        "precio del carro",
        "precio de mi carro",
        "cuánto vale mi carro",
        "cuánto cuesta mi carro",
        "precio del automóvil",
        "precio de un automóvil usado",
        "precio del vehículo",
    ]
    input_schema = CarPriceInputs
    # Training median/mode, taken from artifacts/car_price.meta.json.
    default_input = {
        "year": 2014,
        "present_price": 5.97,
        "kms_driven": 30753.0,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0,
    }

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        price = float(self.pipeline.predict(pd.DataFrame([inputs]))[0])
        return {
            "selling_price_lakh": round(price, 2),
            "selling_price_inr": round(price * RUPEES_PER_LAKH),
            "currency": "INR",
        }

    def speak(self, result: dict[str, Any]) -> str:
        return (
            "I estimate this vehicle would sell for about "
            f"{result['selling_price_inr']:,} Indian rupees "
            f"({result['selling_price_lakh']:.2f} lakh)."
        )
