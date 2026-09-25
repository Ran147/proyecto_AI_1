"""Adapter of the wine quality model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import WineInputs

# Class labels of the model -> Spanish words used in the spoken reply.
SPOKEN_CLASSES = {"low": "baja", "medium": "media", "high": "alta"}


class Wine(BaseModel):
    """Classifies wine quality as low, medium or high."""

    name = "wine"
    title = "Calidad del vino"
    description = (
        "Classifies wine quality (low, medium or high) from its "
        "physicochemical properties. Random Forest; see "
        "training/wine/README.md for the test metrics."
    )
    task_type = "classification"
    keywords = [
        "calidad del vino",
        "clasificar vino",
        "que tan bueno es este vino",
        "vino",
    ]
    input_schema = WineInputs
    # Training median/mode, taken from artifacts/wine.meta.json.
    default_input = {
        "type": "white",
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.3,
        "citric_acid": 0.31,
        "residual_sugar": 2.8,
        "chlorides": 0.047,
        "free_sulfur_dioxide": 28.0,
        "total_sulfur_dioxide": 117.0,
        "density": 0.995,
        "ph": 3.22,
        "sulphates": 0.51,
        "alcohol": 10.4,
    }

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        # One row per request: avoid spinning up a thread pool each time.
        self.pipeline.set_params(model__n_jobs=1)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([inputs])
        probabilities = {
            str(label): round(float(p), 4)
            for label, p in zip(
                self.pipeline.classes_,
                self.pipeline.predict_proba(frame)[0],
            )
        }
        quality = max(probabilities, key=probabilities.get)
        return {"quality": quality, "probabilities": probabilities}

    def speak(self, result: dict[str, Any]) -> str:
        quality = result["quality"]
        percent = round(result["probabilities"][quality] * 100)
        return (
            f"Estimo que ese vino es de calidad {SPOKEN_CLASSES[quality]}, "
            f"con una probabilidad del {percent} %."
        )
