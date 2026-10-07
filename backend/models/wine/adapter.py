"""Adapter of the wine quality model."""
import re
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import WineInputs

# Whole words only: a substring check would match "preferred".
TYPE_PATTERNS = {
    "red": re.compile(r"\b(?:red|tinto)\b"),
    "white": re.compile(r"\b(?:white|blanco)\b"),
}


class Wine(BaseModel):
    """Classifies wine quality as low, medium or high."""

    name = "wine"
    title = "Wine quality"
    description = (
        "Classifies wine quality (low, medium or high) from its "
        "physicochemical properties. Random Forest; see "
        "training/wine/README.md for the test metrics."
    )
    task_type = "classification"
    keywords = [
        "wine quality",
        "quality of the wine",
        "classify wine",
        "how good is this wine",
        "wine",
    ]
    # Spanish commands; the answers stay in English.
    keywords_es = [
        "calidad del vino",
        "clasificar vino",
        "qué tan bueno es este vino",
    ]
    input_schema = WineInputs
    # A real white wine of the held-out test set, rated high quality by
    # the tasters (a clear case). The training medians are in
    # artifacts/wine.meta.json.
    default_input = {
        "type": "white",
        "fixed_acidity": 7.3,
        "volatile_acidity": 0.3,
        "citric_acid": 0.34,
        "residual_sugar": 2.7,
        "chlorides": 0.044,
        "free_sulfur_dioxide": 34.0,
        "total_sulfur_dioxide": 108.0,
        "density": 0.99105,
        "ph": 3.36,
        "sulphates": 0.53,
        "alcohol": 12.8,
    }

    def inputs_from_command(self, text: str) -> dict[str, Any]:
        """Pick the wine type when the command names it ("red wine")."""
        for wine_type, pattern in TYPE_PATTERNS.items():
            if pattern.search(text):
                return {"type": wine_type}
        return {}

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
            f"I estimate this wine is of {quality} quality, "
            f"with a probability of {percent}%."
        )
