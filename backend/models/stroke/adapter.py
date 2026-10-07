"""Adapter of the stroke risk model."""
import json
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import StrokeInputs


class Stroke(BaseModel):
    """Estimates the probability of stroke and a risk level."""

    name = "stroke"
    title = "Stroke risk"
    description = (
        "Estimates the probability of stroke and a risk level (low, "
        "moderate or high) from age, gender, hypertension, heart disease, "
        "glucose, BMI, work, residence, marital and smoking status. Random "
        "Forest; a screening estimate, not a diagnosis. See "
        "training/stroke/README.md for the test metrics."
    )
    task_type = "classification"
    keywords = [
        "stroke risk",
        "risk of a stroke",
        "risk of stroke",
        "stroke",
    ]
    input_schema = StrokeInputs
    # Training median/mode, taken from artifacts/stroke.meta.json.
    default_input = {
        "gender": "Female",
        "age": 45.0,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence_type": "Urban",
        "avg_glucose_level": 91.9,
        "bmi": 28.1,
        "smoking_status": "never smoked",
    }

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        # One row per request: avoid spinning up a thread pool each time.
        self.pipeline.set_params(model__n_jobs=1)
        meta = json.loads(
            self.artifact_path(".meta.json").read_text(encoding="utf-8")
        )
        # Probability thresholds chosen by cross-validation on the training
        # set: below "moderate" the risk is low.
        self.moderate_threshold = float(meta["risk_thresholds"]["moderate"])
        self.high_threshold = float(meta["risk_thresholds"]["high"])
        self.features = list(meta["features"])
        self.positive = list(self.pipeline.classes_).index(
            meta["positive_class"]
        )

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([inputs])[self.features]
        probability = float(
            self.pipeline.predict_proba(frame)[0][self.positive]
        )
        if probability >= self.high_threshold:
            level = "high"
        elif probability >= self.moderate_threshold:
            level = "moderate"
        else:
            level = "low"
        return {
            "stroke_probability": round(probability, 4),
            "risk_level": level,
            # True when the probability reaches the decision threshold.
            "flagged": bool(probability >= self.moderate_threshold),
            "threshold": self.moderate_threshold,
        }

    def speak(self, result: dict[str, Any]) -> str:
        percent = f"{result['stroke_probability'] * 100:.1f}"
        return (
            f"I estimate a {result['risk_level']} risk of stroke, with a "
            f"probability of about {percent}%; this is a rough estimate, "
            "not a medical diagnosis."
        )
