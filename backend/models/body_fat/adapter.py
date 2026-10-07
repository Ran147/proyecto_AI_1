"""Adapter of the body fat model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import BodyFatInputs

# Values copied from artifacts/body_fat.meta.json.
# Range of each input in the training set: outside it the linear model
# extrapolates and the estimate is less reliable.
TRAINING_RANGES = {
    "abdomen_cm": (70.4, 126.2),
    "weight_kg": (56.7, 119.18),
}
# Body fat range seen in training; estimates are clipped to it so that an
# unusual input never produces a negative or absurd percentage.
BODY_FAT_RANGE = (0.7, 47.5)
# Mean absolute error on the held-out test set, in percentage points.
TEST_MAE = 3.2275


class BodyFat(BaseModel):
    """Estimates a man's body fat percentage from two measurements."""

    name = "body_fat"
    title = "Body fat percentage"
    description = (
        "Estimates the body fat percentage of an adult man from his "
        "abdomen circumference (cm) and weight (kg). Ridge regression "
        "trained on 246 men measured by underwater weighing; an estimate, "
        "not a medical assessment. See training/body_fat/README.md for "
        "the test metrics."
    )
    task_type = "regression"
    keywords = [
        "body fat",
        "body fat percentage",
        "how much body fat do i have",
        "fat percentage",
    ]
    input_schema = BodyFatInputs
    # Training medians, taken from artifacts/body_fat.meta.json.
    default_input = {
        "abdomen_cm": 90.85,
        "weight_kg": 78.58,
    }

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = float(self.pipeline.predict(pd.DataFrame([inputs]))[0])
        low, high = BODY_FAT_RANGE
        percent = round(min(max(raw, low), high), 1)
        outside = any(
            not minimum <= inputs[field] <= maximum
            for field, (minimum, maximum) in TRAINING_RANGES.items()
        )
        return {
            "body_fat_percent": percent,
            "fat_mass_kg": round(inputs["weight_kg"] * percent / 100, 1),
            "typical_error_points": round(TEST_MAE, 1),
            "outside_training_range": outside,
        }

    def speak(self, result: dict[str, Any]) -> str:
        caveat = (
            "The measurements are outside the range the model learned "
            "from, so this estimate is not very reliable."
            if result["outside_training_range"]
            else "This is an estimate from the measurements, not a "
            "medical assessment."
        )
        return (
            "I estimate a body fat percentage of about "
            f"{result['body_fat_percent']:.1f}%, roughly "
            f"{result['fat_mass_kg']:.1f} kilograms of fat, with a "
            f"typical error of {result['typical_error_points']:.0f} "
            f"points. {caveat}"
        )
