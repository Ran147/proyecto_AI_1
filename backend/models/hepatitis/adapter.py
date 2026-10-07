"""Adapter of the hepatitis C category model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import HepatitisInputs

# Class labels of the model -> plain words used in the spoken reply.
SPOKEN_CLASSES = {
    "blood_donor": "a blood donor without hepatitis C",
    "hepatitis": "a patient with hepatitis C",
    "fibrosis": "a patient with hepatitis C and fibrosis",
    "cirrhosis": "a patient with hepatitis C and cirrhosis",
}


class Hepatitis(BaseModel):
    """Estimates the hepatitis C category from routine lab values."""

    name = "hepatitis"
    title = "Hepatitis C category"
    description = (
        "Estimates the category of a patient (blood donor, hepatitis, "
        "fibrosis or cirrhosis) from age, sex and 10 routine blood test "
        "values. Random Forest with balanced class weights; an estimate, "
        "not a diagnosis. See training/hepatitis/README.md for the test "
        "metrics and their limits."
    )
    task_type = "classification"
    keywords = [
        "hepatitis type",
        "type of hepatitis",
        "classify hepatitis",
        "hepatitis",
    ]
    # Spanish commands; the answers stay in English.
    keywords_es = [
        "tipo de hepatitis",
        "clasificar hepatitis",
        "hepatitis del paciente",
    ]
    input_schema = HepatitisInputs
    # Training median/mode, taken from artifacts/hepatitis.meta.json. It is
    # a healthy profile: 88 % of the training patients are blood donors.
    default_input = {
        "age": 46,
        "sex": "m",
        "alb": 41.85,
        "alp": 65.4,
        "alt": 22.5,
        "ast": 26.05,
        "bil": 7.45,
        "che": 8.225,
        "chol": 5.29,
        "crea": 76.45,
        "ggt": 23.25,
        "prot": 72.0,
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
        category = max(probabilities, key=probabilities.get)
        return {"category": category, "probabilities": probabilities}

    def speak(self, result: dict[str, Any]) -> str:
        category = result["category"]
        percent = round(result["probabilities"][category] * 100)
        return (
            "I estimate that these lab values correspond to "
            f"{SPOKEN_CLASSES[category]}, with a probability of "
            f"{percent}%; this is an estimate, not a medical diagnosis."
        )
