"""Adapter of the cirrhosis stage model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import CirrhosisInputs


class Cirrhosis(BaseModel):
    """Estimates the histologic stage (1-4) of liver disease."""

    name = "cirrhosis"
    title = "Cirrhosis stage"
    description = (
        "Estimates the histologic stage (1 to 4) of liver disease in a "
        "primary biliary cirrhosis patient from clinical signs and lab "
        "values. Random Forest; the stages are hard to separate (see "
        "training/cirrhosis/README.md for the test metrics). An estimate, "
        "not a diagnosis."
    )
    task_type = "classification"
    keywords = [
        "cirrhosis stage",
        "stage of cirrhosis",
        "type of cirrhosis",
        "cirrhosis",
    ]
    input_schema = CirrhosisInputs
    # A real stage-4 patient of the held-out test set (a clear case).
    # The training medians are in artifacts/cirrhosis.meta.json.
    default_input = {
        "age": 70.9,
        "sex": "F",
        "ascites": "Y",
        "hepatomegaly": "Y",
        "spiders": "Y",
        "edema": "Y",
        "bilirubin": 6.6,
        "cholesterol": 222.0,
        "albumin": 2.33,
        "copper": 138.0,
        "alk_phos": 620.0,
        "sgot": 106.0,
        "triglycerides": 91.0,
        "platelets": 195.0,
        "prothrombin": 12.1,
    }

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        # One row per request: avoid spinning up a thread pool each time.
        model = self.pipeline.named_steps["model"]
        if "n_jobs" in model.get_params():
            model.set_params(n_jobs=1)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([inputs])
        probabilities = {
            str(label): round(float(p), 4)
            for label, p in zip(
                self.pipeline.classes_,
                self.pipeline.predict_proba(frame)[0],
            )
        }
        stage = max(probabilities, key=probabilities.get)
        return {"stage": int(stage), "probabilities": probabilities}

    def speak(self, result: dict[str, Any]) -> str:
        stage = result["stage"]
        percent = round(result["probabilities"][str(stage)] * 100)
        return (
            f"I estimate that the patient's cirrhosis is at stage {stage} "
            f"of 4, with a probability of {percent}%; this is a rough "
            "estimate, not a medical diagnosis."
        )
