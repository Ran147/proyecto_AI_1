"""Adapter of the customer churn model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import ChurnInputs

# Label the pipeline was trained with for "the customer left".
CHURN_LABEL = 1


class Churn(BaseModel):
    """Estimates whether a telecom customer will leave the company."""

    name = "churn"
    title = "Customer churn"
    description = (
        "Estimates the probability that a telecom customer leaves the "
        "company (churn) from the contract, tenure, services and charges. "
        "Gradient Boosting with a decision threshold tuned for the F1 of "
        "the churn class; see training/churn/README.md for the test "
        "metrics."
    )
    task_type = "classification"
    keywords = [
        "customer churn",
        "will the customer leave",
        "is the customer going to leave",
        "customer leaving",
        "churn",
    ]
    input_schema = ChurnInputs
    # A real customer of the held-out test set who did leave: recent,
    # month-to-month contract, fiber optic (a clear case). The training
    # medians are in artifacts/churn.meta.json.
    default_input = {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 7,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Bank transfer (automatic)",
        "monthly_charges": 98.05,
        "total_charges": 713.0,
    }

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        classifier = self.pipeline.named_steps["model"]
        # One row per request: never spin up a worker pool.
        if "n_jobs" in classifier.get_params():
            classifier.set_params(n_jobs=1)
        self.churn_index = list(self.pipeline.classes_).index(CHURN_LABEL)
        # The pipeline decides with a tuned threshold, not with 0.5.
        self.threshold = float(getattr(classifier, "best_threshold_", 0.5))

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([inputs])
        probability = float(
            self.pipeline.predict_proba(frame)[0][self.churn_index]
        )
        leaves = int(self.pipeline.predict(frame)[0]) == CHURN_LABEL
        return {
            "churn": "Yes" if leaves else "No",
            "churn_probability": round(probability, 4),
            "decision_threshold": round(self.threshold, 4),
        }

    def speak(self, result: dict[str, Any]) -> str:
        percent = round(result["churn_probability"] * 100)
        if result["churn"] == "Yes":
            outcome = "this customer is at risk of leaving"
        else:
            outcome = "this customer is likely to stay"
        return (
            "The estimated probability that the customer will leave the "
            f"company is {percent}%: {outcome}."
        )
