"""Trained ``churn`` model.

Tests that run the model are skipped when artifacts/churn.joblib is missing
(training/churn/notebook.ipynb generates it).
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.churn import Churn

ARTIFACT = Churn().artifact_path()

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/churn/notebook.ipynb",
)
client = TestClient(app)

# Two opposite profiles (section 3 of the notebook): a new customer on a
# monthly fiber plan without add-ons, and a long-standing one on a two-year
# contract.
AT_RISK = {
    "tenure": 1,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "payment_method": "Electronic check",
    "monthly_charges": 85.0,
    "total_charges": 85.0,
}
LOYAL = {
    "tenure": 70,
    "contract": "Two year",
    "internet_service": "DSL",
    "online_security": "Yes",
    "tech_support": "Yes",
    "payment_method": "Credit card (automatic)",
    "monthly_charges": 60.0,
    "total_charges": 4200.0,
}


def predict(inputs=None):
    response = client.post(
        "/models/churn/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def test_registered_as_real_model():
    info = get_orchestrator().describe("churn")
    assert info["demo"] is False
    assert info["task_type"] == "classification"
    assert set(info["input_fields"]) == set(Churn.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Churn,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Churn]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("churn")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"contract": "Three year"},
        {"tenure": -1},
        {"monthly_charges": -5},
        {"senior_citizen": 2},
        {"internet_service": "Satellite"},
    ):
        response = client.post(
            "/models/churn/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    assert result["churn"] in ("Yes", "No")
    assert 0 <= result["churn_probability"] <= 1
    assert 0 < result["decision_threshold"] < 1
    # The label follows the probability and the model's threshold.
    flagged = result["churn_probability"] >= result["decision_threshold"]
    assert (result["churn"] == "Yes") == flagged
    assert "%" in body["message"]


@needs_artifact
def test_predict_partial_input_uses_defaults():
    inputs = predict({"tenure": 60, "contract": "Two year"})["inputs"]
    assert inputs["tenure"] == 60
    assert inputs["contract"] == "Two year"
    assert inputs["monthly_charges"] == Churn.default_input["monthly_charges"]


@needs_artifact
def test_new_monthly_customer_is_riskier_than_loyal_one():
    at_risk = predict(AT_RISK)["result"]
    loyal = predict(LOYAL)["result"]
    assert at_risk["churn_probability"] > loyal["churn_probability"]
    assert at_risk["churn"] == "Yes"
    assert loyal["churn"] == "No"


@needs_artifact
def test_longer_contract_and_tenure_lower_the_risk():
    def probability(**inputs):
        return predict(inputs)["result"]["churn_probability"]

    assert probability(contract="Month-to-month") > probability(
        contract="Two year"
    )
    assert probability(tenure=2, total_charges=140.0) > probability(
        tenure=60, total_charges=4200.0
    )


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, customer churn",
    "Jarvis tec will the customer leave",
    "JarvisTEC, is the customer going to leave",
    "JarvisTEC customer leaving",
    "Jarvis tec, churn",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "churn"
    assert 0 <= body["result"]["churn_probability"] <= 1
    assert "customer" in body["message"]
    assert "probability" in body["message"]


@pytest.mark.parametrize("phrase", Churn.keywords)
def test_every_keyword_routes_to_churn(phrase):
    assert get_orchestrator().match(f"JarvisTEC {phrase}")[0] == "churn"


def test_keywords_are_the_english_commands():
    assert Churn.title == "Customer churn"
    assert Churn.keywords == [
        "customer churn",
        "will the customer leave",
        "is the customer going to leave",
        "customer leaving",
        "churn",
    ]


@pytest.mark.parametrize("label, expected", [
    ("Yes", "at risk of leaving"),
    ("No", "likely to stay"),
])
def test_spoken_sentence_is_english(label, expected):
    message = Churn().speak({
        "churn": label,
        "churn_probability": 0.6149,
        "decision_threshold": 0.299,
    })
    assert message.startswith("The estimated probability that the")
    assert "is 61%" in message
    assert expected in message
