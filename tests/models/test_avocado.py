"""Trained ``avocado`` model (next-week price forecast).

Tests that run the model are skipped when artifacts/avocado.joblib is
missing (training/avocado/notebook.ipynb generates it
together with artifacts/avocado.meta.json).
"""
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.avocado import Avocado
from backend.models.avocado.adapter import build_features

ARTIFACT = Avocado().artifact_path()
META = Avocado().artifact_path(".meta.json")

needs_artifact = pytest.mark.skipif(
    not (ARTIFACT.exists() and META.exists()),
    reason=f"{ARTIFACT.name} missing; run training/avocado/notebook.ipynb",
)
client = TestClient(app)


def predict(inputs=None):
    response = client.post(
        "/models/avocado/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def load_meta():
    return json.loads(META.read_text(encoding="utf-8"))


def test_registered_as_real_model():
    info = get_orchestrator().describe("avocado")
    assert info["demo"] is False
    assert info["task_type"] == "time_series"
    assert set(info["input_fields"]) == {"type", "recent_prices", "last_date"}
    assert set(info["input_fields"]) == set(Avocado.default_input)
    assert info["title"] == "Avocado price"
    assert all("avocado" in phrase for phrase in info["keywords"])


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Avocado,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / f"missing{suffix}",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Avocado]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("avocado")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"type": "hass"},
        {"recent_prices": [1.2, 1.3]},  # fewer than the 4 lags
        {"recent_prices": [1.2, 1.3, -1.0, 1.1]},
        {"recent_prices": [1.2, 1.3, 1.25, "expensive"]},
        {"last_date": "2024-05-05"},  # a date without prices
        {"recent_prices": [1.2, 1.3, 1.25, 1.1], "last_date": "may"},
    ):
        response = client.post(
            "/models/avocado/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@pytest.mark.parametrize("last_date", [
    "9999-12-31",  # used to raise OverflowError (HTTP 500)
    "2100-01-02",
    "1999-12-31",
])
def test_out_of_range_last_date_is_rejected(last_date):
    response = client.post(
        "/models/avocado/predict",
        json={"inputs": {
            "recent_prices": [1, 1.1, 1.2, 1.3],
            "last_date": last_date,
        }},
    )
    assert response.status_code == 422


@needs_artifact
@pytest.mark.parametrize("last_date", ["2000-01-01", "2100-01-01"])
def test_last_date_bounds_are_accepted(last_date):
    result = predict({
        "recent_prices": [1, 1.1, 1.2, 1.3],
        "last_date": last_date,
    })["result"]
    assert result["last_data_date"] == last_date


def test_build_features_uses_only_the_last_four_prices():
    features = build_features(
        [9.0, 1.0, 1.1, 1.2, 1.4], "organic", date(2018, 4, 1)
    )
    assert [features[f"lag_{k}"] for k in (1, 2, 3, 4)] == [1.4, 1.2, 1.1, 1.0]
    assert features["roll_mean_4"] == pytest.approx(1.175)
    assert features["diff_1"] == pytest.approx(0.2)
    assert features["type"] == "organic"
    with pytest.raises(ValueError):
        build_features([1.0, 1.1], "organic", date(2018, 4, 1))


@needs_artifact
def test_features_match_the_notebook():
    meta = load_meta()
    forecast_date = date.fromisoformat(meta["forecast_date"])
    for kind, expected in meta["default_features"].items():
        features = build_features(
            meta["history"]["prices"][kind], kind, forecast_date
        )
        assert list(features) == meta["features"]
        assert features.pop("type") == expected.pop("type")
        assert features == pytest.approx(expected)


@needs_artifact
def test_predict_default_input_matches_notebook():
    meta = load_meta()
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    assert result["type"] == "conventional"
    assert result["predicted_price_raw"] == pytest.approx(
        meta["expected_default_prediction"], abs=1e-4
    )
    assert result["predicted_price_usd"] == pytest.approx(
        meta["expected_default_prediction"], abs=0.006
    )
    assert result["last_data_date"] == meta["history"]["last_date"]
    assert result["forecast_date"] == meta["forecast_date"]
    assert result["history_source"] == "dataset"
    assert result["currency"] == "USD"


@needs_artifact
def test_organic_costs_more_than_conventional():
    meta = load_meta()
    organic = predict({"type": "organic"})["result"]
    conventional = predict({"type": "conventional"})["result"]
    assert organic["predicted_price_raw"] == pytest.approx(
        meta["expected_predictions"]["organic"], abs=1e-4
    )
    assert organic["predicted_price_usd"] > conventional["predicted_price_usd"]


@needs_artifact
def test_user_prices_change_the_forecast():
    default = predict()["result"]
    cheap = predict({"recent_prices": [0.8, 0.82, 0.8, 0.79]})["result"]
    dear = predict({
        "recent_prices": [2.0, 2.05, 2.1, 2.1],
        "last_date": "2024-06-02",
    })["result"]
    assert cheap["history_source"] == "user"
    assert cheap["predicted_price_usd"] != default["predicted_price_usd"]
    assert cheap["predicted_price_usd"] < dear["predicted_price_usd"]
    # The forecast stays near the latest price given by the user.
    assert cheap["predicted_price_usd"] == pytest.approx(0.79, abs=0.2)
    assert dear["predicted_price_usd"] == pytest.approx(2.1, abs=0.3)
    # Without a date the forecast carries no calendar date.
    assert cheap["last_data_date"] is None
    assert cheap["forecast_date"] is None
    assert dear["last_data_date"] == "2024-06-02"
    assert dear["forecast_date"] == "2024-06-09"


@needs_artifact
def test_default_sentence_names_the_dataset_dates():
    message = predict()["message"]
    assert "a conventional avocado" in message
    assert "1.05 US dollars per unit" in message
    assert "week of April 1, 2018" in message
    assert "March 25, 2018" in message
    assert "United States price history" in message
    assert "not a current price" in message
    organic = predict({"type": "organic"})["message"]
    assert "an organic avocado" in organic


@needs_artifact
def test_undated_user_prices_are_not_given_the_dataset_date():
    body = predict({"recent_prices": [1, 1.1, 1.2, 1.3]})
    json.dumps(body)  # must be serializable
    message = body["message"]
    assert "in the week after your latest price (1.30 US dollars)" in message
    assert "the prices you gave me" in message
    assert "not a current price" in message
    # No calendar date is attributed to the user's prices.
    assert "March 25" not in message
    assert "April 1" not in message
    assert "last available data point" not in message


@needs_artifact
def test_dated_user_prices_keep_the_dated_sentence():
    message = predict({
        "recent_prices": [2.0, 2.05, 2.1, 2.1],
        "last_date": "2024-06-02",
    })["message"]
    assert "week of June 9, 2024" in message
    assert "your latest price (June 2, 2024, 2.10 US dollars)" in message
    assert "not a current price" in message


@pytest.mark.parametrize("text, expected", [
    ("organic avocado price", {"type": "organic"}),
    ("price of the conventional avocado", {"type": "conventional"}),
    ("avocado price", {}),
    # Whole words only.
    ("inorganic avocado price", {}),
    ("organically grown avocado", {}),
    ("unconventional avocado price", {}),
])
def test_inputs_from_command_matches_whole_words(text, expected):
    assert Avocado().inputs_from_command(text) == expected


@needs_artifact
def test_voice_command_picks_the_type():
    organic = client.post(
        "/command", json={"text": "JarvisTEC, organic avocado price"}
    ).json()
    assert organic["model"] == "avocado"
    assert organic["result"]["type"] == "organic"
    assert "an organic avocado" in organic["message"]
    inorganic = client.post(
        "/command", json={"text": "JarvisTEC, inorganic avocado price"}
    ).json()
    if inorganic["matched"] and inorganic["model"] == "avocado":
        assert inorganic["result"]["type"] == "conventional"


VOICE_COMMANDS = [
    "JarvisTEC, avocado price",
    "Jarvis tec price of avocado",
    "JarvisTEC price of avocados",
    "JarvisTEC, avocado forecast",
    "JarvisTEC, how much will avocado cost?",
    "JarvisTEC avocado",
]


def test_every_keyword_has_a_voice_command():
    assert Avocado.keywords == [
        "avocado price",
        "price of avocado",
        "price of avocados",
        "avocado forecast",
        "how much will avocado cost",
        "avocado",
    ]
    for keyword in Avocado.keywords:
        assert any(keyword in text.lower() for text in VOICE_COMMANDS)


@needs_artifact
@pytest.mark.parametrize("text", VOICE_COMMANDS)
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "avocado"
    assert body["result"]["predicted_price_usd"] > 0
    message = body["message"]
    assert "US dollars" in message
    assert "conventional" in message
    # The sentence states the dates, so it is not taken for a live price.
    assert "2018" in message
    assert "not a current price" in message


@pytest.mark.parametrize("text", VOICE_COMMANDS)
def test_voice_phrases_route_to_avocado(text):
    assert get_orchestrator().match(text)[0] == "avocado"
