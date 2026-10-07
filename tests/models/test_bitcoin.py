"""Trained ``bitcoin`` model (next-day closing price).

Tests that run the model are skipped when artifacts/bitcoin.joblib is
missing (training/bitcoin/notebook.ipynb generates it).
"""
import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.bitcoin import Bitcoin
from backend.models.bitcoin.adapter import (
    HISTORY_DAYS,
    build_features,
    splice_history,
)

ARTIFACT = Bitcoin().artifact_path()
META = Bitcoin().artifact_path(".meta.json")

needs_artifact = pytest.mark.skipif(
    not (ARTIFACT.exists() and META.exists()),
    reason=f"{ARTIFACT.name} missing; run training/bitcoin/notebook.ipynb",
)
client = TestClient(app)


def predict(inputs=None):
    response = client.post(
        "/models/bitcoin/predict", json={"inputs": inputs or {}}
    )
    assert response.status_code == 200
    return response.json()


def meta():
    return json.loads(META.read_text(encoding="utf-8"))


def test_registered_as_real_model():
    info = get_orchestrator().describe("bitcoin")
    assert info["demo"] is False
    assert info["task_type"] == "time_series"
    assert set(info["input_fields"]) == {"last_price", "recent_prices"}
    assert info["title"] == "Bitcoin price for tomorrow"
    assert "untrained" not in info["title"].lower()


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Bitcoin,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / f"missing{suffix}",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Bitcoin]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("bitcoin")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"last_price": -5},
        {"last_price": 0},
        {"last_price": "expensive"},
        {"recent_prices": [2800, -1]},
        {"recent_prices": "2800"},
    ):
        response = client.post(
            "/models/bitcoin/predict", json={"inputs": inputs}
        )
        assert response.status_code == 422


@pytest.mark.parametrize("inputs", [
    '{"last_price": 1e999}',  # parsed as infinity
    '{"last_price": 1e300}',
    '{"last_price": 10000000.01}',
    '{"last_price": NaN}',
    '{"recent_prices": [2800, 1e999]}',
    '{"recent_prices": [2800, 1e300]}',
    '{"last_price": 2800, "recent_prices": [1e8, 2800]}',
])
def test_unbounded_prices_are_rejected(inputs):
    # Raw body: the JSON literals 1e999 and NaN reach the server as
    # written. They used to end in an HTTP 500.
    response = client.post(
        "/models/bitcoin/predict",
        content='{"inputs": ' + inputs + "}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@needs_artifact
def test_largest_accepted_price():
    result = predict({"last_price": 10_000_000})["result"]
    assert result["last_price"] == 10_000_000
    assert result["predicted_price"] == pytest.approx(1e7, rel=0.1)


def test_features_use_only_past_closes():
    closes = [100.0 + day for day in range(HISTORY_DAYS)]
    features = build_features(closes, date(2017, 8, 1))
    assert features["close_lag_1"] == closes[-1]
    assert features["close_lag_7"] == closes[-7]
    assert features["return_lag_1"] > 0
    assert features["day_of_week"] == 1  # 2017-08-01 was a Tuesday
    # Older closes than the window do not change anything.
    assert build_features([1.0] + closes, date(2017, 8, 1)) == features
    with pytest.raises(ValueError):
        build_features(closes[1:], date(2017, 8, 1))


def test_splice_history_keeps_the_daily_pattern():
    stored = [10.0, 20.0, 40.0, 80.0]
    assert splice_history(stored, [8.0]) == [1.0, 2.0, 4.0, 8.0]
    assert splice_history(stored, [4.0, 6.0]) == [1.0, 2.0, 4.0, 6.0]
    assert splice_history(stored, [9.0, 1.0, 2.0, 3.0, 4.0]) == [
        1.0, 2.0, 3.0, 4.0,
    ]


def test_default_input_matches_the_exported_history():
    if not META.exists():
        pytest.skip("bitcoin.meta.json missing")
    exported = meta()
    assert Bitcoin.default_input == {
        **exported["default_input"],
        "recent_prices": [],
    }
    assert Bitcoin.default_input["last_price"] == (
        exported["history"]["closes"][-1]
    )
    assert len(exported["history"]["closes"]) >= HISTORY_DAYS


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    exported = meta()
    # Same forecast the notebook computed with its own feature function.
    assert result["predicted_price"] == pytest.approx(
        exported["expected_default_prediction"], abs=0.01
    )
    assert result["source"] == "dataset"
    assert result["last_data_date"] == exported["history"]["last_date"]
    last_day = date.fromisoformat(result["last_data_date"])
    assert result["forecast_date"] == str(last_day + timedelta(days=1))
    assert result["last_price"] == Bitcoin.default_input["last_price"]
    # The spoken answer names the dates: it is not a live price.
    message = body["message"]
    assert "My data ends on July 31, 2017" in message
    assert "on August 1, 2017" in message
    assert f"{result['predicted_price']:,.2f} US dollars" in message
    assert "not the current price" in message


@needs_artifact
def test_last_price_override_scales_the_forecast():
    default = predict()["result"]
    ratio = default["predicted_price"] / default["last_price"]
    for last_price in (100, 60000):
        body = predict({"last_price": last_price})
        result = body["result"]
        assert result["source"] == "user"
        assert result["forecast_date"] is None
        assert result["last_price"] == last_price
        # Same expected daily change, applied to the user's price.
        assert result["predicted_price"] == pytest.approx(
            last_price * ratio, rel=1e-3
        )
        assert "the next day" in body["message"]
        assert f"{last_price:,.2f} US dollars" in body["message"]
        assert "not from a live price" in body["message"]
        assert "2017" not in body["message"]


@needs_artifact
def test_recent_prices_override():
    recent = [2700.0, 2750.0, 2600.0, 2500.0]
    body = predict({"recent_prices": recent})
    result = body["result"]
    assert result["source"] == "user"
    assert result["last_price"] == recent[-1]
    # A one-day forecast: within 10 % of the last close.
    assert result["predicted_price"] == pytest.approx(recent[-1], rel=0.1)
    # A full window of user prices replaces the stored history entirely.
    full = [3000.0 + 10 * day for day in range(HISTORY_DAYS + 5)]
    result = predict({"recent_prices": full})["result"]
    assert result["last_price"] == full[-1]
    assert result["predicted_price"] == pytest.approx(full[-1], rel=0.1)


VOICE_COMMANDS = [
    "JarvisTEC, bitcoin price",
    "Jarvis tec price of bitcoin",
    "Jarvis tec bitcoin price tomorrow",
    "JarvisTEC, bit coin",
    "JarvisTEC bitcoin",
]


def test_every_keyword_has_a_voice_command():
    assert Bitcoin.keywords == [
        "bitcoin price",
        "price of bitcoin",
        "bitcoin price tomorrow",
        "bit coin",
        "bitcoin",
    ]
    for keyword in Bitcoin.keywords:
        assert any(keyword in text.lower() for text in VOICE_COMMANDS)


@needs_artifact
@pytest.mark.parametrize("text", VOICE_COMMANDS)
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "bitcoin"
    assert body["result"]["predicted_price"] > 0
    assert "US dollars" in body["message"]
    assert "2017" in body["message"]


@pytest.mark.parametrize("phrase", Bitcoin.keywords)
def test_every_keyword_routes_to_bitcoin(phrase):
    orchestrator = get_orchestrator()
    assert orchestrator.match(f"JarvisTEC {phrase}")[0] == "bitcoin"
