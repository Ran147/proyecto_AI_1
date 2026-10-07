"""Adapter of the Bitcoin next-day price model."""
import json
import math
from datetime import date
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..base import BaseModel
from .schema import BitcoinInputs

# ---- Lag features --------------------------------------------------------
# Same constants and function as section 4 of training/bitcoin/notebook.ipynb
# (the backend never imports from training/). Change both together.
N_LAGS = 7
WINDOWS = (7, 30)
HISTORY_DAYS = 31  # closes needed: 30 returns for the longest window

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


def build_features(closes, forecast_date) -> dict[str, float]:
    """Features to forecast the close of ``forecast_date``.

    ``closes`` are the daily closes known the day before, oldest first.
    Nothing from the forecast day itself is used.
    """
    closes = np.asarray(closes, dtype=float)[-HISTORY_DAYS:]
    if len(closes) < HISTORY_DAYS or not (closes > 0).all():
        raise ValueError(f"{HISTORY_DAYS} positive closes are required")
    returns = np.diff(np.log(closes))
    features = {}
    for lag in range(1, N_LAGS + 1):
        features[f"close_lag_{lag}"] = float(closes[-lag])
    for lag in range(1, N_LAGS + 1):
        features[f"return_lag_{lag}"] = float(returns[-lag])
    for window in WINDOWS:
        features[f"return_mean_{window}"] = float(returns[-window:].mean())
        features[f"return_std_{window}"] = float(returns[-window:].std())
        features[f"gap_sma_{window}"] = float(
            np.log(closes[-1] / closes[-window:].mean())
        )
    features["day_of_week"] = int(forecast_date.weekday())
    return features


def splice_history(stored: list[float], recent: list[float]) -> list[float]:
    """Replace the end of the stored history with the user's prices.

    The older stored closes are rescaled so the series stays continuous:
    they keep their day-to-day pattern at the user's price level. A price
    far from the stored one (e.g. today's) is then not read as a one-day
    jump, which the model never saw.
    """
    recent = recent[-len(stored):]
    kept = len(stored) - len(recent)
    scale = recent[0] / stored[kept]
    return [close * scale for close in stored[:kept]] + list(recent)


def english_date(iso_date: str) -> str:
    """Return e.g. 'August 1, 2017' for '2017-08-01'."""
    day = date.fromisoformat(iso_date)
    return f"{MONTHS[day.month - 1]} {day.day}, {day.year}"


class Bitcoin(BaseModel):
    """Forecasts tomorrow's Bitcoin closing price in USD."""

    name = "bitcoin"
    title = "Bitcoin price for tomorrow"
    description = (
        "Forecasts the next-day Bitcoin closing price (USD) from the last "
        "31 daily closes. The model predicts the daily log return from "
        "lagged returns and rolling statistics. Historical data that ends "
        "on 2017-07-31: not a live price. It does not beat the naive "
        "'tomorrow = today' forecast; see training/bitcoin/README.md."
    )
    task_type = "time_series"
    keywords = [
        "bitcoin price",
        "price of bitcoin",
        "bitcoin price tomorrow",
        "bit coin",
        "bitcoin",
    ]
    # Spanish commands; the answers stay in English.
    keywords_es = [
        "precio del bitcoin",
        "precio de bitcoin",
        "precio del bitcoin para mañana",
    ]
    input_schema = BitcoinInputs
    # Last close of the data (2017-07-31), from artifacts/bitcoin.meta.json.
    # With this value the forecast is the one for the day after the data.
    # An empty recent_prices keeps the stored history.
    default_input = {"last_price": 2875.34, "recent_prices": []}

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        estimator = self.pipeline.named_steps["model"]
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)
        meta = json.loads(
            self.artifact_path(".meta.json").read_text(encoding="utf-8")
        )
        # "log_return" or "price": what the pipeline predicts.
        self.target_kind = meta["target_kind"]
        history = meta["history"]
        self.closes = [float(close) for close in history["closes"]]
        self.last_date = history["last_date"]
        self.forecast_date = history["forecast_date"]

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        recent = [float(p) for p in inputs.get("recent_prices") or []]
        if not recent:
            recent = [float(inputs["last_price"])]
        # The stored last close means "no override": forecast the day after
        # the data ends.
        from_user = len(recent) > 1 or not math.isclose(
            recent[0], self.closes[-1], abs_tol=0.005
        )
        closes = splice_history(self.closes, recent) if from_user else (
            self.closes
        )
        # The day of the week is the one after the last data day; a user
        # price carries no date.
        features = build_features(
            closes, date.fromisoformat(self.forecast_date)
        )
        output = float(self.pipeline.predict(pd.DataFrame([features]))[0])
        last_price = closes[-1]
        if self.target_kind == "log_return":
            price = last_price * math.exp(output)
        else:
            price = output
        return {
            "predicted_price": round(price, 2),
            "currency": "USD",
            "last_price": round(last_price, 2),
            "expected_change_pct": round((price / last_price - 1) * 100, 2),
            "last_data_date": self.last_date,
            # None: the forecast is for the day after the user's price.
            "forecast_date": None if from_user else self.forecast_date,
            "source": "user" if from_user else "dataset",
        }

    def speak(self, result: dict[str, Any]) -> str:
        price = f"{result['predicted_price']:,.2f}"
        if result["source"] == "user":
            return (
                "Starting from a last price of "
                f"{result['last_price']:,.2f} US dollars, I estimate "
                f"that bitcoin would close at about {price} US dollars "
                "the next day. The estimate comes from a model trained "
                "on historical data, not from a live price."
            )
        return (
            "My data ends on "
            f"{english_date(result['last_data_date'])}: I estimate that "
            "bitcoin would close at about "
            f"{price} US dollars on "
            f"{english_date(result['forecast_date'])}. This is a "
            "historical figure, not the current price."
        )
