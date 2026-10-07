"""Adapter of the avocado price forecast (one week ahead)."""
import json
import math
import re
from datetime import date, timedelta
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import N_LAGS, AvocadoInputs

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
# Whole words only: "inorganic" must not select the organic type.
TYPE_PATTERNS = {
    "organic": re.compile(r"\borganic\b"),
    "conventional": re.compile(r"\bconventional\b"),
}


def build_features(
    recent_prices: list[float], avocado_type: str, forecast_date: date
) -> dict[str, Any]:
    """Inputs to forecast the price of the week of ``forecast_date``.

    Same function as in training/avocado/notebook.ipynb (section 4); keep
    both identical. ``recent_prices`` are the weekly prices observed BEFORE
    that week, most recent last; only the last N_LAGS are used.
    """
    prices = [float(price) for price in recent_prices[-N_LAGS:]]
    if len(prices) < N_LAGS:
        raise ValueError(f"at least {N_LAGS} weekly prices are needed")
    angle = 2 * math.pi * forecast_date.isocalendar()[1] / 52
    features = {f"lag_{k}": prices[-k] for k in range(1, N_LAGS + 1)}
    features["roll_mean_4"] = sum(prices) / N_LAGS
    features["diff_1"] = prices[-1] - prices[-2]
    features["week_sin"] = math.sin(angle)
    features["week_cos"] = math.cos(angle)
    features["type"] = avocado_type
    return features


def english_date(iso_date: str) -> str:
    """'2018-04-01' -> 'April 1, 2018'."""
    day = date.fromisoformat(iso_date)
    return f"{MONTHS[day.month - 1]} {day.day}, {day.year}"


class Avocado(BaseModel):
    """Forecasts next week's average avocado price in the United States."""

    name = "avocado"
    title = "Avocado price"
    description = (
        "Forecasts the average retail price of a Hass avocado (USD per "
        "unit, United States) for the week after the last observed one, "
        "from the previous 4 weekly prices, the week of the year and the "
        "type. With no inputs it uses the national history stored at "
        "training time (Hass Avocado Board data ending in March 2018); see "
        "training/avocado/README.md for the test metrics."
    )
    task_type = "time_series"
    keywords = [
        "avocado price",
        "price of avocado",
        "price of avocados",
        "avocado forecast",
        "how much will avocado cost",
        "avocado",
    ]
    input_schema = AvocadoInputs
    # No data needed: the history comes from artifacts/avocado.meta.json.
    default_input = {
        "type": "conventional",
        "recent_prices": None,
        "last_date": None,
    }

    def inputs_from_command(self, text: str) -> dict[str, Any]:
        """Pick the type when the command names it ("organic avocado").

        Only whole words count: "inorganic" does not select organic.
        """
        for avocado_type, pattern in TYPE_PATTERNS.items():
            if pattern.search(text):
                return {"type": avocado_type}
        return {}

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        if hasattr(self.pipeline.named_steps["model"], "n_jobs"):
            self.pipeline.set_params(model__n_jobs=1)
        meta = json.loads(
            self.artifact_path(".meta.json").read_text(encoding="utf-8")
        )
        self.features = meta["features"]
        self.history = meta["history"]

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        avocado_type = inputs["type"]
        data_last_date = date.fromisoformat(self.history["last_date"])
        prices = inputs.get("recent_prices")
        own_prices = prices is not None
        # Prices given without a date: the week of the year (seasonal
        # features) is assumed to be the one after the stored data,
        # but no calendar date is reported for the forecast.
        dated = True
        if own_prices:
            dated = inputs.get("last_date") is not None
            last_date = inputs.get("last_date") or data_last_date
        else:
            prices = self.history["prices"][avocado_type]
            last_date = data_last_date
        forecast_date = last_date + timedelta(days=7)
        features = build_features(prices, avocado_type, forecast_date)
        frame = pd.DataFrame([features])[self.features]
        price = float(self.pipeline.predict(frame)[0])
        return {
            "predicted_price_usd": round(price, 2),
            "predicted_price_raw": round(price, 4),
            "type": avocado_type,
            # None: the forecast is for the week after the user's
            # latest price, whose date is unknown.
            "forecast_date": forecast_date.isoformat() if dated else None,
            "last_data_date": last_date.isoformat() if dated else None,
            "last_price_usd": round(float(prices[-1]), 2),
            "history_source": "user" if own_prices else "dataset",
            "currency": "USD",
        }

    def speak(self, result: dict[str, Any]) -> str:
        article = "an" if result["type"] == "organic" else "a"
        estimate = (
            f"I estimate that {article} {result['type']} avocado will "
            f"cost about {result['predicted_price_usd']:,.2f} US dollars "
            "per unit in the week "
        )
        last_price = f"{result['last_price_usd']:,.2f} US dollars"
        if result["history_source"] != "user":
            return (
                f"{estimate}of {english_date(result['forecast_date'])}, "
                "the week after the last available data point "
                f"({english_date(result['last_data_date'])}, "
                f"{last_price}), based on the United States price "
                "history. This is a forecast on historical data that "
                "ends in March 2018, not a current price."
            )
        closing = (
            "This is a forecast from a model trained on historical data "
            "that ends in March 2018, not a current price."
        )
        if result["forecast_date"] is None:
            # The user gave prices without a date: name no date.
            return (
                f"{estimate}after your latest price ({last_price}), "
                f"based on the prices you gave me. {closing}"
            )
        return (
            f"{estimate}of {english_date(result['forecast_date'])}, the "
            "week after your latest price "
            f"({english_date(result['last_data_date'])}, {last_price}), "
            f"based on the prices you gave me. {closing}"
        )
