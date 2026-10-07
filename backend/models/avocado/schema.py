"""Input schema of the avocado price forecast."""
from datetime import date
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field, model_validator

# Weekly prices the lag features need (N_LAGS in the training notebook).
N_LAGS = 4
# Accepted range of last_date: a week is added to it, so dates near
# date.max would overflow.
MIN_LAST_DATE = date(2000, 1, 1)
MAX_LAST_DATE = date(2100, 1, 1)


class AvocadoInputs(Schema):
    """Optional overrides; with none, the stored national history is used."""

    type: Literal["conventional", "organic"] = Field(
        default="conventional",
        description="avocado type: conventional or organic",
    )
    recent_prices: list[float] | None = Field(
        default=None,
        min_length=N_LAGS,
        max_length=520,
        description=(
            f"optional: your own latest weekly average prices, at least "
            f"{N_LAGS}, most recent last, USD per avocado; empty uses the "
            "stored US history"
        ),
    )
    last_date: date | None = Field(
        default=None,
        ge=MIN_LAST_DATE,
        le=MAX_LAST_DATE,
        description=(
            "optional: date (YYYY-MM-DD, 2000-01-01 to 2100-01-01) of "
            "the most recent price in recent_prices; empty gives a "
            "forecast for the week after that price with no calendar "
            "date (the season is taken from the end of the stored "
            "history)"
        ),
    )

    @model_validator(mode="after")
    def check_overrides(self) -> "AvocadoInputs":
        """Reject non-positive prices and a date without prices."""
        prices = self.recent_prices or []
        if any(not 0 < price <= 10 for price in prices):
            raise ValueError(
                "recent_prices must be between 0 and 10 USD per avocado"
            )
        if self.last_date is not None and self.recent_prices is None:
            raise ValueError("last_date needs recent_prices")
        return self
