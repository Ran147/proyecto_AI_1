"""Input schema of the Bitcoin next-day price model."""
from typing import Annotated

from pydantic import BaseModel as Schema
from pydantic import Field

# More days than the model needs are accepted; only the latest are used.
MAX_RECENT_PRICES = 365
# Upper bound of a daily close in USD. Infinity and absurd values
# are rejected here instead of failing later in the model.
MAX_PRICE = 10_000_000

Price = Annotated[
    float, Field(gt=0, le=MAX_PRICE, allow_inf_nan=False)
]


class BitcoinInputs(Schema):
    """Optional overrides of the price history stored with the model.

    Without overrides the model forecasts the day after the last day of its
    data. ``recent_prices`` takes precedence over ``last_price``.
    """

    last_price: Price = Field(
        description=(
            "latest known daily close in USD, greater than 0 and at "
            "most 10,000,000"
        )
    )
    recent_prices: list[Price] = Field(
        default_factory=list,
        max_length=MAX_RECENT_PRICES,
        description=(
            "optional latest daily closes in USD, oldest first, most "
            "recent last; each greater than 0 and at most 10,000,000"
        ),
    )
