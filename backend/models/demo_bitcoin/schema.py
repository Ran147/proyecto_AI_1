"""Input schema of the Bitcoin demo."""
from pydantic import BaseModel as Schema
from pydantic import Field


class BitcoinInputs(Schema):
    """Inputs of the Bitcoin demo."""

    last_price: float = Field(description="last price in USD")
