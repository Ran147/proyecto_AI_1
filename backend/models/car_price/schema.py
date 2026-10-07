"""Input schema of the car price model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field


class CarPriceInputs(Schema):
    """Same snake_case columns the pipeline was trained on."""

    year: int = Field(ge=1980, le=2030, description="year of manufacture")
    present_price: float = Field(
        gt=0,
        description="showroom price of the same vehicle when new, lakh INR",
    )
    kms_driven: float = Field(ge=0, description="distance driven, km")
    fuel_type: Literal["Petrol", "Diesel", "CNG"] = Field(
        description="fuel: Petrol, Diesel or CNG"
    )
    seller_type: Literal["Dealer", "Individual"] = Field(
        description="seller: Dealer or Individual"
    )
    transmission: Literal["Manual", "Automatic"] = Field(
        description="gearbox: Manual or Automatic"
    )
    owner: int = Field(ge=0, le=10, description="number of previous owners")
