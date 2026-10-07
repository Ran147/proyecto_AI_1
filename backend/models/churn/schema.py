"""Input schema of the customer churn model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field

YesNo = Literal["Yes", "No"]
# Add-ons of the internet plan; the third value marks "no internet plan".
InternetAddOn = Literal["Yes", "No", "No internet service"]


class ChurnInputs(Schema):
    """Same snake_case columns the pipeline was trained on."""

    gender: Literal["Female", "Male"] = Field(
        description="customer gender: Female or Male"
    )
    senior_citizen: Literal[0, 1] = Field(
        description="1 if the customer is a senior citizen, else 0"
    )
    partner: YesNo = Field(description="has a partner: Yes or No")
    dependents: YesNo = Field(description="has dependents: Yes or No")
    tenure: int = Field(
        ge=0, le=120, description="time as a customer, months"
    )
    phone_service: YesNo = Field(description="has phone service: Yes or No")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        description="several phone lines: Yes, No or No phone service"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        description="internet plan: DSL, Fiber optic or No"
    )
    online_security: InternetAddOn = Field(
        description="online security add-on: Yes, No or No internet service"
    )
    online_backup: InternetAddOn = Field(
        description="online backup add-on: Yes, No or No internet service"
    )
    device_protection: InternetAddOn = Field(
        description="device protection: Yes, No or No internet service"
    )
    tech_support: InternetAddOn = Field(
        description="tech support add-on: Yes, No or No internet service"
    )
    streaming_tv: InternetAddOn = Field(
        description="TV streaming: Yes, No or No internet service"
    )
    streaming_movies: InternetAddOn = Field(
        description="movie streaming: Yes, No or No internet service"
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="contract term: Month-to-month, One year or Two year"
    )
    paperless_billing: YesNo = Field(
        description="paperless billing: Yes or No"
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        description=(
            "payment method: Electronic check, Mailed check, Bank transfer "
            "(automatic) or Credit card (automatic)"
        )
    )
    monthly_charges: float = Field(
        ge=0, le=500, description="amount billed per month, USD"
    )
    total_charges: float = Field(
        ge=0, le=50000, description="total billed since joining, USD"
    )
