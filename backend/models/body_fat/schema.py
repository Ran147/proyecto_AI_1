"""Input schema of the body fat model."""
from pydantic import BaseModel as Schema
from pydantic import Field


class BodyFatInputs(Schema):
    """Same snake_case columns the pipeline was trained on (metric units).

    The model was trained on adult men with an abdomen of 70-126 cm and a
    weight of 57-119 kg; the bounds below only reject impossible values.
    """

    abdomen_cm: float = Field(
        ge=50,
        le=200,
        description=(
            "abdomen circumference at the level of the navel, in "
            "centimeters (cm)"
        ),
    )
    weight_kg: float = Field(
        ge=30, le=250, description="body weight, in kilograms (kg)"
    )
