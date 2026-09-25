"""Input schema of the wine quality model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field


class WineInputs(Schema):
    """Same snake_case columns the pipeline was trained on."""

    type: Literal["red", "white"] = Field(
        description="wine type: red or white"
    )
    fixed_acidity: float = Field(
        ge=0, description="fixed acidity, g/dm3 of tartaric acid"
    )
    volatile_acidity: float = Field(
        ge=0, description="volatile acidity, g/dm3 of acetic acid"
    )
    citric_acid: float = Field(ge=0, description="citric acid, g/dm3")
    residual_sugar: float = Field(ge=0, description="residual sugar, g/dm3")
    chlorides: float = Field(
        ge=0, description="chlorides, g/dm3 of sodium chloride"
    )
    free_sulfur_dioxide: float = Field(
        ge=0, description="free sulfur dioxide, mg/dm3"
    )
    total_sulfur_dioxide: float = Field(
        ge=0, description="total sulfur dioxide, mg/dm3"
    )
    density: float = Field(gt=0, description="density, g/cm3")
    ph: float = Field(ge=0, le=14, description="pH")
    sulphates: float = Field(
        ge=0, description="sulphates, g/dm3 of potassium sulphate"
    )
    alcohol: float = Field(ge=0, le=100, description="alcohol, % vol.")
