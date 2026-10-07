"""Input schema of the stroke risk model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field


class StrokeInputs(Schema):
    """Same snake_case columns the pipeline was trained on."""

    gender: Literal["Female", "Male"] = Field(
        description="gender: Female or Male"
    )
    age: float = Field(ge=0, le=120, description="age, years")
    hypertension: Literal[0, 1] = Field(
        description="has hypertension: 1 = yes, 0 = no"
    )
    heart_disease: Literal[0, 1] = Field(
        description="has a heart disease: 1 = yes, 0 = no"
    )
    ever_married: Literal["Yes", "No"] = Field(
        description="has ever been married: Yes or No"
    )
    work_type: Literal[
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ] = Field(
        description=(
            "type of work: Private, Self-employed, Govt_job, children or "
            "Never_worked"
        )
    )
    residence_type: Literal["Urban", "Rural"] = Field(
        description="residence: Urban or Rural"
    )
    avg_glucose_level: float = Field(
        ge=30,
        le=600,
        description="average blood glucose level, mg/dL (unit assumed)",
    )
    bmi: float = Field(ge=8, le=100, description="body mass index, kg/m2")
    smoking_status: Literal[
        "never smoked", "formerly smoked", "smokes", "Unknown"
    ] = Field(
        description=(
            "smoking status: never smoked, formerly smoked, smokes or "
            "Unknown"
        )
    )
