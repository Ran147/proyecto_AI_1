"""Input schema of the hepatitis C category model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field


class HepatitisInputs(Schema):
    """Same snake_case columns the pipeline was trained on.

    The units are the usual reporting units of each test; the dataset does
    not document them (see training/hepatitis/README.md).
    """

    age: int = Field(ge=0, le=120, description="age, years")
    sex: Literal["m", "f"] = Field(description="sex: m (male) or f (female)")
    alb: float = Field(ge=0, le=100, description="albumin, g/L")
    alp: float = Field(
        ge=0, le=2000, description="alkaline phosphatase, U/L"
    )
    alt: float = Field(
        ge=0, le=5000, description="alanine aminotransferase, U/L"
    )
    ast: float = Field(
        ge=0, le=5000, description="aspartate aminotransferase, U/L"
    )
    bil: float = Field(ge=0, le=1000, description="bilirubin, umol/L")
    che: float = Field(ge=0, le=50, description="cholinesterase, kU/L")
    chol: float = Field(ge=0, le=30, description="cholesterol, mmol/L")
    crea: float = Field(ge=0, le=3000, description="creatinine, umol/L")
    ggt: float = Field(
        ge=0, le=5000, description="gamma-glutamyl transferase, U/L"
    )
    prot: float = Field(ge=0, le=150, description="total protein, g/L")
