"""Input schema of the cirrhosis stage model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field

YesNo = Literal["Y", "N"]


class CirrhosisInputs(Schema):
    """Same snake_case columns the pipeline was trained on."""

    age: float = Field(ge=0, le=120, description="age, years")
    sex: Literal["F", "M"] = Field(description="sex: F (female) or M (male)")
    ascites: YesNo = Field(
        description="ascites (fluid in the abdomen): Y or N"
    )
    hepatomegaly: YesNo = Field(
        description="hepatomegaly (enlarged liver): Y or N"
    )
    spiders: YesNo = Field(
        description="spider angiomas on the skin: Y or N"
    )
    edema: Literal["N", "S", "Y"] = Field(
        description=(
            "edema: N (none, no diuretics), S (present without diuretics "
            "or resolved by them) or Y (present despite diuretics)"
        )
    )
    bilirubin: float = Field(
        ge=0, le=100, description="serum bilirubin, mg/dl"
    )
    cholesterol: float = Field(
        gt=0, le=3000, description="serum cholesterol, mg/dl"
    )
    albumin: float = Field(gt=0, le=10, description="serum albumin, g/dl")
    copper: float = Field(ge=0, le=2000, description="urine copper, ug/day")
    alk_phos: float = Field(
        gt=0, le=50000, description="alkaline phosphatase, U/liter"
    )
    sgot: float = Field(
        gt=0, le=5000, description="SGOT (aspartate aminotransferase), U/ml"
    )
    triglycerides: float = Field(
        gt=0, le=5000, description="triglycerides, mg/dl"
    )
    platelets: float = Field(
        gt=0, le=2000, description="platelets, thousands per cubic ml"
    )
    prothrombin: float = Field(
        gt=0, le=60, description="prothrombin time, seconds"
    )
