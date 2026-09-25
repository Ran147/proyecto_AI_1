from pydantic import BaseModel as Schema, Field


class VinoInputs(Schema):
    alcohol: float = Field(description="% de alcohol")
    volatile_acidity: float = Field(description="acidez volátil")
