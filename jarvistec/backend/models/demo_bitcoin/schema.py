from pydantic import BaseModel as Schema, Field


class BitcoinInputs(Schema):
    last_price: float = Field(description="último precio en USD")
