# app/schemas.py
from pydantic import BaseModel

class CarFeatures(BaseModel):
    brand: str
    model: str
    body: str
    condition: str
    odometer: float
    saleyear: int
