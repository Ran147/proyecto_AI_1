"""Explicit registry of the models exposed by JarvisTEC.

Each model is an isolated package (backend/models/<name>/) with its adapter
and input schema. Adding its class to MODEL_CLASSES activates it: the
orchestrator (backend/core/orchestrator.py) registers and runs it, and the
API refuses to start if two models share a name.
"""
from .avocado import Avocado
from .base import BaseModel
from .bitcoin import Bitcoin
from .body_fat import BodyFat
from .car_price import CarPrice
from .churn import Churn
from .cirrhosis import Cirrhosis
from .hepatitis import Hepatitis
from .movies import Movies
from .stroke import Stroke
from .wine import Wine

MODEL_CLASSES: list[type[BaseModel]] = [
    Wine,
    CarPrice,
    Churn,
    Stroke,
    Avocado,
    Bitcoin,
    BodyFat,
    Movies,
    Hepatitis,
    Cirrhosis,
]
