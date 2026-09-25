"""Explicit registry of the models exposed by JarvisTEC.

Each model is an isolated package (backend/models/<name>/) with its adapter
and input schema. Adding its class to MODEL_CLASSES activates it: the
orchestrator (backend/core/orchestrator.py) registers and runs it, and the
API refuses to start if two models share a name.
"""
from .base import BaseModel
from .demo_bitcoin import DemoBitcoin
from .wine import Wine

MODEL_CLASSES: list[type[BaseModel]] = [
    DemoBitcoin,  # demo, not trained
    Wine,
]
