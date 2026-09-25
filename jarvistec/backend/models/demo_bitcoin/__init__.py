"""DEMO (no entrenado): precio de Bitcoin. Aplica +1% al último precio.

Existe solo para probar el flujo API/comandos de punta a punta. No tiene
dataset, notebook ni artefacto; reemplazar por el modelo real cuando exista.
"""
from .adapter import DemoBitcoin

__all__ = ["DemoBitcoin"]
