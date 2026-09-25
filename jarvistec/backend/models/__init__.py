"""Registro explícito de los modelos que expone JarvisTEC.

Cada modelo es un paquete aislado (backend/models/<nombre>/) con su adaptador
y su esquema. Para activarlo basta con agregar su clase a MODEL_CLASSES; el
orquestador (backend/core/orchestrator.py) lo registra y ejecuta, y la API
falla al iniciar si hay nombres duplicados.
"""
from .base import BaseModel
from .demo_bitcoin import DemoBitcoin
from .demo_vino import DemoVino

MODEL_CLASSES: list[type[BaseModel]] = [
    DemoBitcoin,  # demo, sin entrenar
    DemoVino,     # demo, sin entrenar
]
