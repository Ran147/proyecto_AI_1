"""Stroke risk: Random Forest trained in training/stroke/.

Artifacts: artifacts/stroke.joblib and artifacts/stroke.meta.json (generated
by training/stroke/notebook.ipynb).
"""
from .adapter import Stroke

__all__ = ["Stroke"]
