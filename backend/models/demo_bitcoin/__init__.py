"""DEMO (not trained): Bitcoin price, computed as the last price + 1%.

It only exists to exercise the API and command flow end to end. It has no
dataset, notebook or artifact; replace it with the real model.
"""
from .adapter import DemoBitcoin

__all__ = ["DemoBitcoin"]
