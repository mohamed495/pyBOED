"""Compatibility shim for legacy imports from ``boed.design.selection``.

Sensor-selection implementations now live in ``boed.design._selection_impl``.
This module re-exports the legacy public functions.
"""
from boed.design._selection_impl import (
    compare_to_greedy,
    evaluate_design,
    score_design,
    select_sensors_maxvol,
    select_sensors_qr_pivot,
)

__all__ = [
    "select_sensors_qr_pivot",
    "select_sensors_maxvol",
    "evaluate_design",
    "score_design",
    "compare_to_greedy",
]
