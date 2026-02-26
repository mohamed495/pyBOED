"""Compatibility shim for legacy imports from ``boed.design.greedy``.

Greedy design implementations now live in ``boed.design._greedy_impl``.
This module re-exports the legacy public functions.
"""
from boed.design._greedy_impl import (
    run_greedy_oed,
    run_greedy_oed_LG,
    run_sboed_trajectories,
    run_sequential_step_oed,
)

__all__ = [
    "run_greedy_oed",
    "run_greedy_oed_LG",
    "run_sequential_step_oed",
    "run_sboed_trajectories",
]
