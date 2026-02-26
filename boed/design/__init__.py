"""Design criteria and optimization routines."""

from .criteria import DesignCriteria
from .greedy import run_greedy_oed, run_greedy_oed_LG, run_sboed_trajectories, run_sequential_step_oed
from .selection import (
    select_sensors_qr_pivot,
    select_sensors_maxvol,
    evaluate_design,
    score_design,
    compare_to_greedy,
)

__all__ = [
    "DesignCriteria",
    "run_greedy_oed",
    "run_greedy_oed_LG",
    "select_sensors_qr_pivot",
    "select_sensors_maxvol",
    "evaluate_design",
    "score_design",
    "compare_to_greedy",
]
