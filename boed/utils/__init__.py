"""Utility helpers grouped by concern and re-exported for convenience.

This package exposes the most commonly used helpers while keeping the
implementation split across focused submodules (linear algebra, validation,
I/O, reporting, etc.).
"""
from .linalg import logdet, trace
from .observation import build_selection_matrices, reduce_system, parse_theta
from .validation import validate_matrix, validate_vector, validate_covariance, check_dimensions_match
from .io import load_experiment, save_experiment
from .logging_utils import Logger
from .normalization import (
    absolute_error,
    minmax_normalize,
    normalize_columns,
    normalize_rows,
    normalize_vector,
    relative_error,
    zscore_normalize,
)
from .reporting import comparison_table, print_metrics
from .config import load_config, merge_configs, save_config

__all__ = [
    "logdet",
    "trace",
    "compute_W",
    "validate_matrix",
    "validate_vector",
    "validate_covariance",
    "check_dimensions_match",
    "save_experiment",
    "load_experiment",
    "Logger",
    "normalize_vector",
    "normalize_rows",
    "normalize_columns",
    "zscore_normalize",
    "minmax_normalize",
    "relative_error",
    "absolute_error",
    "print_metrics",
    "comparison_table",
    "save_config",
    "load_config",
    "merge_configs",
    "build_selection_matrices", 
    "reduce_system", 
    "parse_theta",
]
