"""Compatibility shim for legacy imports from ``boed.core.utils``.

The utility implementations now live in focused modules under ``boed.utils``.
This module re-exports the legacy names to preserve backward compatibility.
"""
from boed.utils.config import load_config, merge_configs, save_config
from boed.utils.io import (
    _make_json_serializable,
    _make_numpy_arrays,
    load_experiment,
    save_experiment,
)
from boed.utils.linalg import logdet, trace
from boed.utils.logging_utils import Logger
from boed.utils.normalization import (
    absolute_error,
    minmax_normalize,
    normalize_columns,
    normalize_rows,
    normalize_vector,
    relative_error,
    zscore_normalize,
)
from boed.utils.observation import compute_observation_map, compute_W
from boed.utils.reporting import comparison_table, print_metrics
from boed.utils.synthetic import (
    generate_synthetic_observations,
    generate_synthetic_prior,
    generate_test_problem,
)
from boed.utils.validation import (
    check_dimensions_match,
    validate_covariance,
    validate_matrix,
    validate_vector,
)

__all__ = [
    "logdet",
    "trace",
    "compute_observation_map",
    "compute_W",
    "validate_matrix",
    "validate_vector",
    "validate_covariance",
    "check_dimensions_match",
    "generate_synthetic_prior",
    "generate_synthetic_observations",
    "generate_test_problem",
    "save_experiment",
    "load_experiment",
    "_make_json_serializable",
    "_make_numpy_arrays",
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
]
