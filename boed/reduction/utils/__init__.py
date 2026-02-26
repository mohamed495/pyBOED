"""Utility functions used by dimensionality-reduction demos and workflows."""

from .covariance import (
    gaussian_covariance,
    delta_covariance,
    spherical_covariance,
    matern_12_covariance,
    matern_32_covariance,
    matern_52_covariance,
)
from .problems import (
    DiffusionProblem1D,
    kappa_piecewise,
    kappa_smooth,
    u0_gaussian,
)
from .data_generation import generate_structured_data

__all__ = [
    "gaussian_covariance",
    "delta_covariance",
    "spherical_covariance",
    "matern_12_covariance",
    "matern_32_covariance",
    "matern_52_covariance",
    "DiffusionProblem1D",
    "kappa_piecewise",
    "kappa_smooth",
    "u0_gaussian",
    "generate_structured_data",
]
