"""
Compatibility namespace mirroring the historical ``dimreduction.methods`` API.
"""

from ..inference import ActiveSubspaces, DataAveragedLIS, LikelihoodInformedSubspaces
from ..inference.active_subspace import generate_ridge_function
from ..linear import KLE, PCA, POD
from ..parametric import APosterioriError, ParametricProblemRB, ReducedBasis
from ..utils import (
    DiffusionProblem1D,
    delta_covariance,
    matern_12_covariance,
    gaussian_covariance,
    generate_structured_data,
    kappa_piecewise,
    kappa_smooth,
    matern_32_covariance,
    matern_52_covariance,
    spherical_covariance,
    u0_gaussian,
)

__all__ = [
    "PCA",
    "KLE",
    "POD",
    "ReducedBasis",
    "ParametricProblemRB",
    "APosterioriError",
    "ActiveSubspaces",
    "LikelihoodInformedSubspaces",
    "DataAveragedLIS",
    "generate_structured_data",
    "generate_ridge_function",
    "matern_12_covariance",
    "gaussian_covariance",
    "delta_covariance",
    "spherical_covariance",
    "matern_32_covariance",
    "matern_52_covariance",
    "DiffusionProblem1D",
    "kappa_piecewise",
    "kappa_smooth",
    "u0_gaussian",
]
