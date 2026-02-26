"""Public inference API for posterior and inverse models.

This module provides stable, user-facing imports for inference classes.
The implementation is organized across focused modules under ``boed.inference``.

Examples
--------
>>> from boed.inference import LinearGaussianModel, ObservationMap
>>> model = LinearGaussianModel(A, Sigma_noise, mu_prior, Sigma_prior)
>>> obs = ObservationMap.from_indices([0, 2, 4], n_obs=A.shape[0])
>>> post = model.posterior_result(y, obs_map=obs)
>>> samples = model.sample(y, n_samples=100, obs_map=obs)
"""

from .base import InverseModel
from .linear_gaussian import LinearGaussianModel
from .nonlinear_laplace import NonLinearInverseLaplace, NonlinearLaplaceModel
from .observation_map import (
    ObservationMap,
    coerce_observation_map,
    normalize_observation_map,
)
from .results import LaplaceResult, MCMCResult, PosteriorResult

__all__ = [
    "InverseModel",
    "LinearGaussianModel",
    "NonlinearLaplaceModel",
    "NonLinearInverseLaplace",
    "ObservationMap",
    "coerce_observation_map",
    "normalize_observation_map",
    "PosteriorResult",
    "LaplaceResult",
    "MCMCResult",
]
