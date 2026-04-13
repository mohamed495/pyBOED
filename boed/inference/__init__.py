"""Public inference API for posterior and inverse models.

This module provides stable, user-facing imports for inference classes.
The implementation is organized across focused modules under ``boed.inference``.

Examples
--------
>>> from boed.inference import LinearGaussianModel
>>> import numpy as np
>>> model = LinearGaussianModel(model=A, Sigma_obs=Sigma_noise, mu_prior=mu_prior, Sigma_prior=Sigma_prior)
>>> W = np.eye(A.shape[0])[:, [0, 2, 4]]
>>> post = model.posterior_result(y, W=W)
>>> samples = model.sample(y, n_samples=100, W=W)
"""

from .base import InverseModel
from .linear_gaussian import LinearGaussianModel
from .nonlinear_laplace import NonLinearInverseLaplace, NonlinearLaplaceModel
from .results import LaplaceResult, MCMCResult, PosteriorResult


__all__ = [
    "InverseModel",
    "LinearGaussianModel",
    "NonlinearLaplaceModel",
    "NonLinearInverseLaplace",
    "PosteriorResult",
    "LaplaceResult",
    "MCMCResult",
]
