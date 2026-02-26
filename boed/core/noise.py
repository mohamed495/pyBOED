# ============================================================================
# boed/core/noise.py
# ============================================================================
"""
Noise models for observations.

Implements various noise structures: white, colored, heteroscedastic.
"""

import numpy as np
from typing import Optional
from scipy.linalg import toeplitz

from boed.core.base import NoiseModelBase


class NoiseModel(NoiseModelBase):
    """
    White Gaussian noise model.

    η ~ N(0, σ²I)

    Parameters
    ----------
    sigma_noise : float
        Standard deviation of noise
    """

    def __init__(self, sigma_noise: float):
        if sigma_noise <= 0:
            raise ValueError(f"Noise std must be positive, got {sigma_noise}")
        # Keep both names for backward compatibility across examples/notebooks.
        self.sigma = sigma_noise
        self.sigma_noise = sigma_noise

    def get_covariance(self, size: int) -> np.ndarray:
        """Get diagonal covariance matrix."""
        return (self.sigma**2) * np.eye(size)

    def sample(self, size: int, n_samples: int = 1) -> np.ndarray:
        """Sample from noise distribution."""
        return np.random.normal(0, self.sigma, size=(n_samples, size))

    def __repr__(self) -> str:
        return f"WhiteNoise(σ={self.sigma:.4f})"


class ColoredNoise(NoiseModelBase):
    """
    Colored (correlated) Gaussian noise with exponential correlation.

    Covariance: C[i,j] = σ² exp(-|i-j|/ℓ)
    """

    def __init__(self, sigma_noise: float, correlation_length: float = 1.0):
        self.sigma = sigma_noise
        self.ell = correlation_length

    def get_covariance(self, size: int) -> np.ndarray:
        """Get Toeplitz covariance matrix."""
        r = np.arange(size)
        c = (self.sigma**2) * np.exp(-r / self.ell)
        return toeplitz(c)

    def sample(self, size: int, n_samples: int = 1) -> np.ndarray:
        """Sample using Cholesky decomposition."""
        C = self.get_covariance(size)
        L = np.linalg.cholesky(C)
        z = np.random.randn(n_samples, size)
        return (L @ z.T).T


class HeteroscedasticNoise(NoiseModelBase):
    """
    Heteroscedastic (non-uniform) noise model.

    σ[i] varies with location or measurement type.
    """

    def __init__(self, sigma_vector: np.ndarray):
        self.sigma = np.asarray(sigma_vector)
        if np.any(self.sigma <= 0):
            raise ValueError("All noise variances must be positive")

    def get_covariance(self, size: Optional[int] = None) -> np.ndarray:
        """Get diagonal covariance with varying variances."""
        if size and size != len(self.sigma):
            raise ValueError(f"Size mismatch: {size} vs {len(self.sigma)}")
        return np.diag(self.sigma**2)

    def sample(self, size: Optional[int] = None, n_samples: int = 1) -> np.ndarray:
        """Sample with location-dependent variance."""
        if size is None:
            size = len(self.sigma)
        return np.random.normal(0, 1, (n_samples, size)) * self.sigma

