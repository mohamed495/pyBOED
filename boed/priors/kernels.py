# ============================================================================
# boed/core/prior.py
# ============================================================================
"""
Prior covariance kernels for Gaussian processes.

Implements various stationary and non-stationary kernels for defining
prior distributions over parameter fields.
"""

import numpy as np
from boed.core.base import KernelBase

class Gaussian(KernelBase):
    """
    RBF/Gaussian kernel.

    k(x, x') = σ² exp(-||x - x'||² / (2l²))

    Infinitely differentiable, very smooth.

    Parameters
    ----------
    length_scale : float, default=0.2
        Correlation length l
    sigma : float, default=1.0
        Signal variance σ²
    
    Examples
    --------
    >>> kernel = Gaussian(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __init__(self, length_scale: float = 0.2, sigma: float = 1.0):
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        super().__init__(length_scale=length_scale, sigma=sigma)
        self.l = length_scale
        self.sigma = sigma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        d = np.abs(np.subtract.outer(x1, x2))
        return (self.sigma**2) * np.exp(-(d**2) / (2 * self.l**2))

    def gradient(self, x1: np.ndarray, x2: np.ndarray, param: str) -> np.ndarray:
        """Gradient w.r.t. hyperparameters."""
        K = self(x1, x2)
        d = np.abs(np.subtract.outer(x1, x2))

        if param == "length_scale":
            return K * (d**2) / (self.l**3)
        elif param == "sigma":
            return 2 * K / self.sigma
        else:
            raise ValueError(f"Unknown parameter: {param}")


class Matern12(KernelBase):
    """
    Matérn kernel with ν=1/2 (Exponential kernel).

    k(x, x') = σ² exp(-|x - x'| / ℓ)

    Once differentiable in mean square.
    """

    def __init__(self, length_scale: float = 0.2, sigma: float = 1.0):
        super().__init__(length_scale=length_scale, sigma=sigma)
        self.l = length_scale
        self.sigma = sigma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        d = np.abs(np.subtract.outer(x1, x2))
        return (self.sigma**2) * np.exp(-d / self.l)


class Matern32(KernelBase):
    """
    Matérn kernel with ν=3/2.

    k(x, x') = σ² (1 + √3r/ℓ) exp(-√3r/ℓ)

    Once differentiable.
    """

    def __init__(self, length_scale: float = 0.2, sigma: float = 1.0):
        super().__init__(length_scale=length_scale, sigma=sigma)
        self.l = length_scale
        self.sigma = sigma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        d = np.abs(np.subtract.outer(x1, x2))
        r = np.sqrt(3) * d / self.l
        return (self.sigma**2) * (1 + r) * np.exp(-r)


class Matern52(KernelBase):
    """
    Matérn kernel with ν=5/2.

    k(x, x') = σ² (1 + √5r/ℓ + 5r²/(3ℓ²)) exp(-√5r/ℓ)

    Twice differentiable.
    """

    def __init__(self, length_scale: float = 0.2, sigma: float = 1.0):
        super().__init__(length_scale=length_scale, sigma=sigma)
        self.l = length_scale
        self.sigma = sigma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        d = np.abs(np.subtract.outer(x1, x2))
        r = np.sqrt(5) * d / self.l
        return (self.sigma**2) * (1 + r + r**2 / 3) * np.exp(-r)


class RationalQuadratic(KernelBase):
    """
    Rational quadratic kernel (mixture of SE kernels).

    k(x, x') = σ² (1 + ||x-x'||²/(2αℓ²))^(-α) 
    """

    def __init__(self, length_scale: float = 0.2, sigma: float = 1.0, alpha: float = 1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, alpha=alpha)
        self.l = length_scale
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        d = np.abs(np.subtract.outer(x1, x2))
        return (self.sigma**2) * (1 + d**2 / (2 * self.alpha * self.l**2)) ** (
            -self.alpha
        )


class Periodic(KernelBase):
    """
    Periodic kernel for periodic phenomena.

    k(x, x') = σ² exp(-2 sin²(π|x-x'|/p) / ℓ²)
    """

    def __init__(
        self, length_scale: float = 0.2, sigma: float = 1.0, period: float = 1.0
    ):
        super().__init__(length_scale=length_scale, sigma=sigma, period=period)
        self.l = length_scale
        self.sigma = sigma
        self.period = period

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        d = np.abs(np.subtract.outer(x1, x2))
        arg = np.pi * d / self.period
        return (self.sigma**2) * np.exp(-2 * np.sin(arg) ** 2 / (self.l**2))


# Backward-compatible re-export: keep ``boed.priors.kernels.GaussianProcessPrior``
# pointing to the canonical implementation in ``boed.priors.gp_priors``.
from .gp_priors import GaussianProcessPrior  # noqa: E402
