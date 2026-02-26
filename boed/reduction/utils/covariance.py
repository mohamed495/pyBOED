"""Covariance kernels for stochastic field examples."""

import numpy as np


def _pairwise_abs_diff(z, z_prime):
    if np.ndim(z) == 1 and np.ndim(z_prime) == 1:
        return np.abs(z[:, None] - z_prime[None, :])
    return np.abs(z - z_prime)


def gaussian_covariance(z, z_prime, length_scale=0.1, variance=1.0):
    """Squared-exponential covariance."""
    if np.ndim(z) == 1 and np.ndim(z_prime) == 1:
        diff = z[:, None] - z_prime[None, :]
    else:
        diff = z - z_prime
    return variance**2 * np.exp(-0.5 * (diff / length_scale) ** 2)


def delta_covariance(z, z_prime, variance=1.0):
    """Kronecker/Dirac-style covariance on matching points."""
    diff = _pairwise_abs_diff(z, z_prime)
    return np.where(diff == 0, variance**2, 0.0)


def spherical_covariance(z, z_prime, length_scale=0.1, variance=1.0):
    """Compact-support spherical covariance."""
    diff = _pairwise_abs_diff(z, z_prime)
    ratio = np.clip(diff / length_scale, 0, 1)
    term1 = np.arccos(ratio)
    term2 = ratio * np.sqrt(1 - ratio**2)
    return np.where(
        diff <= length_scale,
        (2 * variance**2 / np.pi) * (term1 - term2),
        0.0,
    )

def matern_12_covariance(z, z_prime, length_scale=0.1, variance=1.0):
    """Matern covariance with nu=1/2."""
    diff = _pairwise_abs_diff(z, z_prime)
    return variance**2 * np.exp(-diff / length_scale)


def matern_32_covariance(z, z_prime, length_scale=0.1, variance=1.0):
    """Matern covariance with nu=3/2."""
    diff = _pairwise_abs_diff(z, z_prime)
    return variance**2 * (1 + np.sqrt(3) * diff / length_scale) * np.exp(
        -np.sqrt(3) * diff / length_scale
    )


def matern_52_covariance(z, z_prime, length_scale=0.1, variance=1.0):
    """Matern covariance with nu=5/2."""
    diff = _pairwise_abs_diff(z, z_prime)
    return variance**2 * (
        1 + np.sqrt(5) * diff / length_scale + 5 * diff / (3 * length_scale**2)
    ) * np.exp(-np.sqrt(5) * diff / length_scale)
