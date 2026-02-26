"""Synthetic data generation for reduction demos."""

from typing import Tuple

import numpy as np


def generate_structured_data(
    n_samples: int = 500,
    d_ambient: int = 50,
    d_intrinsic: int = 5,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Generate low-rank noisy data in ambient dimension.

    Returns
    -------
    X : ndarray, shape (n_samples, d_ambient)
    info : dict with latent/eigensystem details
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if d_intrinsic < 1:
        raise ValueError("d_intrinsic must be >= 1")
    if d_ambient < d_intrinsic:
        raise ValueError("d_ambient must be >= d_intrinsic")
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0")

    rng = np.random.RandomState(random_state)

    base_eigs = np.array([10.0, 5.0, 2.0], dtype=float)
    if d_intrinsic <= base_eigs.size:
        true_eigenvalues = base_eigs[:d_intrinsic]
    else:
        extra_count = d_intrinsic - base_eigs.size
        extra_eigs = base_eigs[-1] * (0.5 ** np.arange(1, extra_count + 1))
        true_eigenvalues = np.concatenate([base_eigs, extra_eigs])
    true_eigenvalues = true_eigenvalues / true_eigenvalues.sum()

    latent_vars = rng.randn(n_samples, d_intrinsic) * np.sqrt(true_eigenvalues).reshape(1, -1)

    mixing_matrix = rng.randn(d_ambient, d_intrinsic)
    mixing_matrix, _ = np.linalg.qr(mixing_matrix)

    X_signal = latent_vars @ mixing_matrix.T
    noise = rng.randn(n_samples, d_ambient) * noise_level
    X = X_signal + noise

    mean_shift = rng.randn(d_ambient) * 0.5
    X = X + mean_shift

    info = {
        "true_eigenvalues": true_eigenvalues,
        "true_mixing_matrix": mixing_matrix,
        "latent_vars": latent_vars,
        "mean_shift": mean_shift,
        "noise_level": noise_level,
    }
    return X, info
