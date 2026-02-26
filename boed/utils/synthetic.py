"""Synthetic test-problem generation utilities."""

from typing import Dict, Optional

import numpy as np

def generate_synthetic_prior(
    N: int,
    sigma: float = 1.0,
    l: float = 10.0,
    expo: float = 1.0,
) -> np.ndarray:
    """
    Generate a prior covariance using an exponential-family kernel.
    
    Σ[i,j] = σ² * exp(-||x_i - x_j||^expo / l^expo)
    
    Args:
        N: Parameter dimension
        sigma: Amplitude
        l: Correlation length
        expo: Exponent (1 = exponential, 2 = Gaussian)
    
    Returns:
        Covariance matrix with shape ``(N, N)``
    """
    x = np.linspace(0, 1, N)
    X, X_t = np.meshgrid(x, x)
    distances = np.abs(X - X_t)
    
    Sigma = sigma**2 * np.exp(-(distances / l)**expo)
    
    return Sigma


def generate_synthetic_observations(
    A: np.ndarray,
    m_true: np.ndarray,
    sigma_obs: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate noisy observations.
    
    y = A @ m_true + noise
    
    Args:
        A: Forward operator matrix
        m_true: True parameter vector
        sigma_obs: Observation noise term (added directly)
        random_state: Random seed
    
    Returns:
        Noisy observations with shape ``(m,)``
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_true = A @ m_true
    
    return y_true + sigma_obs


def generate_test_problem(
    N: int = 100,
    n_obs: int = 50,
    n_times: int = 3,
    sigma_noise: float = 1e-3,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a complete synthetic test problem.
    
    Args:
        N: Parameter dimension
        n_obs: Number of sensors
        n_times: Number of observation times
        sigma_noise: Noise level
        random_state: Random seed
    
    Returns:
        Dictionary containing:
          - 'm_true': true parameter
          - 'Sigma_prior': prior covariance
          - 'A': forward matrix
          - 'G': design matrix
          - 'Y_obs': observations
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Prior
    Sigma_prior = generate_synthetic_prior(N, sigma=0.1, l=10.0)
    
    # True parameter
    m_true = np.sin(2 * np.pi * np.linspace(0, 1, N))
    
    # Synthetic matrices
    A = np.random.randn(N, N)
    A = A / np.linalg.norm(A, axis=0)  # Normalize
    
    # Design matrix (times + sensors)
    G = np.random.randn(n_obs * n_times, N)
    
    # Observations
    Y_obs = generate_synthetic_observations(G, m_true, sigma_noise, random_state)
    
    return {
        'm_true': m_true,
        'Sigma_prior': Sigma_prior,
        'A': A,
        'G': G,
        'Y_obs': Y_obs,
        'sigma_noise': sigma_noise,
    }
