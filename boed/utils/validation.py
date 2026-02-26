"""Validation helpers for arrays and covariance matrices."""

from typing import Optional, Tuple

import numpy as np

def validate_matrix(
    A: np.ndarray,
    name: str = "Matrix",
    min_shape: Tuple[int, int] = (1, 1),
) -> bool:
    """
    Validate a matrix.
    
    Args:
        A: Matrix to validate
        name: Name used in error messages
        min_shape: Minimum required shape
    
    Returns:
        True if valid; raises an exception otherwise
    """
    if not isinstance(A, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(A)}")
    
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}")
    
    if A.shape[0] < min_shape[0] or A.shape[1] < min_shape[1]:
        raise ValueError(f"{name} is too small: {A.shape} < {min_shape}")
    
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        raise ValueError(f"{name} contains NaN or Inf")
    
    return True


def validate_vector(
    v: np.ndarray,
    name: str = "Vector",
    expected_size: Optional[int] = None,
) -> bool:
    """
    Validate a vector.
    
    Args:
        v: Vector to validate
        name: Name used in error messages
        expected_size: Expected size (if None, no size check)
    
    Returns:
        True if valid
    """
    if not isinstance(v, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(v)}")
    
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {v.shape}")
    
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        raise ValueError(f"{name} contains NaN or Inf")
    
    if expected_size is not None and len(v) != expected_size:
        raise ValueError(f"{name} size {len(v)} ≠ {expected_size}")
    
    return True


def validate_covariance(
    Sigma: np.ndarray,
    name: str = "Covariance",
    tol: float = 1e-10,
) -> bool:
    """
    Validate a covariance matrix.
    
    Checks symmetry and positive semidefiniteness.
    
    Args:
        Sigma: Matrix expected to be a covariance
        name: Name used in error messages
        tol: Tolerance
    
    Returns:
        True if valid
    """
    validate_matrix(Sigma, name, (2, 2))
    
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"{name} must be square, got {Sigma.shape}")
    
    # Symmetry
    if not np.allclose(Sigma, Sigma.T, atol=tol):
        raise ValueError(f"{name} is not symmetric")
    
    # Positive semidefinite
    eigs = np.linalg.eigvalsh(Sigma)
    if np.any(eigs < -tol):
        raise ValueError(f"{name} has negative eigenvalues: min = {eigs[0]:.2e}")
    
    return True


def check_dimensions_match(
    A: np.ndarray,
    b: np.ndarray,
) -> bool:
    """Check that dimensions are compatible for ``A @ x = b``."""
    if A.shape[0] != len(b):
        raise ValueError(f"Incompatible dimensions: {A.shape[0]} ≠ {len(b)}")
    return True
