"""Linear algebra helper utilities."""

import numpy as np

def logdet(matrix: np.ndarray) -> float:
    """Compute log determinant safely for symmetric positive definite matrices."""
    sign, logdet_val = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Matrix must be positive definite for logdet.")
    return float(logdet_val)


def trace(matrix: np.ndarray) -> float:
    return float(np.trace(matrix))
