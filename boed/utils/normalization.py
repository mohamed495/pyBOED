"""Normalization and error metrics utilities."""

import numpy as np

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector: ``v / ||v||``."""
    return v / (np.linalg.norm(v) + 1e-15)


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """Normalize the rows of ``A``."""
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (row_norms + 1e-15)


def normalize_columns(A: np.ndarray) -> np.ndarray:
    """Normalize the columns of ``A``."""
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    return A / (col_norms + 1e-15)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalization: ``(x - mean) / std``."""
    return (x - np.mean(x)) / (np.std(x) + 1e-15)


def minmax_normalize(x: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """Min-max normalize to the range ``[a, b]``."""
    x_min = np.min(x)
    x_max = np.max(x)
    return a + (b - a) * (x - x_min) / (x_max - x_min + 1e-15)


def relative_error(x_true: np.ndarray, x_approx: np.ndarray) -> float:
    """Relative error: ``||x_true - x_approx|| / ||x_true||``."""
    return np.linalg.norm(x_true - x_approx) / (np.linalg.norm(x_true) + 1e-15)


def absolute_error(x_true: np.ndarray, x_approx: np.ndarray) -> float:
    """Absolute error: ``||x_true - x_approx||``."""
    return np.linalg.norm(x_true - x_approx)
