"""Observation selection/compression helper utilities."""

import numpy as np

from boed.inference.observation_map import ObservationMap

def compute_observation_map(N: int, indices: np.ndarray) -> ObservationMap:
    """Build a canonical ``ObservationMap`` from active observation indices.

    Parameters
    ----------
    N : int
        Full observation dimension.
    indices : np.ndarray
        Active indices in ``[0, N-1]``.
    """
    return ObservationMap.from_indices(indices, n_obs=int(N))


def compute_W(N: int, indices: np.ndarray, format: str = "mask") -> np.ndarray:
    """Build legacy selection matrices from active indices.

    Parameters
    ----------
    N : int
        Full observation dimension.
    indices : np.ndarray
        Active sensor indices.
    format : str, default="mask"
        Output format:
        - "mask" (legacy): diagonal 0/1 matrix of shape (N, N)
        - "selection" / "U": column selection matrix of shape (N, m)
        - "left_selection": row selection matrix of shape (m, N)

    Notes
    -----
    Prefer ``compute_observation_map(...)`` for new code. The returned
    ``ObservationMap`` can be passed to inference and design APIs that accept
    ``obs_map=...``. ``compute_W(...)`` is kept for backward compatibility
    with legacy matrix-based code paths.
    """
    N = int(N)
    obs_map = compute_observation_map(N, indices)
    idx = obs_map.as_indices()

    fmt = str(format).lower()

    if fmt == "mask":
        W = np.zeros((N, N))
        W[idx, idx] = 1.0
        return W

    if fmt in {"selection", "u"}:
        return np.eye(N)[:, idx]

    if fmt == "left_selection":
        return np.eye(N)[idx, :]

    raise ValueError(
        "Unknown format for compute_W. Expected 'mask', 'selection'/'U', or 'left_selection'."
    )
