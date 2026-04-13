"""Observation selection/compression helper utilities."""

import numpy as np


def build_selection_matrices(n_obs: int, active_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build selection matrices W (active) et Wc (unactive).

    W.T  @ y  →  active sensors observations
    Wc.T @ y  →  unactive sensors observations
    """
    active_indices = np.unique(np.asarray(active_indices, dtype=int))

    if np.any(active_indices < 0) or np.any(active_indices >= n_obs):
        raise ValueError("Indices out of bound.")

    inactive_indices = np.setdiff1d(np.arange(n_obs), active_indices)

    def _make_selector(n_rows, cols):
        S = np.zeros((n_rows, len(cols)), dtype=int)
        S[cols, np.arange(len(cols))] = 1
        return S

    return _make_selector(n_obs, active_indices), _make_selector(n_obs, inactive_indices)


def reduce_system(
    y: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    W: np.ndarray | None = None,
    Wc: np.ndarray | None = None,
) -> dict:
    """Reduce the full observation system (y = Hx + ε) to selected sensors.

    Parameters
    ----------
    y  : full observations vector
    H  : observation matrix          (n_obs × n_state)
    R  : observation noise covariance (n_obs × n_obs)
    W  : active sensors selection matrix   (None → keep all)
    Wc : inactive sensors selection matrix (optional)
    """
    y = np.asarray(y, dtype=float).ravel()
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)

    if W is None:
        return {"y_active": y, "H_active": H, "R_active": R}

    if y.size != W.shape[0]:
        raise ValueError(f"y must have {W.shape[0]} elements, got {y.size}.")

    def _project(M, left, right=None):
        return left.T @ M @ right if right is not None else left.T @ M

    result = {
        "y_active": _project(y, W),
        "H_active": _project(H, W),
        "R_active": _project(R, W, W),
    }

    if Wc is not None:
        result.update({
            "y_inactive": _project(y, Wc),
            "H_inactive": _project(H, Wc),
            "R_inactive": _project(R, Wc, Wc),
        })

    return result

def parse_theta(theta: np.ndarray, n_param : int , name: str = "theta") -> np.ndarray:
    theta = np.asarray(theta, dtype=float).ravel()
    expected = n_param
    if theta.size != expected:
        raise ValueError(f"{name} must have length {expected}, got {theta.size}.")
    return theta