"""Observation compression/selection helpers for inference APIs.

This module provides a small, explicit wrapper around observation mappings used
throughout the inference layer. It supports three user-facing input styles:

- sensor indices (discrete selection)
- legacy binary diagonal masks
- projection/compression matrices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _normalize_indices(indices: Any, n_obs: int) -> np.ndarray:
    """Validate and normalize sensor indices."""
    if n_obs <= 0:
        raise ValueError("n_obs must be a positive integer.")

    idx = np.asarray(indices)
    if idx.ndim != 1:
        raise ValueError("Sensor indices must be a 1D array-like.")
    if idx.size == 0:
        raise ValueError("Sensor indices must not be empty.")

    if not np.issubdtype(idx.dtype, np.integer):
        rounded = np.round(idx)
        if not np.allclose(idx, rounded):
            raise TypeError("Sensor indices must be integers.")
        idx = rounded.astype(int)
    else:
        idx = idx.astype(int, copy=False)

    if np.any(idx < 0) or np.any(idx >= int(n_obs)):
        raise ValueError(f"Sensor indices must be in [0, {int(n_obs) - 1}].")

    if np.unique(idx).size != idx.size:
        raise ValueError("Sensor indices must be unique.")

    return idx


def _is_binary_diagonal_mask(mask: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if ``mask`` is a (near) binary diagonal matrix."""
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        return False

    if not np.allclose(mask, np.diag(np.diag(mask)), atol=tol, rtol=0.0):
        return False

    diag = np.diag(mask)
    return bool(np.all(np.isclose(diag, 0.0, atol=tol) | np.isclose(diag, 1.0, atol=tol)))


@dataclass(slots=True)
class ObservationMap:
    """Canonical representation of observation compression/selection."""

    kind: str
    n_obs: int
    indices: np.ndarray | None = None
    projection: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_obs = int(self.n_obs)
        if self.n_obs <= 0:
            raise ValueError("n_obs must be a positive integer.")

        valid_kinds = {"identity", "indices", "mask", "projection"}
        if self.kind not in valid_kinds:
            raise ValueError(
                f"Unknown ObservationMap kind: {self.kind!r}. "
                f"Expected one of {sorted(valid_kinds)}."
            )

        if self.kind == "identity":
            self.indices = None
            self.projection = None
            return

        if self.kind in {"indices", "mask"}:
            if self.indices is None:
                raise ValueError(f"{self.kind!r} observation map requires indices.")
            self.indices = _normalize_indices(self.indices, self.n_obs)
            self.projection = None
            return

        if self.projection is None:
            raise ValueError("'projection' observation map requires a projection matrix.")

        proj = np.asarray(self.projection, dtype=float)
        if proj.ndim != 2:
            raise ValueError("Projection matrix must be 2D.")
        if proj.shape[0] != self.n_obs:
            raise ValueError(
                "Projection matrix row dimension must match n_obs "
                f"({proj.shape[0]} != {self.n_obs})."
            )
        if proj.shape[1] == 0:
            raise ValueError("Projection matrix must have at least one column.")
        self.projection = proj
        self.indices = None

    @classmethod
    def identity(cls, n_obs: int) -> "ObservationMap":
        """Identity map (no compression)."""
        return cls(kind="identity", n_obs=n_obs)

    @classmethod
    def from_indices(cls, indices: Any, n_obs: int) -> "ObservationMap":
        """Create a discrete sensor-selection map from active indices."""
        return cls(kind="indices", n_obs=n_obs, indices=np.asarray(indices))

    @classmethod
    def from_mask(cls, mask: Any) -> "ObservationMap":
        """Create a discrete sensor-selection map from a legacy binary mask."""
        arr = np.asarray(mask, dtype=float)
        if not _is_binary_diagonal_mask(arr):
            raise ValueError(
                "Mask must be a square binary diagonal matrix with entries in {0, 1}."
            )
        idx = np.flatnonzero(np.diag(arr) > 0.5)
        if idx.size == 0:
            raise ValueError("Mask must activate at least one observation.")
        return cls(kind="mask", n_obs=arr.shape[0], indices=idx)

    @classmethod
    def from_projection(
        cls,
        projection: Any,
        n_obs: int | None = None,
    ) -> "ObservationMap":
        """Create a compression map from a projection matrix ``U`` (shape ``(p, m)``)."""
        proj = np.asarray(projection, dtype=float)
        if proj.ndim != 2:
            raise ValueError("Projection matrix must be 2D.")
        n_obs_inferred = proj.shape[0] if n_obs is None else int(n_obs)
        return cls(kind="projection", n_obs=n_obs_inferred, projection=proj)

    @property
    def is_identity(self) -> bool:
        return self.kind == "identity"

    @property
    def is_projection(self) -> bool:
        return self.kind == "projection"

    @property
    def is_discrete(self) -> bool:
        return self.kind in {"indices", "mask"}

    @property
    def reduced_dim(self) -> int:
        """Compressed observation dimension."""
        if self.is_identity:
            return self.n_obs
        if self.is_projection:
            assert self.projection is not None
            return int(self.projection.shape[1])
        assert self.indices is not None
        return int(self.indices.size)

    def as_indices(self) -> np.ndarray:
        """Return active sensor indices for discrete/identity maps."""
        if self.is_identity:
            return np.arange(self.n_obs, dtype=int)
        if not self.is_discrete:
            raise ValueError("Projection maps do not have a unique index representation.")
        assert self.indices is not None
        return self.indices.copy()

    def as_mask(self) -> np.ndarray:
        """Return the legacy diagonal mask representation."""
        mask = np.zeros((self.n_obs, self.n_obs), dtype=float)
        mask[self.as_indices(), self.as_indices()] = 1.0
        return mask

    def as_projection(self) -> np.ndarray:
        """Return a projection/selection matrix with shape ``(n_obs, reduced_dim)``."""
        if self.is_projection:
            assert self.projection is not None
            return self.projection.copy()
        if self.is_identity:
            return np.eye(self.n_obs)
        idx = self.as_indices()
        return np.eye(self.n_obs)[:, idx]

    def apply(self, y: np.ndarray) -> np.ndarray:
        """Apply the observation map to a vector ``y``."""
        vec = np.asarray(y, dtype=float).reshape(-1)
        if vec.shape[0] != self.n_obs:
            raise ValueError(
                f"Observation vector length must match n_obs ({vec.shape[0]} != {self.n_obs})."
            )
        if self.is_projection:
            assert self.projection is not None
            return self.projection.T @ vec
        if self.is_identity:
            return vec.copy()
        assert self.indices is not None
        return vec[self.indices]


def coerce_observation_map(
    obs_map: ObservationMap | np.ndarray | list[int] | tuple[int, ...] | None,
    n_obs: int | None = None,
) -> ObservationMap | None:
    """Coerce user input into an ``ObservationMap`` instance.

    Accepted inputs
    ---------------
    - ``None``
    - ``ObservationMap``
    - 1D integer array-like of indices
    - 2D binary diagonal mask
    - 2D projection matrix
    """
    if obs_map is None:
        return None

    if isinstance(obs_map, ObservationMap):
        if n_obs is not None and obs_map.n_obs != int(n_obs):
            raise ValueError(
                f"ObservationMap n_obs mismatch ({obs_map.n_obs} != {int(n_obs)})."
            )
        return obs_map

    arr = np.asarray(obs_map)
    if arr.ndim == 1:
        if n_obs is None:
            raise ValueError("n_obs is required when coercing sensor indices.")
        return ObservationMap.from_indices(arr, n_obs=int(n_obs))

    if arr.ndim == 2:
        if _is_binary_diagonal_mask(np.asarray(arr, dtype=float)):
            return ObservationMap.from_mask(arr)
        return ObservationMap.from_projection(arr, n_obs=n_obs)

    raise TypeError(
        "obs_map must be None, an ObservationMap, a 1D index array, "
        "a 2D binary diagonal mask, or a 2D projection matrix."
    )


def normalize_observation_map(
    obs_map: ObservationMap | np.ndarray | list[int] | tuple[int, ...] | None,
    *,
    n_obs: int,
) -> ObservationMap:
    """Normalize user input into an ``ObservationMap`` (identity if ``None``)."""
    coerced = coerce_observation_map(obs_map, n_obs=n_obs)
    return ObservationMap.identity(n_obs) if coerced is None else coerced


__all__ = [
    "ObservationMap",
    "coerce_observation_map",
    "normalize_observation_map",
]
