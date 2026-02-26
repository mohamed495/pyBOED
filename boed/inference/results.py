"""Structured result containers for inference APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PosteriorResult:
    """Posterior mean/covariance and optional metadata."""

    mean: np.ndarray
    cov: np.ndarray
    obs_dim: int | None = None
    compression_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LaplaceResult:
    """Laplace approximation outputs around a MAP point."""

    mean: np.ndarray | None
    cov: np.ndarray
    precision: np.ndarray
    jacobian: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MCMCResult:
    """Posterior sample container with sampler diagnostics."""

    samples: np.ndarray
    acceptance_rate: float | None = None
    initial_state: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PosteriorResult",
    "LaplaceResult",
    "MCMCResult",
]
