"""Initial condition utilities for tests and examples.

Provides a small library of reproducible 1D initial conditions (u0) that can
be used across experiments, demos, and unit tests.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np


def gaussian_bump(x: np.ndarray, center: float = 0.3, width: float = 0.1, amplitude: float = 1.0) -> np.ndarray:
    """Single Gaussian bump."""
    if width <= 0:
        raise ValueError("width must be positive")
    x = np.asarray(x)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def double_gaussian(
    x: np.ndarray,
    centers=(0.3, 0.7),
    widths=(0.12, 0.03),
    amplitudes=(1.0, 0.5),
) -> np.ndarray:
    """Sum of two Gaussian bumps."""
    if len(centers) != 2 or len(widths) != 2 or len(amplitudes) != 2:
        raise ValueError("centers, widths, amplitudes must be length-2")
    return (
        gaussian_bump(x, centers[0], widths[0], amplitudes[0])
        + gaussian_bump(x, centers[1], widths[1], amplitudes[1])
    )


def sine_mode(
    x: np.ndarray,
    k: int = 1,
    amplitude: float = 1.0,
    phase: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Single sine mode."""
    x = np.asarray(x)
    return offset + amplitude * np.sin(2 * np.pi * k * x + phase)


def box_pulse(x: np.ndarray, start: float = 0.25, end: float = 0.5, amplitude: float = 1.0) -> np.ndarray:
    """Box pulse on [start, end]."""
    if end <= start:
        raise ValueError("end must be > start")
    x = np.asarray(x)
    return amplitude * ((x >= start) & (x <= end)).astype(float)


def ramp(x: np.ndarray, start: float = 0.0, end: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """Linear ramp between start and end, 0 outside."""
    if end <= start:
        raise ValueError("end must be > start")
    x = np.asarray(x)
    y = (x - start) / (end - start)
    y = np.clip(y, 0.0, 1.0)
    return amplitude * y


def random_fourier(
    x: np.ndarray,
    n_modes: int = 6,
    amplitude: float = 1.0,
    decay: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Smooth random field from a truncated Fourier series (deterministic with seed)."""
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    ks = np.arange(1, n_modes + 1)
    coeff_sin = rng.normal(size=n_modes) / (ks ** decay)
    coeff_cos = rng.normal(size=n_modes) / (ks ** decay)

    u = np.zeros_like(x, dtype=float)
    for i, k in enumerate(ks):
        u += coeff_sin[i] * np.sin(2 * np.pi * k * x)
        u += coeff_cos[i] * np.cos(2 * np.pi * k * x)

    max_abs = np.max(np.abs(u))
    if max_abs > 0:
        u = amplitude * u / max_abs
    return u


@dataclass(frozen=True)
class U0Spec:
    """Metadata and builder for a named initial condition."""

    name: str
    builder: Callable[..., np.ndarray]
    defaults: Dict[str, object]
    description: str
    tags: Tuple[str, ...] = ()

    def build(self, x: np.ndarray, **overrides) -> np.ndarray:
        params = dict(self.defaults)
        params.update(overrides)
        return self.builder(x, **params)


def _build_registry() -> Dict[str, U0Spec]:
    specs = [
        U0Spec(
            name="gaussian",
            builder=gaussian_bump,
            defaults={"center": 0.3, "width": 0.12, "amplitude": 1.0},
            description="Single Gaussian bump.",
            tags=("smooth", "local"),
        ),
        U0Spec(
            name="double_gaussian",
            builder=double_gaussian,
            defaults={
                "centers": (0.3, 0.7),
                "widths": (0.12, 0.03),
                "amplitudes": (1.0, 0.5),
            },
            description="Sum of two Gaussian bumps.",
            tags=("smooth", "local"),
        ),
        U0Spec(
            name="sine",
            builder=sine_mode,
            defaults={"k": 2, "amplitude": 0.7, "phase": 0.2, "offset": 0.0},
            description="Single sine mode.",
            tags=("periodic",),
        ),
        U0Spec(
            name="box",
            builder=box_pulse,
            defaults={"start": 0.25, "end": 0.5, "amplitude": 1.0},
            description="Box pulse on an interval.",
            tags=("piecewise",),
        ),
        U0Spec(
            name="ramp",
            builder=ramp,
            defaults={"start": 0.0, "end": 1.0, "amplitude": 1.0},
            description="Linear ramp on an interval.",
            tags=("piecewise", "trend"),
        ),
        U0Spec(
            name="random_fourier",
            builder=random_fourier,
            defaults={"n_modes": 6, "amplitude": 1.0, "decay": 1.5, "seed": 0},
            description="Smooth random field from truncated Fourier series.",
            tags=("random", "smooth"),
        ),
    ]
    return {spec.name: spec for spec in specs}


_U0_REGISTRY = _build_registry()
DEFAULT_U0_KIND = "double_gaussian"


def available_u0() -> Iterable[str]:
    """Return available u0 names."""
    return _U0_REGISTRY.keys()


def u0_registry() -> Dict[str, U0Spec]:
    """Return a copy of the u0 registry."""
    return dict(_U0_REGISTRY)


def get_u0_spec(kind: str) -> U0Spec:
    """Return metadata for a given u0 kind."""
    if kind not in _U0_REGISTRY:
        raise ValueError(f"Unknown kind '{kind}'. Available: {list(_U0_REGISTRY.keys())}")
    return _U0_REGISTRY[kind]


def make_u0(x: np.ndarray, kind: str = DEFAULT_U0_KIND, **overrides) -> np.ndarray:
    """Build an initial condition by name with optional overrides.

    Example
    -------
    >>> x = np.linspace(0, 1, 100)
    >>> u0 = make_u0(x, "gaussian", center=0.4, width=0.08)
    """
    spec = get_u0_spec(kind)
    return spec.build(x, **overrides)


def make_u0_bank(x: np.ndarray, kinds: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
    """Build a dictionary of multiple initial conditions."""
    if kinds is None:
        kinds = _U0_REGISTRY.keys()
    return {k: make_u0(x, k) for k in kinds}
