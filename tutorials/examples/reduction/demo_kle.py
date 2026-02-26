"""Reduction example: Karhunen-Loeve expansion on a Gaussian covariance."""

import sys
from pathlib import Path

import numpy as np

# Allow running as a standalone script: python tutorials/examples/reduction/demo_kle.py
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
else:
    # Notebook / interactive mode: locate project root from cwd.
    cwd = Path.cwd().resolve()
    PROJECT_ROOT = next(
        (p for p in [cwd, *cwd.parents] if (p / "boed").is_dir()),
        cwd,
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boed.reduction.linear import KLE
from boed.reduction.utils import (
    gaussian_covariance,
    matern_12_covariance,
    matern_32_covariance,
    matern_52_covariance,
    spherical_covariance,
    delta_covariance
)


def main() -> None:
    kle = KLE(domain=(0.0, 1.0), n_points=100)
    kle.fit(gaussian_covariance, n_modes=12)

    samples, _ = kle.sample(n_samples=5, rank=5, random_state=0)
    captured = np.sum(kle.eigenvalues[:5]) / np.sum(kle.eigenvalues)

    print(f"KLE eigenfunctions shape: {kle.eigenfunctions.shape}")
    print(f"Generated samples shape: {samples.shape}")
    print(f"Captured energy with 12 modes: {captured:.3f}")


if __name__ == "__main__":
    main()
