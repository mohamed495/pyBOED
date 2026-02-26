"""Reduction example: PCA on synthetic low-rank data."""

import sys
from pathlib import Path

import numpy as np

# Allow running as a standalone script: python tutorials/examples/reduction/demo_pca.py
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

from boed.reduction.linear import PCA


def main() -> None:
    rng = np.random.default_rng(0)

    n_samples, n_features, rank_true = 300, 40, 4
    Z = rng.normal(size=(n_samples, rank_true))
    W = rng.normal(size=(rank_true, n_features))
    X = Z @ W + 0.05 * rng.normal(size=(n_samples, n_features))

    pca = PCA(n_components=rank_true)
    X_reduced = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_reduced)

    rel_err = np.linalg.norm(X - X_recon) / np.linalg.norm(X)
    explained = float(np.sum(pca.explained_variance_ratio_))

    print(f"PCA reduced shape: {X_reduced.shape}")
    print(f"Explained variance (stored components): {explained:.3f}")
    print(f"Relative reconstruction error: {rel_err:.3e}")


if __name__ == "__main__":
    main()
