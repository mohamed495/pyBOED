"""Reduction example: POD on synthetic trajectory snapshots."""

import sys
from pathlib import Path

import numpy as np

# Allow running as a standalone script: python tutorials/examples/reduction/demo_pod.py
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

from boed.reduction.linear import POD


def main() -> None:
    x = np.linspace(0.0, 1.0, 120)
    t = np.linspace(0.0, 1.0, 80)

    snapshots = []
    for ti in t:
        field = np.sin(np.pi * x) * np.exp(-ti)
        field += 0.3 * np.sin(2 * np.pi * x) * np.exp(-2.0 * ti)
        snapshots.append(field)

    S = np.array(snapshots).T  # (state_dim, n_snapshots)
    pod = POD(energy_threshold=0.999)
    pod.fit(S, use_snapshot_method=True)

    coords = pod.transform(S)
    recon = pod.inverse_transform(coords)
    rel_err = np.linalg.norm(S - recon) / np.linalg.norm(S)

    print(f"POD basis shape: {pod.basis_.shape}")
    print(f"Selected modes: {pod.n_components}")
    print(f"Relative reconstruction error: {rel_err:.3e}")


if __name__ == "__main__":
    main()
