"""
POD - Proper Orthogonal Decomposition
"""

import numpy as np
from typing import Optional, Dict, Tuple
from .base import DimensionalityReductionBase


class POD(DimensionalityReductionBase):
    """
    Proper Orthogonal Decomposition using snapshot method.

    Approximates the solution manifold of parametric PDEs with a low-rank basis.
    """

    def __init__(self, n_components: Optional[int] = None,
                 energy_threshold: float = 0.9999):
        """
        Parameters
        ----------
        n_components : int, optional
            Basis rank (if None, determined automatically)
        energy_threshold : float
            Fraction of energy to capture
        """
        super().__init__(n_components=n_components)
        self.energy_threshold = energy_threshold

        self.basis_ = None
        self.eigenvalues_ = None
        self.mean_snapshot_ = None
        self.energy_ratio_ = None

    def fit(self, snapshots: np.ndarray, use_snapshot_method: bool = True) -> 'POD':
        """
        Construct the POD basis.

        Parameters
        ----------
        snapshots : ndarray of shape (m, N)
            Snapshot matrix (each column is a solution)
        use_snapshot_method : bool
            Use snapshot method if N < m

        Returns
        -------
        self
        """
        m, N = snapshots.shape

        print(f"POD: m = {m} (state dimension), N = {N} (snapshots)")

        # Center snapshots
        self.mean_snapshot_ = snapshots.mean(axis=1)
        snapshots_centered = snapshots - self.mean_snapshot_[:, np.newaxis]

        # Decomposition
        if use_snapshot_method and N < m:
            # Snapshot method (N << m)
            print("Using snapshot method (N < m)")

            # Correlation matrix
            C = (snapshots_centered.T @ snapshots_centered) / N

            # Spectral decomposition
            eigenvalues, eigenvectors_tilde = np.linalg.eigh(C)

            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors_tilde = eigenvectors_tilde[:, idx]

            # Recover POD basis
            basis = snapshots_centered @ eigenvectors_tilde

            # Normalization
            for i in range(N):
                if eigenvalues[i] > 1e-14:
                    basis[:, i] /= np.sqrt(N * eigenvalues[i])
                else:
                    basis[:, i] = 0.0

        else:
            # Standard SVD method
            print("Using standard SVD method")
            U, s, Vt = np.linalg.svd(snapshots_centered, full_matrices=False)
            basis = U
            eigenvalues = (s ** 2) / N

        # Rank selection
        total_energy = eigenvalues.sum()
        self.energy_ratio_ = eigenvalues / total_energy

        if self.n_components is None:
            cumsum = np.cumsum(self.energy_ratio_)
            self.n_components = np.searchsorted(cumsum, self.energy_threshold) + 1

            print(f"Selected rank: {self.n_components} for "
                  f"{self.energy_threshold*100:.2f}% energy")

        # Truncate
        self.basis_ = basis[:, :self.n_components]
        self.eigenvalues_ = eigenvalues[:self.n_components]

        reconstruction_error = eigenvalues[self.n_components:].sum()
        print(f"Reconstruction error (theoretical L²): {reconstruction_error:.4e}")

        self._is_fitted = True
        return self

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Project snapshots to reduced coordinates.

        Parameters
        ----------
        snapshots : ndarray of shape (m,) or (m, K)
            Snapshot(s) to project

        Returns
        -------
        coords : ndarray of shape (n_components,) or (n_components, K)
            Reduced coordinates
        """
        self._check_fitted()
        if snapshots.ndim == 1:
            snapshot_centered = snapshots - self.mean_snapshot_
            return self.basis_.T @ snapshot_centered
        else:
            snapshot_centered = snapshots - self.mean_snapshot_[:, np.newaxis]
            return self.basis_.T @ snapshot_centered

    def inverse_transform(self, coords: np.ndarray) -> np.ndarray:
        """
        Reconstruct from reduced coordinates.

        Parameters
        ----------
        coords : ndarray of shape (n_components,) or (n_components, K)
            Reduced coordinates

        Returns
        -------
        snapshots : ndarray of shape (m,) or (m, K)
            Reconstructed snapshots
        """
        self._check_fitted()
        if coords.ndim == 1:
            return self.mean_snapshot_ + self.basis_ @ coords
        else:
            return self.mean_snapshot_[:, np.newaxis] + self.basis_ @ coords

    def compute_reconstruction_error(self, snapshots: np.ndarray) -> Dict:
        """
        Compute reconstruction error statistics.

        Parameters
        ----------
        snapshots : ndarray of shape (m, K)
            Test snapshots

        Returns
        -------
        dict
            Dictionary with error statistics
        """
        self._check_fitted()
        coords = self.transform(snapshots)
        reconstructed = self.inverse_transform(coords)

        errors = np.linalg.norm(snapshots - reconstructed, axis=0)
        norms = np.linalg.norm(snapshots, axis=0)

        relative_errors = errors / (norms + 1e-14)

        return {
            'mean': relative_errors.mean(),
            'max': relative_errors.max(),
            'std': relative_errors.std(),
            'all': relative_errors
        }
