"""
KLE - Karhunen-Loève Expansion
"""

import numpy as np
from typing import Optional, Callable
from .base import DimensionalityReductionBase


class KLE(DimensionalityReductionBase):
    """
    Karhunen-Loève Expansion for random field representation.

    Solves the Fredholm integral equation numerically using the Nyström method.
    """

    def __init__(self, domain=(0, 1), n_points=100, quadrature='trapezoid'):
        """
        Parameters
        ----------
        domain : tuple of float
            Spatial domain [a, b]
        n_points : int
            Number of discretization points
        quadrature : {'trapezoid', 'simpson', 'gauss-legendre'}
            Numerical integration method
        """
        super().__init__(n_components=n_points)
        self.domain = domain
        self.n_points = n_points
        self.quadrature = quadrature

        # Spatial discretization
        self.z = np.linspace(domain[0], domain[1], n_points)

        # Quadrature weights
        self.W = None
        self.W_sqrt = None

        # Results
        self.eigenvalues = None
        self.eigenfunctions = None
        self.C_matrix = None
        self.mean_func = None

    def _compute_quadrature_weights(self):
        """Compute numerical integration weights."""
        a, b = self.domain
        m = self.n_points

        if self.quadrature == 'trapezoid':
            # Trapezoidal rule
            dz = (b - a) / (m - 1)
            W = np.full(m, dz)
            W[0] = dz / 2
            W[-1] = dz / 2

        elif self.quadrature == 'simpson':
            # Simpson's rule (requires odd m)
            if m % 2 == 0:
                raise ValueError("Simpson requires odd number of points")
            dz = (b - a) / (m - 1)
            W = np.full(m, dz)
            W[1:-1:2] *= 4
            W[2:-1:2] *= 2
            W /= 3

        elif self.quadrature == 'gauss-legendre':
            # Gauss-Legendre quadrature
            W = np.full(m, (b - a) / m)

        else:
            raise ValueError(f"Unknown quadrature: {self.quadrature}")

        self.W = W
        self.W_sqrt = np.sqrt(W)
        return W

    def fit(
        self,
        covariance_func: Callable,
        mean_func: Optional[Callable] = None,
        n_modes: Optional[int] = None,
    ) -> 'KLE':
        """
        Solve the Fredholm integral equation.

        Parameters
        ----------
        covariance_func : callable
            Covariance function c(z, z')
        mean_func : callable, optional
            Mean function μ(z)

        Returns
        -------
        self
        """
        # Build covariance matrix
        print("Building covariance matrix...")
        self.C_matrix = covariance_func(self.z, self.z)

        if not np.allclose(self.C_matrix, self.C_matrix.T):
            print("⚠️  Warning: Covariance matrix is not symmetric")

        # Compute quadrature weights
        self._compute_quadrature_weights()

        # Transform to standard eigenvalue problem
        W_sqrt_diag = np.diag(self.W_sqrt)
        C_weighted = W_sqrt_diag @ self.C_matrix @ W_sqrt_diag

        # Spectral decomposition
        print("Solving eigenvalue problem...")
        eigenvalues, eigenvectors_tilde = np.linalg.eigh(C_weighted)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors_tilde = eigenvectors_tilde[:, idx]

        # Recover eigenfunctions
        W_inv_sqrt_diag = np.diag(1.0 / self.W_sqrt)
        eigenfunctions = W_inv_sqrt_diag @ eigenvectors_tilde

        # Normalize in L² norm
        norms_L2 = np.sqrt(np.diag(eigenfunctions.T @ np.diag(self.W) @ eigenfunctions))
        eigenfunctions = eigenfunctions / norms_L2

        if n_modes is not None:
            if n_modes <= 0:
                raise ValueError("n_modes must be positive")
            n_modes = min(int(n_modes), len(eigenvalues))
            self.n_components = n_modes
            eigenvalues = eigenvalues[:n_modes]
            eigenfunctions = eigenfunctions[:, :n_modes]
        else:
            self.n_components = len(eigenvalues)

        self.eigenvalues = eigenvalues
        self.eigenfunctions = eigenfunctions

        if mean_func is None:
            self.mean_func = lambda z: np.zeros_like(z)
        else:
            self.mean_func = mean_func

        print(f"✓ KLE computed: {len(eigenvalues)} modes")
        self._is_fitted = True
        return self

    def sample(self, n_samples: int = 1, rank: Optional[int] = None,
               random_state: Optional[int] = None):
        """
        Generate random field realizations.

        Parameters
        ----------
        n_samples : int
            Number of realizations to generate
        rank : int, optional
            Number of modes to use (if None, use all)
        random_state : int, optional
            Random seed

        Returns
        -------
        samples : ndarray of shape (n_samples, n_points)
            Field realizations
        coefficients : ndarray of shape (n_samples, rank)
            Random coefficients ξᵢ
        """
        self._check_fitted()

        if random_state is not None:
            np.random.seed(random_state)

        if rank is None:
            rank = len(self.eigenvalues)

        # Generate random coefficients ξᵢ ~ N(0, λᵢ)
        safe_eigenvalues = np.maximum(self.eigenvalues[:rank], 0)
        xi = np.random.randn(n_samples, rank) * np.sqrt(safe_eigenvalues)

        # Reconstruction: a = μ + Σ ξᵢ φᵢ
        mean = self.mean_func(self.z)
        samples = mean + xi @ self.eigenfunctions[:, :rank].T

        return samples, xi

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Project field samples to KLE coefficients."""
        self._check_fitted()
        mean = self.mean_func(self.z)
        centered = samples - mean
        return centered @ np.diag(self.W) @ self.eigenfunctions

    def reconstruction_error_bound(self, rank: int) -> float:
        """Theoretical L² error bound for truncation at rank."""
        if rank >= len(self.eigenvalues):
            return 0.0
        return self.eigenvalues[rank:].sum()
