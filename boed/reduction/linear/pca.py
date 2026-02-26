"""
PCA - Principal Component Analysis
"""

import numpy as np
from typing import Optional
from .base import DimensionalityReductionBase


class PCA(DimensionalityReductionBase):
    """
    Principal Component Analysis.

    Finds the directions of maximum variance in data using SVD or
    eigendecomposition of the covariance matrix.

    THEORETICAL RECALL (Chapter 2.1.1.1):
    -----------------------------------------------------
    Objective: min_{V_s} ||X - V_s V_s^T X||_F^2
            subject to constraint V_s^T V_s = I_s
    
    Solution: V_s = [v1, ..., v_s], the s dominant eigenvectors of Cov(X)
    
    KEY EQUATIONS:
    ----------------
    1. Centered: X_tilde = X - 1_n mu^T where mu = (1/n) sum x_i
    2. Covariance: C = (1/(n-1)) X_tilde^T X_tilde (variance normalization)
    3. Decomposition: C v_i = lambda_i v_i
    4. Reconstruction: X_hat = V_s V_s^T X_tilde + 1_n mu^T

    """

    def __init__(self, n_components: Optional[int] = None,
                 variance_threshold: float = 0.9):
        """
        Parameters
        ----------
        n_components : int, optional
            Number of components to keep (if None, determined automatically)
        variance_threshold : float
            Fraction of variance to capture (default: 0.99)
        """
        super().__init__(n_components=n_components)
        self.variance_threshold = variance_threshold

        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, method: str = 'svd') -> 'PCA':
        """
        Perform PCA decomposition.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d_features)
            Training data
        method : {'svd', 'eig_cov'}
            Decomposition method

        Returns
        -------
        self
        """
        n_samples, d_features = X.shape

        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        if method == 'svd':
            # ═════════════════════════════════════════════════════════════════
            # METHOD 1: SINGULAR VALUE DECOMPOSITION (RECOMMENDED)
            # ═════════════════════════════════════════════════════════════════
            #
            # SVD method (recommended for numerical stability)
            #
            # Recall : X = U S V^T  where
            # - U : (n, n) left singular vectors
            # - S : (min(n,d),) singular values
            # - V : (d, d) right singular vectors
            #
            # Link with covariance:
            # Cov(X) = (1/(n-1)) X^T X
            #        = (1/(n-1)) V S^T U^T U S V^T
            #        = (1/(n-1)) V S^2 V^T    (since U^T U = I)
            #
            # So: v_i (eigenvectors) = columns of V
            #     lambda_i (eigenvalues) = s_i^2 / (n-1)

            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt.T
            self.eigenvalues_ = (s ** 2) / (n_samples - 1)

        elif method == 'eig_cov':
            # Diagonalization of covariance matrix
            C = (X_centered.T @ X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(C)

            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            self.eigenvalues_ = eigenvalues[idx]
            self.components_ = eigenvectors[:, idx]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Variance explained by each component (eq. 2.6)
        total_variance = self.eigenvalues_.sum()
        self.explained_variance_ratio_ = self.eigenvalues_ / total_variance

        # Rank selection
        # If the number of components to keep is not specified,
        # compute it using the explained-variance threshold
        if self.n_components is None:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components = np.searchsorted(cumsum, self.variance_threshold) + 1

        # Truncate components
        self.components_ = self.components_[:, :self.n_components]
        self.eigenvalues_ = self.eigenvalues_[:self.n_components]

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to reduced space.

        EQUATION:
        ----------
        X_reduced = (X - mu) V_s
        
        INTERPRETATION:
        ----------------
        Each row of X_reduced contains the coordinates in the basis
        formed by the eigenvectors (principal components).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, n_components)
            Reduced data
        """
        self._check_fitted()
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from reduced space.

        EQUATION:
        ----------
        X_hat = X_reduced V_s^T + mu
        
        RECONSTRUCTION ERROR (eq. 2.6):
        ------------------------------------
        ||X - X_hat||_F^2 = sum_{i=r+1}^d lambda_i

        Parameters
        ----------
        X_reduced : ndarray of shape (n_samples, n_components)
            Reduced data

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data
        """
        self._check_fitted()
        return X_reduced @ self.components_.T + self.mean_
