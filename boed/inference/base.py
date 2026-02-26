"""Base abstractions and helper utilities for inference models."""

from abc import ABC, abstractmethod

import numpy as np

def _compress_operator(
    H: np.ndarray,
    Sigma: np.ndarray,
    W: np.ndarray | None = None,
    U: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compress observation operator and covariance consistently.

    Base model: y = H u + eps,  eps ~ N(0, Sigma)

    U : (p, m)   -> H_eff = U.T H,    Sigma_eff = U.T Sigma U
    W : diag 0/1 -> H_eff = H[idx,:], Sigma_eff = Sigma[idx,idx]
    None         -> unchanged

    Priority: U > W > identity.
    """
    if U is not None:
        return U.T @ H, U.T @ Sigma @ U
    if W is not None:
        idx = np.where(np.diag(W) > 0)[0]
        return H[idx, :], Sigma[np.ix_(idx, idx)]
    return H, Sigma


class InverseModel(ABC):
    """Abstract base class for inverse problems  y = f(theta) + eps."""

    @abstractmethod
    def log_likelihood(self, y: np.ndarray, theta: np.ndarray, **kwargs) -> float:
        """log p(y | theta)"""
        ...

    @abstractmethod
    def log_prior(self, theta: np.ndarray) -> float:
        """log p(theta)"""
        ...

    def log_posterior(self, y: np.ndarray, theta: np.ndarray, **kwargs) -> float:
        """log p(theta | y)  proportional to  log_likelihood + log_prior"""
        return self.log_likelihood(y, theta, **kwargs) + self.log_prior(theta)

    @abstractmethod
    def sample_posterior(
        self, y: np.ndarray, n_samples: int, **kwargs
    ) -> np.ndarray:
        """Returns posterior samples of shape (n_samples, d)."""
        ...

    def qoi_samples(
        self,
        samples: np.ndarray,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply QoI map G to posterior samples.
        If G is None, returns samples as-is (QoI = identity).
        """
        if G is None:
            return samples
        return samples @ G.T  # (n_samples, q)

    @staticmethod
    def _compress(
        y: np.ndarray,
        W: np.ndarray | None,
        U: np.ndarray | None,
    ) -> np.ndarray:
        """Compress observation vector. Priority: U > W > identity."""
        if U is not None:
            return U.T @ y
        if W is not None:
            idx = np.where(np.diag(W) > 0)[0]
            return y[idx]
        return y
