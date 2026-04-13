"""Analytical linear-Gaussian posterior inference model."""

import numpy as np
import numpy.linalg as la
from scipy.linalg import eigh

from .base import InverseModel
from boed.utils.observation import reduce_system, parse_theta

class LinearGaussianModel(InverseModel):
    """
    Linear Gaussian inverse model.
        A is extract from model (PDE)

        y = A @ theta + eps
        A is in model
        theta ~ N(mu_prior, Sigma_prior)
        eps   ~ N(0, Sigma_obs)

    Optional design/compression:
        y_m = W.T @ yboed/inference/linear_gaussian.py
        y_ortho = Wc.T @ y

    ``W``:
        - projection matrix of shape (n_obs, m)
    """

    def __init__(self, model, n_steps, Sigma_obs, mu_prior, Sigma_prior):
        if not hasattr(model, "get_forward_operator"):
            raise TypeError("model must implement get_forward_operator().")
        if n_steps is None:
            raise ValueError("n_steps is required when model provides get_forward_operator().")

        n_steps = int(n_steps)
        try:
            A = np.asarray(model.get_forward_operator(n_steps=n_steps), dtype=float)
        except TypeError:
            A = np.asarray(model.get_forward_operator(n_steps), dtype=float)

        if A.ndim != 2:
            raise ValueError("get_forward_operator() must return a 2D matrix.")

        n_obs, n_params = A.shape

        def _check_shape(arr, expected_shape, name):
            arr = np.asarray(arr, dtype=float)
            if arr.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}.")
            return arr

        self.model = model
        self.A = A
        self.Sigma_obs = _check_shape(Sigma_obs,   (n_obs, n_obs),         "Sigma_obs")
        self.Sigma_prior = _check_shape(Sigma_prior, (n_params, n_params), "Sigma_prior")
        self.mu_prior = _check_shape(mu_prior, (n_params,),                "mu_prior").ravel()


    def posterior(
        self,
        y: np.ndarray,
        W: np.ndarray | None = None,
        Wc: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the posterior mean and covariance given observations y."""
        sys = reduce_system(y, self.A, self.Sigma_obs, W=W, Wc=Wc)

        R_inv = la.inv(sys["R_active"]) #  R  : observation noise covariance (n_obs × n_obs)
        H = sys["H_active"] #  H  : observation matrix (from model)

        Sigma_post = la.inv(H.T @ R_inv @ H + la.inv(self.Sigma_prior))
        mu_post = Sigma_post @ (H.T @ R_inv @ sys["y_active"] + la.inv(self.Sigma_prior) @ self.mu_prior)

        return mu_post, Sigma_post


    def log_likelihood(
        self,
        y: np.ndarray,
        theta: np.ndarray,
    ) -> float:
        pass

    def log_prior(self, theta: np.ndarray) -> float:
        pass
    
    def sample_posterior(self, mu_post : np.ndarray
            , Sigma_post : np.ndarray
            , n_samples : int ) -> np.ndarray : 
        return np.random.multivariate_normal(mu_post, Sigma_post, size=int(n_samples))

    def Sigma_Y(self) -> np.ndarray:
        """Marginal covariance of Y:  Sigma_Y = Sigma_obs + A Sigma_prior A.T"""
        return self.Sigma_obs + self.A @ self.Sigma_prior @ self.A.T

    def eig(
        self,
        W: np.ndarray | None = None,
    ) -> float:
        
        if W is None : 
            S_num = self.Sigma_Y()
            S_den = self.Sigma_obs
        else : 
            S_num = W.T @ self.Sigma_Y() @ W
            S_den = W.T @ self.Sigma_obs @ W

        sign_num, logdet_num = la.slogdet(S_num)
        sign_den, logdet_den = la.slogdet(S_den)
        if sign_num <= 0 or sign_den <= 0:
            raise ValueError("Compressed covariances must be positive definite.")
        return float(0.5 * (logdet_num - logdet_den))

    def expected_information_gain(
        self,
        W: np.ndarray | None = None,
    ) -> float:
        return float(self.eig(W=W))

