"""Gaussian Process priors for Bayesian inverse problems.

Provides Gaussian Process prior distributions with flexible kernel choices
for parameter estimation in PDE-based inverse problems.
"""
import numpy as np
from boed.core.base import validate_positive_definite


class GaussianProcessPrior:
    """Gaussian Process prior over parameter fields.
    
    Defines a zero-mean Gaussian Process prior with a specified kernel.
    Computes and stores the prior covariance matrix with numerical
    stabilization (jitter terms) to ensure positive definiteness.
    
    Parameters
    ----------
    kernel : KernelBase
        Covariance kernel (e.g., Gaussian, Matern32)
    nx : int
        Number of spatial grid points
    domain : tuple, default=(0, 1)
        Domain interval for spatial grid
    jitter : float, default=1e-10
        Small positive value added to diagonal for numerical stability
        
    Attributes
    ----------
    kernel : KernelBase
        The covariance kernel
    nx : int
        Number of spatial grid points
    domain : tuple
        Domain interval
    jitter : float
        Regularization parameter
    x : np.ndarray
        Spatial grid points, shape (nx,)
    Sigma : np.ndarray
        Prior covariance matrix, shape (nx, nx)
    mu : np.ndarray
        Prior mean (zero), shape (nx,)
    Sigma_inv : np.ndarray
        Inverse of prior covariance, shape (nx, nx)
        
    Examples
    --------
    >>> from boed.priors.kernels import Gaussian, Matern32
    >>> from boed.priors.gp_priors import GaussianProcessPrior
    >>> 
    >>> # Using Gaussian (RBF) kernel
    >>> kernel = Gaussian(length_scale=0.2, sigma=1.0)
    >>> prior = GaussianProcessPrior(kernel, nx=100)
    >>> 
    >>> # Using Matérn kernel
    >>> kernel = Matern32(length_scale=0.5, sigma=2.0)
    >>> prior = GaussianProcessPrior(kernel, nx=200, domain=(0, 2))
    >>> 
    >>> # Access components
    >>> print(prior.Sigma.shape)  # (200, 200)
    >>> print(prior.x.shape)       # (200,)
    """
    
    def __init__(self, kernel, nx: int, domain=(0, 1), jitter=1e-10):
        """Initialize Gaussian Process prior.
        
        Parameters
        ----------
        kernel : KernelBase
            Covariance kernel instance
        nx : int
            Number of spatial discretization points
        domain : tuple, default=(0, 1)
            Spatial domain [a, b] for grid generation
        jitter : float, default=1e-10
            Initial jitter term for numerical stability
        """
        self.kernel = kernel
        self.nx = nx
        self.domain = domain
        self.jitter = jitter

        self.x = np.linspace(domain[0], domain[1], nx)
        self.Sigma = self._build_covariance()
        self.mu = np.zeros(nx)
        self.Sigma_inv = np.linalg.inv(self.Sigma)

    def _build_covariance(self) -> np.ndarray:
        """Build and regularize covariance matrix.
        
        Evaluates the kernel at all grid point pairs and adds jitter
        for numerical stability. Attempts Cholesky decomposition as a
        validation check.
        
        Returns
        -------
        np.ndarray
            Regularized covariance matrix, shape (nx, nx)
        """
        K = self.kernel(self.x, self.x)
        K += self.jitter * np.eye(self.nx)
        try:
            np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(self.nx)
        return K
