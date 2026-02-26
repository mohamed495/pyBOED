"""Core base classes for BOED models.

Provides abstract interfaces for forward models, kernels, and noise models
with validation utilities.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ModelMetadata:
    """Metadata container for forward models.
    
    Parameters
    ----------
    name : str
        Name of the model
    description : str, optional
        Detailed description of the model
    parameters : dict, optional
        Additional parameters specific to the model
    spatial_dim : int, default=1
        Spatial dimensionality (1, 2, or 3)
    temporal : bool, default=True
        Whether the model is time-dependent
    """
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    spatial_dim: int = 1
    temporal: bool = True


class ForwardModelBase(ABC):
    """Abstract base class for forward PDE models.
    
    Defines the interface that all forward models must implement.
    Handles spatial discretization and time stepping.
    
    Parameters
    ----------
    N : int
        Number of spatial grid points
    dt : float
        Time step size (must be positive)
    metadata : ModelMetadata, optional
        Model metadata (auto-generated if not provided)
        
    Attributes
    ----------
    N : int
        Number of spatial grid points
    dt : float
        Time step size
    metadata : ModelMetadata
        Model metadata
        
    Examples
    --------
    Subclass implementation for advection-diffusion:
    
    >>> from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
    >>> model = AdvectionDiffusion1D_CN(N=100, dt=0.01)
    >>> M = model.get_transition_matrix()
    >>> stable, msg = model.check_stability()
    """
    
    def __init__(self, N: int, dt: float, metadata: Optional[ModelMetadata] = None):
        self.N = N
        self.dt = dt
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)
        self._validate_params(N, dt)

    @staticmethod
    def _validate_params(N: int, dt: float) -> None:
        """Validate input parameters.
        
        Parameters
        ----------
        N : int
            Number of spatial points
        dt : float
            Time step
            
        Raises
        ------
        ValueError
            If N <= 0 or dt <= 0
        """
        if N <= 0:
            raise ValueError("N must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")

    @abstractmethod
    def get_transition_matrix(self) -> np.ndarray:
        """Compute the time-stepping transition matrix.
        
        Returns
        -------
        np.ndarray
            Shape (N, N) transition matrix for one time step
        """
        ...
    
    @abstractmethod
    def check_stability(self) -> Tuple[bool, str]:
        """Check numerical stability of the time-stepping scheme.
        
        Returns
        -------
        stable : bool
            True if the scheme is stable
        message : str
            Descriptive message about stability
        """
        ...

    def get_spatial_grid(self) -> np.ndarray:
        """Get spatial grid points in [0, 1].
        
        Returns
        -------
        np.ndarray
            Shape (N,) array of interior grid points
        """
        return np.linspace(0, 1, self.N + 2)[1:-1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N}, dt={self.dt:.4e})"


class KernelBase(ABC):
    """Abstract base class for covariance kernels.
    
    Used for defining prior covariance structures in Gaussian processes.
    
    Parameters
    ----------
    **hyperparams : dict
        Hyperparameters specific to the kernel (e.g., length_scale, sigma)
        
    Attributes
    ----------
    hyperparams : dict
        Dictionary of hyperparameters
        
    Examples
    --------
    >>> from boed.priors.kernels import Gaussian, Matern32
    >>> kernel_se = Gaussian(length_scale=0.5, sigma=1.0)
    >>> kernel_m32 = Matern32(length_scale=0.5, sigma=1.0)
    """
    
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams

    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Evaluate the kernel at point pairs.
        
        Parameters
        ----------
        x1 : np.ndarray
            First set of points, shape (n,)
        x2 : np.ndarray
            Second set of points, shape (m,)
            
        Returns
        -------
        np.ndarray
            Kernel matrix K(x1, x2), shape (n, m)
        """
        ...

    def gradient(self, x1: np.ndarray, x2: np.ndarray, param: str) -> np.ndarray:
        """Compute gradient w.r.t. a hyperparameter.
        
        Parameters
        ----------
        x1 : np.ndarray
            First set of points
        x2 : np.ndarray
            Second set of points
        param : str
            Name of the hyperparameter
            
        Returns
        -------
        np.ndarray
            Gradient matrix ∂K/∂param
            
        Raises
        ------
        NotImplementedError
            If gradient not available for this kernel/parameter
        """
        raise NotImplementedError(f"Gradient for {param} not implemented")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())
        return f"{self.__class__.__name__}({params})"


class NoiseModelBase(ABC):
    """Abstract base class for noise models.
    
    Defines observation noise structure for inverse problems.
    
    Examples
    --------
    >>> from boed.core.noise import NoiseModel
    >>> noise = NoiseModel(sigma_noise=0.01)
    >>> Sigma = noise.get_covariance(size=100)
    """
    
    @abstractmethod
    def get_covariance(self, size: int) -> np.ndarray:
        """Get noise covariance matrix.
        
        Parameters
        ----------
        size : int
            Dimension of the noise
            
        Returns
        -------
        np.ndarray
            Covariance matrix, shape (size, size), must be positive definite
        """
        ...
        
    @abstractmethod
    def sample(self, size: int, n_samples: int = 1) -> np.ndarray:
        """Generate samples from the noise distribution.
        
        Parameters
        ----------
        size : int
            Dimension of each sample
        n_samples : int, default=1
            Number of samples to generate
            
        Returns
        -------
        np.ndarray
            Noise samples, shape (n_samples, size) or (size,) if n_samples=1
        """
        ...


# ===== Validation Utilities =====

def validate_positive_definite(matrix: np.ndarray, name: str = "Matrix") -> None:
    """Check if matrix is symmetric positive definite.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to validate
    name : str, optional
        Name for error messages
        
    Raises
    ------
    ValueError
        If matrix is not symmetric or has non-positive eigenvalues
    """
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")
    eigvals = np.linalg.eigvalsh(matrix)
    if np.any(eigvals <= 0):
        raise ValueError(f"{name} must be PD, min eig: {eigvals.min():.2e}")


def validate_design_indices(indices: np.ndarray, N: int, allow_duplicates: bool = False) -> None:
    """Validate sensor placement indices.
    
    Parameters
    ----------
    indices : np.ndarray
        Array of indices to validate
    N : int
        Maximum allowed index (upper bound is N-1)
    allow_duplicates : bool, default=False
        Whether to allow duplicate indices
        
    Raises
    ------
    ValueError
        If indices are invalid, out of range, or duplicates when not allowed
    """
    indices = np.asarray(indices)
    if len(indices) == 0:
        raise ValueError("Indices cannot be empty")
    if np.any(indices < 0) or np.any(indices >= N):
        raise ValueError(f"indices in [0,{N})")
    if not allow_duplicates and len(indices) != len(np.unique(indices)):
        raise ValueError("Duplicate indices not allowed")
