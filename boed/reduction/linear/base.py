"""
Base Classes for Dimensionality Reduction
==========================================

Abstract base classes that define the common interface for all
dimensionality reduction methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class DimensionalityReductionBase(ABC):
    """
    Abstract base class for dimensionality reduction methods.
    
    All reduction methods should inherit from this class and implement
    the required abstract methods.
    
    Attributes:
    -----------
    n_components : int
        Number of reduced dimensions
    is_fitted : bool
        Whether the model has been fitted to data
    
    Methods:
    --------
    fit(X, **kwargs) : Fit the reduction model
    transform(X) : Transform data to reduced space
    inverse_transform(X_reduced) : Reconstruct from reduced space
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components: Optional[int] = n_components
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Compatibility property used across merged modules."""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        self._is_fitted = bool(value)

    def _check_fitted(self) -> None:
        """Raise an explicit error when called before fit()."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. Call fit() first."
            )
    
    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'DimensionalityReductionBase':
        """
        Fit the dimensionality reduction model.
        
        Parameters:
        -----------
        X : ndarray
            Training data
        **kwargs : dict
            Method-specific parameters
        
        Returns:
        --------
        self : DimensionalityReductionBase
            Fitted model
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to reduced space.
        
        Parameters:
        -----------
        X : ndarray
            Data to transform
        
        Returns:
        --------
        X_reduced : ndarray
            Transformed data in reduced space
        """
        pass
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from reduced space (optional).
        
        Not all methods support inverse transformation.
        
        Parameters:
        -----------
        X_reduced : ndarray
            Data in reduced space
        
        Returns:
        --------
        X_reconstructed : ndarray
            Reconstructed data in original space
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )
    
    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit model and transform data in one step.
        
        Parameters:
        -----------
        X : ndarray
            Training data
        **kwargs : dict
            Method-specific parameters
        
        Returns:
        --------
        X_reduced : ndarray
            Transformed data
        """
        self.fit(X, **kwargs)
        return self.transform(X)
    
    def get_reduction_ratio(self) -> float:
        """
        Get the dimensionality reduction ratio.
        
        Returns:
        --------
        ratio : float
            Reduction ratio (original_dim / reduced_dim)
        """
        if not self._is_fitted or self.n_components is None:
            return 1.0
        
        # This is a placeholder - subclasses should override if they know the original dimension
        return 1.0
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        if self._is_fitted:
            return f"{class_name}(n_components={self.n_components}, fitted=True)"
        else:
            return f"{class_name}(fitted=False)"


class LinearReduction(DimensionalityReductionBase):
    """
    Base class for linear reduction methods (PCA, KLE, POD).
    
    Linear methods have the form: X_reduced = U^T X
    where U is a basis matrix.
    
    Additional Attributes:
    ----------------------
    basis_ : ndarray
        Basis vectors (columns)
    """
    
    def __init__(self):
        super().__init__()
        self.basis_: Optional[np.ndarray] = None
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using linear projection: X_reduced = U^T X
        
        Parameters:
        -----------
        X : ndarray, shape (n_features, n_samples) or (n_samples, n_features)
            Data to transform
        
        Returns:
        --------
        X_reduced : ndarray
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if self.basis_ is None:
            raise RuntimeError("Basis not computed.")
        
        # Handle both (n, m) and (m, n) conventions
        if X.shape[0] == self.basis_.shape[0]:
            # X is (n_features, n_samples)
            return self.basis_[:, :self.n_components].T @ X
        else:
            # X is (n_samples, n_features)
            return X @ self.basis_[:, :self.n_components]
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from reduced space: X ≈ U X_reduced
        
        Parameters:
        -----------
        X_reduced : ndarray
            Data in reduced space
        
        Returns:
        --------
        X_reconstructed : ndarray
            Reconstructed data
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if self.basis_ is None:
            raise RuntimeError("Basis not computed.")
        
        # Handle both conventions
        if X_reduced.shape[0] == self.n_components:
            # X_reduced is (n_components, n_samples)
            return self.basis_[:, :self.n_components] @ X_reduced
        else:
            # X_reduced is (n_samples, n_components)
            return X_reduced @ self.basis_[:, :self.n_components].T


class ParametricReduction(DimensionalityReductionBase):
    """
    Base class for parametric reduction methods (Reduced Basis, etc.).
    
    Parametric methods build reduced models for parameter-dependent systems.
    """
    
    def __init__(self):
        super().__init__()
        self.parameter_samples_: Optional[np.ndarray] = None
    
    @abstractmethod
    def evaluate_reduced(self, parameter: np.ndarray) -> np.ndarray:
        """
        Evaluate the reduced model for a given parameter.
        
        Parameters:
        -----------
        parameter : ndarray
            Parameter vector
        
        Returns:
        --------
        solution : ndarray
            Solution in reduced or full space
        """
        pass


class InferenceReduction(DimensionalityReductionBase):
    """
    Base class for reduction methods designed for inference (AS, LIS).
    
    These methods identify important subspaces for parameter estimation.
    
    Additional Attributes:
    ----------------------
    subspace_basis_ : ndarray
        Basis for the informative subspace
    """
    
    def __init__(self):
        super().__init__()
        self.subspace_basis_: Optional[np.ndarray] = None
    
    @abstractmethod
    def fit_from_gradients(
        self,
        samples: np.ndarray,
        gradients: np.ndarray
    ) -> 'InferenceReduction':
        """
        Fit the model using gradient information.
        
        Parameters:
        -----------
        samples : ndarray (n_samples, n_params)
            Parameter samples
        gradients : ndarray (n_samples, n_params)
            Gradients of quantity of interest
        
        Returns:
        --------
        self : InferenceReduction
            Fitted model
        """
        pass
