"""
Minimal test suite for boed package.
Tests basic functionality and imports.
"""
import pytest # pyright: ignore[reportMissingImports]
import numpy as np
from boed.core.utils import logdet, trace, validate_matrix


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_logdet(self):
        """Test log determinant calculation."""
        A = np.eye(5) * 2
        result = logdet(A)
        expected = np.log(2) * 5
        assert np.isclose(result, expected)

    def test_trace(self):
        """Test trace calculation."""
        A = np.eye(5) * 3
        result = trace(A)
        assert np.isclose(result, 15)

    def test_validate_matrix_valid(self):
        """Test matrix validation with valid input."""
        A = np.random.randn(5, 5)
        assert validate_matrix(A) is True

    def test_validate_matrix_not_ndarray(self):
        """Test matrix validation rejects non-numpy arrays."""
        with pytest.raises(TypeError):
            validate_matrix([[1, 2], [3, 4]])

    def test_validate_matrix_not_2d(self):
        """Test matrix validation rejects 1D arrays."""
        with pytest.raises(ValueError):
            validate_matrix(np.array([1, 2, 3]))

    def test_validate_matrix_with_nan(self):
        """Test matrix validation rejects NaN values."""
        A = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError):
            validate_matrix(A)

    def test_validate_matrix_with_inf(self):
        """Test matrix validation rejects Inf values."""
        A = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(ValueError):
            validate_matrix(A)


class TestPackageImports:
    """Test that package and modules import correctly."""
    
    def test_import_boed(self):
        """Test basic boed import."""
        import boed
        assert boed is not None

    def test_import_core_utils(self):
        """Test core.utils imports."""
        from boed.core import utils
        assert hasattr(utils, 'logdet')
        assert hasattr(utils, 'trace')

    def test_import_core_base(self):
        """Test core.base imports."""
        from boed.core import base
        assert base is not None

    def test_import_priors(self):
        """Test priors module imports."""
        from boed import priors
        assert priors is not None

    def test_import_design(self):
        """Test design module imports."""
        from boed import design
        assert design is not None

    def test_import_observations(self):
        """Test observations module imports."""
        from boed import observations
        assert observations is not None


class TestBasicFunctionality:
    """Test basic BOED functionality."""
    
    def test_numpy_matrix_operations(self):
        """Test numpy matrix operations work."""
        A = np.random.randn(10, 5)
        assert A.shape == (10, 5)
        
    def test_matrix_inverse(self):
        """Test matrix inversion."""
        A = np.random.randn(5, 5)
        A = A @ A.T + np.eye(5)  # Make it positive definite
        A_inv = np.linalg.inv(A)
        assert A_inv.shape == (5, 5)
        
    def test_eigen_decomposition(self):
        """Test eigenvalue decomposition."""
        A = np.random.randn(5, 5)
        A = A @ A.T  # Make it symmetric positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(A)
        assert len(eigvals) == 5
        assert eigvecs.shape == (5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
