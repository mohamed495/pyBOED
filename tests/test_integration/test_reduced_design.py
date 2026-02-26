"""
Tests for ReducedPriorDesign integration module
"""

import pytest
import numpy as np
from boed.integration import ReducedPriorDesign
from boed.priors import GaussianProcessPrior
from boed.priors.kernels import Gaussian


class TestReducedPriorDesign:
    """Test suite for ReducedPriorDesign"""
    
    @pytest.fixture
    def prior_small(self):
        """Small GP prior for quick tests"""
        kernel = Gaussian(length_scale=0.5, sigma=1.0)
        return GaussianProcessPrior(kernel, nx=50)
    
    @pytest.fixture
    def prior_large(self):
        """Large GP prior for dimension reduction tests"""
        kernel = Gaussian(length_scale=0.1, sigma=1.0)
        return GaussianProcessPrior(kernel, nx=200)
    
    def test_initialization_kle(self, prior_small):
        """Test KLE-based initialization"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        assert reduced.n_components == 10
        assert reduced.method == 'kle'
        assert reduced.Sigma_reduced.shape == (10, 10)
        assert hasattr(reduced, 'reducer')
    
    def test_initialization_pca(self, prior_small):
        """Test PCA-based initialization"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='pca',
            pca_n_samples=100
        )
        
        assert reduced.n_components == 10
        assert reduced.method == 'pca'
        assert reduced.Sigma_reduced.shape == (10, 10)
    
    def test_n_components_clamping(self, prior_small):
        """Test that n_components is clamped to prior dimension"""
        with pytest.warns(UserWarning):
            reduced = ReducedPriorDesign(
                prior=prior_small,
                n_components=100,  # Larger than prior dimension
                method='kle'
            )
        
        assert reduced.n_components == 50  # Should be clamped to prior size
    
    def test_reduced_covariance_positive_definite(self, prior_small):
        """Test that reduced covariance is positive definite"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(reduced.Sigma_reduced)
        assert np.all(eigenvalues > 0), "Reduced covariance not positive definite"
    
    def test_reduced_covariance_symmetric(self, prior_small):
        """Test that reduced covariance is symmetric"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        # Check symmetry
        assert np.allclose(
            reduced.Sigma_reduced, 
            reduced.Sigma_reduced.T
        ), "Reduced covariance not symmetric"
    
    def test_reconstruction_dimension(self, prior_small):
        """Test that reconstruction returns correct dimension"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        # Create a reduced posterior
        Sigma_post_reduced = 0.5 * reduced.Sigma_reduced
        
        # Reconstruct
        Sigma_post_full, _ = reduced.reconstruct_posterior(Sigma_post_reduced)
        
        assert Sigma_post_full.shape == (50, 50)
    
    def test_reconstruction_preserves_symmetry(self, prior_small):
        """Test that reconstruction preserves symmetry"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        Sigma_post_reduced = 0.5 * reduced.Sigma_reduced
        Sigma_post_full, _ = reduced.reconstruct_posterior(Sigma_post_reduced)
        
        assert np.allclose(Sigma_post_full, Sigma_post_full.T)
    
    def test_speedup_estimate_reasonable(self, prior_large):
        """Test that speedup estimate is reasonable"""
        reduced = ReducedPriorDesign(
            prior=prior_large,
            n_components=20,
            method='kle'
        )
        
        speedup = reduced.get_speedup_estimate()
        
        # Speedup should be positive and greater than 1
        assert speedup > 1.0
        
        # For 200 → 20, expect speedup ~10-100x
        assert 5 < speedup < 1000
    
    def test_energy_ratio_bounded(self, prior_small):
        """Test that energy ratio is between 0 and 1"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        energy = reduced.get_energy_ratio()
        
        assert 0 <= energy <= 1.0
    
    def test_energy_ratio_increases_with_components(self, prior_small):
        """Test that more components capture more energy"""
        energy_10 = ReducedPriorDesign(
            prior=prior_small, n_components=10, method='kle'
        ).get_energy_ratio()
        
        energy_20 = ReducedPriorDesign(
            prior=prior_small, n_components=20, method='kle'
        ).get_energy_ratio()
        
        assert energy_20 >= energy_10
    
    def test_summary_method(self, prior_small):
        """Test that summary method runs without error"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        summary = reduced.summary()
        
        assert isinstance(summary, str)
        assert 'KLE' in summary
        assert '50' in summary  # Original dimension
        assert '10' in summary  # Reduced dimension
    
    def test_kle_vs_pca_consistency(self, prior_small):
        """Test that KLE and PCA give similar results for GP priors"""
        reduced_kle = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        reduced_pca = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='pca',
            pca_n_samples=500
        )
        
        # Both should capture similar amounts of energy
        energy_kle = reduced_kle.get_energy_ratio()
        energy_pca = reduced_pca.get_energy_ratio()
        
        # Should be within 10% of each other
        assert abs(energy_kle - energy_pca) < 0.1
    
    def test_invalid_method_raises(self, prior_small):
        """Test that invalid method raises ValueError"""
        with pytest.raises(ValueError, match="Unknown method"):
            ReducedPriorDesign(
                prior=prior_small,
                n_components=10,
                method='invalid_method'
            )
    
    def test_reconstruction_with_mean(self, prior_small):
        """Test reconstruction with both mean and covariance"""
        reduced = ReducedPriorDesign(
            prior=prior_small,
            n_components=10,
            method='kle'
        )
        
        mu_reduced = np.random.randn(10)
        Sigma_reduced = 0.5 * reduced.Sigma_reduced
        
        Sigma_full, mu_full = reduced.reconstruct_posterior(
            Sigma_reduced, mu_reduced
        )
        
        assert Sigma_full.shape == (50, 50)
        assert mu_full.shape == (50,)


class TestReducedPriorDesignPerformance:
    """Performance and scaling tests"""
    
    @pytest.mark.slow
    def test_large_scale_kle(self):
        """Test KLE on large problem"""
        kernel = Gaussian(length_scale=0.05, sigma=1.0)
        prior = GaussianProcessPrior(kernel, nx=1000)
        
        reduced = ReducedPriorDesign(
            prior=prior,
            n_components=50,
            method='kle'
        )
        
        assert reduced.n_components == 50
        assert reduced.get_energy_ratio() > 0.9
    
    @pytest.mark.slow
    def test_reconstruction_accuracy(self):
        """Test accuracy of reconstruction"""
        kernel = Gaussian(length_scale=0.1, sigma=1.0)
        prior = GaussianProcessPrior(kernel, nx=100)
        
        reduced = ReducedPriorDesign(
            prior=prior,
            n_components=30,
            method='kle'
        )
        
        # Create a test covariance in reduced space
        Sigma_test = reduced.Sigma_reduced.copy()
        
        # Reconstruct
        Sigma_recon, _ = reduced.reconstruct_posterior(Sigma_test)
        
        # Project back to reduced space
        if reduced.method == 'kle':
            U = reduced.reducer.eigenfunctions[:, :reduced.n_components]
            W = np.diag(reduced.reducer.W)
            Sigma_double_reduced = U.T @ W @ Sigma_recon @ W @ U
        else:
            U = reduced.reducer.components_[:reduced.n_components].T
            Sigma_double_reduced = U.T @ Sigma_recon @ U
        
        # Should be close to original
        rel_error = np.linalg.norm(Sigma_test - Sigma_double_reduced, 'fro') / \
                    np.linalg.norm(Sigma_test, 'fro')
        
        assert rel_error < 1e-10, f"Reconstruction error too large: {rel_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
