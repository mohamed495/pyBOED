"""
Hybrid Bayesian Inference
==========================

Combines Linear Gaussian posterior (from BOED) with Likelihood-Informed
Subspaces (LIS) for efficient high-dimensional inference.

Example:
--------
>>> from boed.integration import HybridBayesianInference
>>> 
>>> inference = HybridBayesianInference(
...     forward_operator=H,
...     noise_covariance=R,
...     prior_mean=mu_prior,
...     prior_covariance=Sigma_prior,
...     lis_rank=10
... )
>>> 
>>> # Identify informative subspace
>>> inference.identify_informative_subspace(observations, n_samples=1000)
>>> 
>>> # Compute posterior in reduced space
>>> mu_post_reduced, Sigma_post_reduced = inference.posterior_in_subspace(observations)
>>> 
>>> # Reconstruct if needed
>>> mu_post_full = inference.reconstruct_from_subspace(mu_post_reduced)
"""

from typing import Optional, Tuple
import numpy as np
import warnings


class HybridBayesianInference:
    """
    Hybrid inference combining BOED posterior with LIS dimension reduction.
    
    This class integrates:
    1. LinearGaussianModel: Analytical posterior for linear models
    2. LIS: Identifies parameter subspace informed by likelihood
    
    The workflow is:
    1. Identify informative subspace using prior samples
    2. Compute posterior in reduced subspace (faster)
    3. Reconstruct full-dimensional posterior if needed
    
    Attributes:
    -----------
    lgm : LinearGaussianModel
        Linear Gaussian model for full posterior
    lis : LikelihoodInformedSubspaces
        LIS for subspace identification
    lis_rank : int
        Dimension of informative subspace
    is_fitted : bool
        Whether LIS has been fitted
    
    Methods:
    --------
    identify_informative_subspace(...) : Fit LIS using prior samples
    posterior_in_subspace(...) : Compute posterior in LIS subspace
    reconstruct_from_subspace(...) : Reconstruct full parameter
    """
    
    def __init__(
        self,
        forward_operator: np.ndarray,
        noise_covariance: np.ndarray,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray,
        lis_rank: int = 10
    ):
        """
        Initialize hybrid Bayesian inference.
        
        Parameters:
        -----------
        forward_operator : ndarray (m, n)
            Forward operator H mapping parameters to observations
        noise_covariance : ndarray (m, m)
            Observation noise covariance R
        prior_mean : ndarray (n,)
            Prior mean μ_prior
        prior_covariance : ndarray (n, n)
            Prior covariance Σ_prior
        lis_rank : int
            Target dimension for LIS subspace
        """
        # Import here to avoid circular dependencies
        from ..inference import LinearGaussianModel
        
        # Store full model
        self.lgm = LinearGaussianModel(
            A=forward_operator,
            Sigma_noise=noise_covariance,
            mu_prior=prior_mean,
            Sigma_prior=prior_covariance
        )
        
        self.lis_rank = lis_rank
        self.is_fitted = False
        self.lis = None
        
        # Dimensions
        self.n_params = len(prior_mean)
        self.n_obs = len(noise_covariance)
        self._log_likelihood_normalization = 0.5 * (
            float(np.linalg.slogdet(self.lgm.Sigma_noise)[1])
            + self.n_obs * np.log(2 * np.pi)
        )
        
        if lis_rank > self.n_params:
            warnings.warn(
                f"lis_rank ({lis_rank}) > parameter dimension ({self.n_params}). "
                f"Setting lis_rank = {self.n_params}"
            )
            self.lis_rank = self.n_params
    
    def identify_informative_subspace(
        self,
        observations: np.ndarray,
        n_samples: int = 1000,
        method: str = 'gradient',
        verbose: bool = True
    ) -> None:
        """
        Identify the likelihood-informed subspace using prior samples.
        
        Parameters:
        -----------
        observations : ndarray (m,)
            Observed data y
        n_samples : int
            Number of prior samples for LIS
        method : {'gradient', 'hessian'}
            LIS construction method:
            - 'gradient': Use gradient of log-likelihood
            - 'hessian': Use Hessian of log-likelihood (more expensive)
        verbose : bool
            Print progress
        """
        from ..reduction.inference.lis import LikelihoodInformedSubspaces
        
        if verbose:
            print(f"\n{'='*60}")
            print("Identifying Likelihood-Informed Subspace")
            print(f"{'='*60}")
            print(f"Parameter dimension: {self.n_params}")
            print(f"LIS rank: {self.lis_rank}")
            print(f"Prior samples: {n_samples}")
            print(f"Method: {method}")
            print(f"{'='*60}\n")
        
        # Sample from prior
        if verbose:
            print("Sampling from prior...")
        
        prior_samples = np.random.multivariate_normal(
            self.lgm.mu_prior,
            self.lgm.Sigma_prior,
            size=n_samples
        )
        
        # Build LIS
        if verbose:
            print("Building LIS basis...")
        
        self.lis = LikelihoodInformedSubspaces(n_components=self.lis_rank)
        
        if method not in {"gradient", "hessian"}:
            raise ValueError(
                f"Unknown method: {method}. Use 'gradient' or 'hessian'."
            )

        if method == "hessian":
            warnings.warn(
                "Hessian mode is not available in merged LIS yet; "
                "falling back to gradient mode."
            )

        gradients = np.array([
            self._gradient_log_likelihood(sample, observations)
            for sample in prior_samples
        ])
        self.lis.fit(gradients, prior_covariance=self.lgm.Sigma_prior)
        
        self.is_fitted = True
        
        if verbose:
            print(f"✓ LIS basis constructed: {self.lis_rank} dimensions")
            print(f"  Variance captured: {self._compute_variance_ratio():.2%}")
            print(f"{'='*60}\n")
    
    def _linear_residual(
        self,
        parameter: np.ndarray,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Compute the linear residual ``y - A @ theta``."""
        return observations - (self.lgm.A @ parameter)

    def _solve_noise_system(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``Sigma_noise x = rhs`` using the linear-Gaussian model noise covariance."""
        return np.linalg.solve(self.lgm.Sigma_noise, rhs)

    def _log_likelihood(
        self,
        parameter: np.ndarray,
        observations: np.ndarray
    ) -> float:
        """
        Compute log p(y|θ) for Gaussian likelihood.
        
        Parameters:
        -----------
        parameter : ndarray (n,)
            Parameter vector θ
        observations : ndarray (m,)
            Observations y
        
        Returns:
        --------
        log_lik : float
            Log-likelihood value
        """
        # Reuse LinearGaussianModel for the quadratic form and add the
        # Gaussian normalization term specific to this helper.
        quad_term = self.lgm.log_likelihood(observations, parameter)
        return float(quad_term - self._log_likelihood_normalization)
    
    def _gradient_log_likelihood(
        self,
        parameter: np.ndarray,
        observations: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of log-likelihood ∇_θ log p(y|θ).
        
        For linear Gaussian: ∇_θ log p(y|θ) = H^T R^{-1} (y - Hθ)
        
        Parameters:
        -----------
        parameter : ndarray (n,)
            Parameter vector θ
        observations : ndarray (m,)
            Observations y
        
        Returns:
        --------
        gradient : ndarray (n,)
            Gradient vector
        """
        residual = self._linear_residual(parameter, observations)

        # ∇_θ log p(y|θ) = H^T R^{-1} residual
        gradient = self.lgm.A.T @ self._solve_noise_system(residual)

        return gradient
    
    def posterior_in_subspace(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior in LIS subspace.
        
        Parameters:
        -----------
        observations : ndarray (m,)
            Observed data y
        
        Returns:
        --------
        mu_post_reduced : ndarray (r,)
            Posterior mean in LIS subspace
        Sigma_post_reduced : ndarray (r, r)
            Posterior covariance in LIS subspace
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Must call identify_informative_subspace() first"
            )
        
        # Compute full posterior
        mu_post, Sigma_post = self.lgm.posterior(observations)
        
        # Project to LIS subspace
        U = self._get_lis_basis()  # (n, r)
        
        mu_post_reduced = U.T @ mu_post  # (r,)
        Sigma_post_reduced = U.T @ Sigma_post @ U  # (r, r)
        
        # Ensure symmetry
        Sigma_post_reduced = 0.5 * (Sigma_post_reduced + Sigma_post_reduced.T)
        
        return mu_post_reduced, Sigma_post_reduced
    
    def reconstruct_from_subspace(
        self,
        param_reduced: np.ndarray,
        add_prior_mean: bool = True
    ) -> np.ndarray:
        """
        Reconstruct full parameter from LIS subspace.
        
        Parameters:
        -----------
        param_reduced : ndarray (r,)
            Parameter in LIS subspace
        add_prior_mean : bool
            Whether to add prior mean (default: True)
        
        Returns:
        --------
        param_full : ndarray (n,)
            Reconstructed full parameter
        """
        if not self.is_fitted:
            raise RuntimeError("LIS not fitted")
        
        # Reconstruct: θ = U * θ_reduced
        param_full = self._get_lis_basis() @ param_reduced
        
        if add_prior_mean:
            param_full += self.lgm.mu_prior
        
        return param_full
    
    def _compute_variance_ratio(self) -> float:
        """
        Compute fraction of informative LIS spectrum captured by selected rank.

        Returns:
        --------
        variance_ratio : float
            Fraction of LIS informative spectrum captured (0 to 1)
        """
        if not self.is_fitted:
            return 0.0

        # LIS eigenvalues quantify information content in each direction.
        eig = np.asarray(getattr(self.lis, "eigenvalues_", []), dtype=float)
        if eig.size == 0:
            return 0.0

        eig = np.maximum(eig, 0.0)
        total = float(np.sum(eig))
        if total <= 0.0:
            return 0.0

        return float(np.sum(eig[: self.lis_rank]) / total)
    
    def compare_posteriors(
        self,
        observations: np.ndarray
    ) -> dict:
        """
        Compare full vs subspace posterior computation.
        
        Parameters:
        -----------
        observations : ndarray (m,)
            Observations
        
        Returns:
        --------
        comparison : dict
            Dictionary with comparison metrics
        """
        if not self.is_fitted:
            raise RuntimeError("LIS not fitted")
        
        import time
        
        # Full posterior
        start = time.time()
        mu_full, Sigma_full = self.lgm.posterior(observations)
        time_full = time.time() - start
        
        # Subspace posterior
        start = time.time()
        mu_reduced, Sigma_reduced = self.posterior_in_subspace(observations)
        mu_reconstructed = self.reconstruct_from_subspace(mu_reduced)
        time_reduced = time.time() - start
        
        # Reconstruction error
        error_mean = np.linalg.norm(mu_full - mu_reconstructed) / np.linalg.norm(mu_full)
        
        # Covariance reconstruction (approximate)
        U = self._get_lis_basis()
        Sigma_reconstructed = U @ Sigma_reduced @ U.T
        error_cov = np.linalg.norm(Sigma_full - Sigma_reconstructed, 'fro') / np.linalg.norm(Sigma_full, 'fro')
        
        return {
            'mean_rel_error': error_mean,
            'cov_rel_error': error_cov,
            'time_full': time_full,
            'time_reduced': time_reduced,
            'speedup': time_full / time_reduced,
            'dimension_reduction': f"{self.n_params} → {self.lis_rank}"
        }
    
    def summary(self) -> str:
        """
        Get summary of hybrid inference setup.
        
        Returns:
        --------
        summary : str
            Summary information
        """
        if not self.is_fitted:
            status = "Not fitted yet"
            variance_info = "N/A"
        else:
            status = "Fitted"
            variance_info = f"{self._compute_variance_ratio():.2%}"
        
        summary = f"""
Hybrid Bayesian Inference Summary
{'='*50}
Parameter dimension:    {self.n_params}
Observation dimension:  {self.n_obs}
LIS rank:               {self.lis_rank}
Status:                 {status}
Variance captured:      {variance_info}

Reduction ratio:        {self.n_params / self.lis_rank:.1f}x
Expected speedup:       ~{(self.n_params / self.lis_rank)**2:.0f}x (matrix ops)
{'='*50}
"""
        return summary

    def _get_lis_basis(self) -> np.ndarray:
        """Return LIS basis across API variants."""
        if self.lis is None:
            raise RuntimeError("LIS not fitted")
        if hasattr(self.lis, "basis"):
            return self.lis.basis
        if hasattr(self.lis, "informed_directions_"):
            return self.lis.informed_directions_
        raise RuntimeError("LIS object has no recognized basis attribute")
