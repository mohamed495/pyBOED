"""
Reduced Prior Design
====================

BOED with dimensionality-reduced priors using KLE or PCA.

This module enables efficient experimental design for high-dimensional
parameter spaces by:
1. Reducing the prior dimension (e.g., 10,000 → 50 parameters)
2. Running greedy design in reduced space (fast)
3. Reconstructing full-dimensional posterior if needed

Example:
--------
>>> from boed.priors import GaussianProcessPrior
>>> from boed.priors.kernels import Gaussian
>>> from boed.integration import ReducedPriorDesign
>>> 
>>> # High-dimensional prior
>>> kernel = Gaussian(length_scale=0.5, sigma=1.0)
>>> prior = GaussianProcessPrior(kernel, nx=10000)
>>> 
>>> # Reduce to 50 dimensions
>>> reduced_design = ReducedPriorDesign(
...     prior=prior,
...     n_components=50,
...     method='kle'
... )
>>> 
>>> # Run design in reduced space
>>> design, history, Sigma_post_reduced = reduced_design.run_design(
...     model=forward_model,
...     noise_model=noise,
...     candidates_x=candidates_x,
...     candidates_t=candidates_t,
...     n_budget=20,
...     criterion_type="A"
... )
>>> 
>>> # Reconstruct full posterior if needed
>>> Sigma_post_full = reduced_design.reconstruct_posterior(Sigma_post_reduced)
"""

from typing import Optional, Tuple, Literal
import numpy as np
import warnings


class ReducedPriorDesign:
    """
    BOED with dimensionality-reduced GP priors.
    
    Attributes:
    -----------
    prior_full : GaussianProcessPrior
        Original high-dimensional prior
    n_components : int
        Number of reduced dimensions
    method : {'kle', 'pca'}
        Reduction method
    reducer : KLE or PCA
        Fitted dimensionality reduction object
    Sigma_reduced : ndarray
        Reduced prior covariance matrix (n_components, n_components)
    
    Methods:
    --------
    run_design(...) : Run greedy BOED in reduced space
    reconstruct_posterior(...) : Reconstruct full-dimensional posterior
    get_speedup_estimate() : Estimate computational speedup
    """
    
    def __init__(
        self,
        prior,
        n_components: int = 50,
        method: Literal['kle', 'pca'] = 'kle',
        kle_domain: Optional[Tuple[float, float]] = None,
        pca_n_samples: int = 1000
    ):
        """
        Initialize reduced prior design.
        
        Parameters:
        -----------
        prior : GaussianProcessPrior
            High-dimensional GP prior
        n_components : int
            Target number of reduced dimensions
        method : {'kle', 'pca'}
            Reduction method:
            - 'kle': Karhunen-Loève Expansion (preferred for GPs)
            - 'pca': Principal Component Analysis
        kle_domain : tuple, optional
            Domain for KLE (default: [0, 1])
        pca_n_samples : int
            Number of samples to draw for PCA (if method='pca')
        """
        self.prior_full = prior
        self.n_components = n_components
        self.method = method
        
        # Get original dimension
        self.n_full = prior.Sigma.shape[0]
        
        if n_components > self.n_full:
            warnings.warn(
                f"n_components ({n_components}) > full dimension ({self.n_full}). "
                f"Setting n_components = {self.n_full}"
            )
            self.n_components = self.n_full
        
        # Build reducer
        if method == 'kle':
            self._build_kle_reducer(kle_domain)
        elif method == 'pca':
            self._build_pca_reducer(pca_n_samples)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'kle' or 'pca'")
        
        # Compute reduced covariance
        self.Sigma_reduced = self._compute_reduced_covariance()
        
        print(f"✓ Reduced prior built:")
        print(f"  Method: {method.upper()}")
        print(f"  Dimension: {self.n_full} → {self.n_components}")
        print(f"  Speedup estimate: ~{self.get_speedup_estimate():.0f}x")
    
    def _build_kle_reducer(self, domain: Optional[Tuple[float, float]]):
        """Build KLE reducer for GP prior"""
        from ..reduction.linear.kle import KLE
        
        if domain is None:
            domain = [0, 1]
        
        self.reducer = KLE(domain=domain, n_points=self.n_full)
        
        # Use the prior's kernel as covariance function
        def cov_func(z, zp):
            # Convert scalar positions to arrays if needed
            z_arr = np.atleast_1d(z)
            zp_arr = np.atleast_1d(zp)
            
            # Evaluate kernel across both APIs (kernel(z, z') or kernel.evaluate(dist))
            if hasattr(self.prior_full.kernel, "evaluate"):
                dist = np.abs(z_arr[:, None] - zp_arr[None, :])
                return self.prior_full.kernel.evaluate(dist)
            return self.prior_full.kernel(z_arr, zp_arr)
        
        self.reducer.fit(cov_func, n_modes=self.n_components)
    
    def _build_pca_reducer(self, n_samples: int):
        """Build PCA reducer by sampling from prior"""
        from ..reduction.linear.pca import PCA
        
        # Sample from prior
        samples = np.random.multivariate_normal(
            self.prior_full.mu,
            self.prior_full.Sigma,
            size=n_samples
        )
        
        self.reducer = PCA(n_components=self.n_components)
        self.reducer.fit(samples)
    
    def _compute_reduced_covariance(self) -> np.ndarray:
        """
        Compute covariance matrix in reduced space.
        
        Returns:
        --------
        Sigma_reduced : ndarray (n_components, n_components)
            Reduced prior covariance
        """
        if self.method == 'kle':
            # For KLE: Sigma_reduced = diag(λ_1, ..., λ_r)
            eigenvalues = self.reducer.eigenvalues[:self.n_components]
            return np.diag(eigenvalues)
        
        else:  # pca
            # For PCA: support both components shapes (n, r) and (r, n)
            components = self.reducer.components_
            if components.shape[0] == self.n_full:
                # (n, r)
                U = components[:, :self.n_components]
                Sigma_reduced = U.T @ self.prior_full.Sigma @ U
            else:
                # (r, n)
                U = components[:self.n_components]
                Sigma_reduced = U @ self.prior_full.Sigma @ U.T
            
            # Ensure symmetry
            Sigma_reduced = 0.5 * (Sigma_reduced + Sigma_reduced.T)
            
            return Sigma_reduced
    
    def run_design(
        self,
        model,
        noise_model,
        candidates_x: np.ndarray,
        candidates_t: np.ndarray,
        n_budget: int,
        criterion_type: str = "A",
        verbose: bool = True
    ) -> Tuple[list, list, np.ndarray]:
        """
        Run greedy BOED in reduced space.
        
        Parameters:
        -----------
        model : ForwardModelBase
            Forward model (e.g., AdvectionDiffusion1D_CN)
        noise_model : NoiseModel
            Observation noise model
        candidates_x : ndarray
            Spatial candidate positions
        candidates_t : ndarray
            Temporal candidate positions
        n_budget : int
            Number of sensors to select
        criterion_type : str
            Design criterion ('A', 'D', 'C', or 'EIG')
        verbose : bool
            Print progress
        
        Returns:
        --------
        design : list
            Selected sensor positions (x, t)
        history : list
            Criterion values at each iteration
        Sigma_post_reduced : ndarray
            Posterior covariance in reduced space
        """
        # Import here to avoid circular dependencies
        from ..design.greedy import run_greedy_oed
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Greedy {criterion_type}-optimal Design (Reduced Space)")
            print(f"{'='*60}")
            print(f"Prior dimension: {self.n_full} → {self.n_components}")
            print(f"Budget: {n_budget} sensors")
            print(f"Candidates: {len(candidates_x)} × {len(candidates_t)}")
            print(f"{'='*60}\n")

        # Current greedy implementation expects covariance size to match model state size.
        model_dim = getattr(model, "N", None)
        if model_dim is not None and model_dim != self.n_components:
            raise ValueError(
                "ReducedPriorDesign currently requires model.N == n_components "
                f"(got model.N={model_dim}, n_components={self.n_components}). "
                "Provide a reduced forward operator or choose a compatible dimension."
            )
        
        # Run greedy in reduced space
        design, history, Sigma_post_reduced = run_greedy_oed(
            N=self.n_components,
            model=model,
            prior_kernel=self.Sigma_reduced,
            noise_model=noise_model,
            candidates_x=candidates_x,
            candidates_t=candidates_t,
            n_budget=n_budget,
            criterion_type=criterion_type,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n✓ Design complete")
            print(f"  Selected {len(design)} sensors")
            print(f"  Final {criterion_type}-criterion: {history[-1]:.4e}")
        
        return design, history, Sigma_post_reduced
    
    def reconstruct_posterior(
        self,
        Sigma_post_reduced: np.ndarray,
        mu_post_reduced: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Reconstruct full-dimensional posterior from reduced space.
        
        Parameters:
        -----------
        Sigma_post_reduced : ndarray (n_components, n_components)
            Posterior covariance in reduced space
        mu_post_reduced : ndarray (n_components,), optional
            Posterior mean in reduced space
        
        Returns:
        --------
        Sigma_post_full : ndarray (n_full, n_full)
            Reconstructed posterior covariance
        mu_post_full : ndarray (n_full,), optional
            Reconstructed posterior mean (if mu_post_reduced provided)
        """
        if self.method == 'kle':
            # U = eigenfunctions matrix (n_full, n_components)
            U = self.reducer.eigenfunctions[:, :self.n_components]
            Sigma_post_full = U @ Sigma_post_reduced @ U.T
            
            if mu_post_reduced is not None:
                mu_post_full = U @ mu_post_reduced
            else:
                mu_post_full = None
        
        else:  # pca
            components = self.reducer.components_
            if components.shape[0] == self.n_full:
                # components = (n, r)
                U = components[:, :self.n_components]
                Sigma_post_full = U @ Sigma_post_reduced @ U.T
            else:
                # components = (r, n)
                U = components[:self.n_components]
                Sigma_post_full = U.T @ Sigma_post_reduced @ U
            
            if mu_post_reduced is not None:
                if components.shape[0] == self.n_full:
                    mu_post_full = U @ mu_post_reduced + self.reducer.mean_
                else:
                    mu_post_full = U.T @ mu_post_reduced + self.reducer.mean_
            else:
                mu_post_full = None
        
        # Ensure symmetry
        Sigma_post_full = 0.5 * (Sigma_post_full + Sigma_post_full.T)
        
        return Sigma_post_full, mu_post_full
    
    def get_speedup_estimate(self) -> float:
        """
        Estimate computational speedup.
        
        The greedy algorithm complexity is approximately O(n³) where n
        is the parameter dimension. Reducing n → r gives speedup ~ (n/r)³
        for matrix operations.
        
        Returns:
        --------
        speedup : float
            Estimated speedup factor
        """
        reduction_factor = self.n_full / self.n_components
        
        # Matrix operations scale as O(n³)
        speedup_matrix = reduction_factor ** 3
        
        # Conservative estimate (accounting for overhead)
        speedup = 0.5 * speedup_matrix
        
        return speedup
    
    def get_energy_ratio(self) -> float:
        """
        Get the energy (variance) captured by reduced basis.
        
        Returns:
        --------
        energy_ratio : float
            Fraction of total variance captured (0 to 1)
        """
        if self.method == 'kle':
            total_energy = np.sum(self.reducer.eigenvalues)
            captured_energy = np.sum(
                self.reducer.eigenvalues[:self.n_components]
            )
            return captured_energy / total_energy
        
        else:  # pca
            return np.sum(self.reducer.explained_variance_ratio_)
    
    def summary(self) -> str:
        """
        Get summary of the reduced design setup.
        
        Returns:
        --------
        summary : str
            Summary information
        """
        energy = self.get_energy_ratio()
        speedup = self.get_speedup_estimate()
        
        summary = f"""
Reduced Prior Design Summary
{'='*50}
Method:              {self.method.upper()}
Original dimension:  {self.n_full}
Reduced dimension:   {self.n_components}
Reduction ratio:     {self.n_full / self.n_components:.1f}x
Energy captured:     {energy:.2%}
Speedup estimate:    ~{speedup:.0f}x

Prior (full):        {self.prior_full.Sigma.shape}
Prior (reduced):     {self.Sigma_reduced.shape}
{'='*50}
"""
        return summary
