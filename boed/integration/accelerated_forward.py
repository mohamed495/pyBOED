"""
Accelerated Forward Model
==========================

Forward model acceleration using POD (Proper Orthogonal Decomposition).

This module provides a two-phase approach:
1. **Offline phase**: Build POD basis from snapshots (expensive, done once)
2. **Online phase**: Fast evaluations in reduced space (for design optimization)

Example:
--------
>>> from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
>>> from boed.integration import PODForwardModel
>>> 
>>> # Full model
>>> full_model = AdvectionDiffusion1D_CN(N=1000, dt=0.001)
>>> 
>>> # Build reduced model (offline)
>>> pod_model = PODForwardModel(full_model, energy_threshold=0.9999)
>>> pod_model.build_reduced_model(parameter_samples, n_steps=100)
>>> 
>>> # Fast online evaluations
>>> trajectory = pod_model.evolve_reduced(u0_new, n_steps=100)
>>> 
>>> print(f"Speedup: {pod_model.get_speedup():.1f}x")
"""

from typing import List, Optional, Tuple
import numpy as np
import time
import warnings


class PODForwardModel:
    """
    POD-accelerated forward model for PDE evolution.
    
    This class wraps a full-order PDE solver and provides accelerated
    evaluations via POD reduction. The workflow is:
    
    1. Offline: Generate snapshots for representative parameters
    2. Offline: Build POD basis (one-time cost)
    3. Online: Fast evaluations for new parameters
    
    Attributes:
    -----------
    full_model : ForwardModelBase
        Original high-fidelity PDE solver
    pod : POD
        Fitted POD object
    is_fitted : bool
        Whether the reduced model has been built
    n_modes : int
        Number of POD modes retained
    
    Methods:
    --------
    build_reduced_model(...) : Offline phase - build POD basis
    evolve_reduced(...) : Online phase - fast evaluation
    get_speedup() : Estimate computational speedup
    compare_accuracy(...) : Validate reduced model accuracy
    """
    
    def __init__(
        self,
        full_model,
        energy_threshold: float = 0.9,
        max_modes: Optional[int] = None
    ):
        """
        Initialize POD forward model.
        
        Parameters:
        -----------
        full_model : ForwardModelBase
            Full-order PDE model (e.g., AdvectionDiffusion1D_CN)
        energy_threshold : float
            Energy threshold for POD truncation (default: 0.9999 = 99.99%)
        max_modes : int, optional
            Maximum number of POD modes (overrides energy_threshold)
        """
        self.full_model = full_model
        self.energy_threshold = energy_threshold
        self.max_modes = max_modes
        self.is_fitted = False
        self.pod = None
        self.n_modes = None
        
        # Timing information
        self.offline_time = None
        self.last_online_time = None
        self.last_full_time = None
    
    def build_reduced_model(
        self,
        parameter_samples: List[np.ndarray],
        n_steps: int = 100,
        verbose: bool = True
    ) -> None:
        """
        Build POD reduced model (offline phase).
        
        This generates snapshots by evolving the full model for different
        parameter values, then builds the POD basis.
        
        Parameters:
        -----------
        parameter_samples : list of ndarray
            List of initial conditions u0 or parameter vectors
        n_steps : int
            Number of time steps to evolve each sample
        verbose : bool
            Print progress information
        """
        from ..reduction.linear.pod import POD
        
        if verbose:
            print(f"\n{'='*60}")
            print("Building POD Reduced Model (Offline Phase)")
            print(f"{'='*60}")
            print(f"Full model dimension: {self.full_model.N}")
            print(f"Parameter samples: {len(parameter_samples)}")
            print(f"Time steps per sample: {n_steps}")
            print(f"Energy threshold: {self.energy_threshold:.2%}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Generate snapshots
        if verbose:
            print("Generating snapshots...")
        
        snapshots = []
        for i, u0 in enumerate(parameter_samples):
            if verbose and (i + 1) % max(1, len(parameter_samples) // 10) == 0:
                print(f"  Progress: {i+1}/{len(parameter_samples)} samples")
            
            # Evolve the full model
            trajectory = self.full_model.evolve(u0, n_steps=n_steps)
            
            # Support both conventions:
            # - (n_steps+1, N) from pyCBOED solvers
            # - (N, n_steps+1) from reduction modules
            if trajectory.shape[0] == self.full_model.N:
                for t in range(trajectory.shape[1]):
                    snapshots.append(trajectory[:, t])
            else:
                for t in range(trajectory.shape[0]):
                    snapshots.append(trajectory[t, :])
        
        snapshots = np.array(snapshots).T  # (N, total_snapshots)
        
        if verbose:
            print(f"✓ Generated {snapshots.shape[1]} snapshots")
            print(f"\nBuilding POD basis...")
        
        # Build POD basis
        self.pod = POD(energy_threshold=self.energy_threshold)
        self.pod.fit(snapshots, use_snapshot_method=True)
        
        # Apply max_modes if specified
        if self.max_modes is not None:
            self.n_modes = min(self.max_modes, self.pod.n_components)
        else:
            self.n_modes = self.pod.n_components
        
        self.is_fitted = True
        self.offline_time = time.time() - start_time
        
        if verbose:
            print(f"✓ POD basis built: {self.n_modes} modes")
            cumulative_energy = np.cumsum(self.pod.energy_ratio_)
            print(f"  Energy captured: {cumulative_energy[self.n_modes-1]:.4%}")
            print(f"  Offline time: {self.offline_time:.2f}s")
            print(f"  Speedup estimate: ~{self.get_speedup():.0f}x")
            print(f"{'='*60}\n")
    
    def evolve_reduced(
        self,
        u0: np.ndarray,
        n_steps: int = 100,
        method: str = 'project'
    ) -> np.ndarray:
        """
        Evolve system in reduced space (online phase).
        
        Parameters:
        -----------
        u0 : ndarray (N,)
            Initial condition
        n_steps : int
            Number of time steps
        method : {'project', 'galerkin'}
            Reduction method:
            - 'project': Project-evolve-project (simpler, less accurate)
            - 'galerkin': Galerkin projection of operator (more accurate)
        
        Returns:
        --------
        trajectory : ndarray (N, n_steps+1)
            Approximate solution trajectory
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Must call build_reduced_model() before evolve_reduced()"
            )
        
        start_time = time.time()
        
        if method == 'project':
            trajectory = self._evolve_project_method(u0, n_steps)
        elif method == 'galerkin':
            trajectory = self._evolve_galerkin_method(u0, n_steps)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.last_online_time = time.time() - start_time
        
        return trajectory
    
    def _evolve_project_method(
        self,
        u0: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Project-evolve-project method.
        
        Less accurate but simpler: project to reduced space, evolve in
        full space, project back at each step.
        """
        # Get POD basis
        Phi = self.pod.basis_[:, :self.n_modes]  # (N, r)
        
        # Project initial condition
        u0_reduced = Phi.T @ u0  # (r,)
        
        # Storage for reduced trajectory
        trajectory_reduced = np.zeros((self.n_modes, n_steps + 1))
        trajectory_reduced[:, 0] = u0_reduced
        
        # Evolve projected state with the full transition operator.
        M = self.full_model.get_transition_matrix()
        u_current = u0.copy()
        for t in range(n_steps):
            u_next = M @ u_current
            
            # Project to reduced space
            u_next_reduced = Phi.T @ u_next
            trajectory_reduced[:, t + 1] = u_next_reduced
            
            # Reconstruct for next step
            u_current = Phi @ u_next_reduced
        
        # Reconstruct full trajectory in solver convention: (n_steps+1, N)
        trajectory_full = Phi @ trajectory_reduced  # (N, n_steps+1)
        return trajectory_full.T
    
    def _evolve_galerkin_method(
        self,
        u0: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Galerkin projection method.
        
        More accurate: project the evolution operator itself to reduced space.
        """
        # Get POD basis
        Phi = self.pod.basis_[:, :self.n_modes]  # (N, r)
        
        # Get full transition matrix
        A_full = self.full_model.get_transition_matrix()  # (N, N)
        
        # Project to reduced space: A_reduced = Phi^T A Phi
        A_reduced = Phi.T @ A_full @ Phi  # (r, r)
        
        # Project initial condition
        u0_reduced = Phi.T @ u0  # (r,)
        
        # Storage for reduced trajectory
        trajectory_reduced = np.zeros((self.n_modes, n_steps + 1))
        trajectory_reduced[:, 0] = u0_reduced
        
        # Evolve in reduced space
        u_reduced = u0_reduced.copy()
        for t in range(n_steps):
            u_reduced = A_reduced @ u_reduced
            trajectory_reduced[:, t + 1] = u_reduced
        
        # Reconstruct full trajectory in solver convention: (n_steps+1, N)
        trajectory_full = Phi @ trajectory_reduced  # (N, n_steps+1)
        return trajectory_full.T
    
    def get_speedup(self) -> float:
        """
        Estimate computational speedup.
        
        For linear systems, the speedup is approximately (N/r)² where
        N is the full dimension and r is the number of POD modes.
        
        Returns:
        --------
        speedup : float
            Estimated speedup factor
        """
        if not self.is_fitted:
            return 1.0
        
        N = self.full_model.N
        r = self.n_modes
        
        # For matrix-vector products: O(N²) → O(r²)
        speedup_theoretical = (N / r) ** 2
        
        # Account for projection overhead
        # Projection: O(Nr) cost
        overhead_factor = 1.0 / (1.0 + 2 * r / N)
        
        speedup_effective = speedup_theoretical * overhead_factor
        
        return speedup_effective
    
    def compare_accuracy(
        self,
        u0: np.ndarray,
        n_steps: int = 100,
        method: str = 'project'
    ) -> dict:
        """
        Compare accuracy and speed of reduced vs full model.
        
        Parameters:
        -----------
        u0 : ndarray
            Initial condition
        n_steps : int
            Number of time steps
        method : str
            Reduction method
        
        Returns:
        --------
        comparison : dict
            Dictionary with keys:
            - 'rel_error': Relative L2 error
            - 'max_error': Maximum pointwise error
            - 'time_full': Full model time
            - 'time_reduced': Reduced model time
            - 'speedup': Actual speedup achieved
        """
        if not self.is_fitted:
            raise RuntimeError("Must build reduced model first")
        
        # Full model evolution
        start_time = time.time()
        trajectory_full = self.full_model.evolve(u0, n_steps=n_steps)
        time_full = time.time() - start_time
        
        # Reduced model evolution
        start_time = time.time()
        trajectory_reduced = self.evolve_reduced(u0, n_steps=n_steps, method=method)
        time_reduced = time.time() - start_time
        
        # Compute errors
        error = trajectory_full - trajectory_reduced
        rel_error = np.linalg.norm(error) / np.linalg.norm(trajectory_full)
        max_error = np.max(np.abs(error))
        
        # Actual speedup
        speedup = time_full / time_reduced
        
        return {
            'rel_error': rel_error,
            'max_error': max_error,
            'time_full': time_full,
            'time_reduced': time_reduced,
            'speedup': speedup
        }
    
    def summary(self) -> str:
        """
        Get summary of the reduced model.
        
        Returns:
        --------
        summary : str
            Summary information
        """
        if not self.is_fitted:
            return "POD model not fitted yet. Call build_reduced_model() first."
        
        energy = np.cumsum(self.pod.energy_ratio_)[self.n_modes - 1]
        speedup = self.get_speedup()
        
        summary = f"""
            POD Forward Model Summary
            {'='*50}
            Full model dimension:   {self.full_model.N}
            POD modes:              {self.n_modes}
            Reduction ratio:        {self.full_model.N / self.n_modes:.1f}x
            Energy captured:        {energy:.4%}
            Speedup estimate:       ~{speedup:.0f}x

            Offline time:           {self.offline_time:.2f}s
            {'='*50}
            """
        return summary
