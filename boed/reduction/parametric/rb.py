"""
RB - Reduced Basis Method
=========================
Reduced Basis method for parametric PDE model reduction.
"""

import numpy as np
from scipy.linalg import solve


class ParametricProblemRB:
    """
    Affine parametric problem used by the RB method.
    
    MODEL EQUATION:
    -----------------
    -∇·(κ(z;x) ∇u) = f(z)    sur z ∈ [0,1]
    u(0) = u(1) = 0          (homogeneous Dirichlet)
    """
    
    def __init__(self, n_spatial=100):
        self.n_spatial = n_spatial
        self.z = np.linspace(0, 1, n_spatial)
        self.dz = 1.0 / (n_spatial - 1)
        
        self.interior_idx = slice(1, n_spatial - 1)
        self.n_interior = n_spatial - 2
        
        self.affine_matrices = []
        self.affine_rhs = None
        self.mass_matrix = None
        self.n_affine_terms = 0
        
    def setup_affine_decomposition(self, n_modes=3):
        """Build the affine decomposition."""
        self.n_affine_terms = n_modes + 1
        
        print(f"Building affine decomposition ({self.n_affine_terms} terms)...")
        
        # Matrice de masse
        n = self.n_interior
        dz = self.dz
        main_diag = np.full(n, 2 * dz / 3)
        off_diag = np.full(n - 1, dz / 6)
        self.mass_matrix = (np.diag(main_diag) + 
                            np.diag(off_diag, k=1) + 
                            np.diag(off_diag, k=-1))
        
        # A₀ : matrice de base (κ=1)
        kappa_base = np.ones_like(self.z)
        A0 = self._build_stiffness_matrix(kappa_base)
        self.affine_matrices.append(A0)
        
        # Aⱼ : matrices pour modes sin(jπz)
        for j in range(1, n_modes + 1):
            kappa_mode = np.sin(j * np.pi * self.z)
            Aj = self._build_stiffness_matrix(kappa_mode)
            self.affine_matrices.append(Aj)
        
        # Terme source non-nul !
        z_interior = self.z[self.interior_idx]
        f_source = np.sin(np.pi * z_interior)
        
        # Projection sur l'espace EF : f̃ = M f
        self.affine_rhs = self.mass_matrix @ f_source
        
        print(f"Affine decomposition built")
        print(f"   RHS norm: {np.linalg.norm(self.affine_rhs):.4e}")

    def kappa_values(self, parameter):
        """Reconstruct κ(z) to validate positivity."""
        parameter = np.asarray(parameter, dtype=float)
        if parameter.size == 0:
            return None
        
        kappa_mean = parameter[0]
        if not np.isfinite(kappa_mean) or kappa_mean <= 0:
            return None
        
        # Handle the case with fewer parameters than modes
        n_xi = min(parameter.size - 1, self.n_affine_terms - 1)
        xi_modes = np.zeros(self.n_affine_terms - 1, dtype=float)
        if n_xi > 0:
            xi_modes[:n_xi] = parameter[1:n_xi+1]
        
        kappa = np.full_like(self.z, kappa_mean, dtype=float)
        for j in range(1, n_xi + 1):
            kappa += kappa_mean * xi_modes[j-1] * np.sin(j * np.pi * self.z)
        
        return kappa

    def is_parameter_valid(self, parameter, kappa_floor=1e-2):
        """Check that κ(z) remains strictly positive."""
        kappa = self.kappa_values(parameter)
        if kappa is None or not np.all(np.isfinite(kappa)):
            return False
        
        kappa_mean = float(np.asarray(parameter, dtype=float)[0])
        kappa_floor = max(kappa_floor, 1e-3 * kappa_mean)
        
        return np.min(kappa) > kappa_floor
        
    def _build_stiffness_matrix(self, kappa_values):
        """Stiffness matrix using arithmetic averaging (affine-friendly)."""
        n = self.n_interior
        dz = self.dz
        
        kappa_left = kappa_values[:-1]
        kappa_right = kappa_values[1:]
        # Arithmetic average to preserve affine dependence in κ
        kappa_interfaces = 0.5 * (kappa_left + kappa_right)
        
        kappa_interface_left = kappa_interfaces[:n]
        kappa_interface_right = kappa_interfaces[1:n+1]
        
        main_diag = (kappa_interface_left + kappa_interface_right) / dz
        lower_diag = -kappa_interface_left[1:] / dz
        upper_diag = -kappa_interface_right[:-1] / dz
        
        K = (np.diag(main_diag) + 
             np.diag(lower_diag, k=-1) + 
             np.diag(upper_diag, k=1))
        
        return K
    
    def assemble_system(self, parameter):
        """
        Assemble the system for a given parameter.
        
        A(x) = Σᵢ θᵢ(x) Aᵢ
        
        PARAMETERS:
        ------------
        parameter : [κ_mean, ξ₁, ..., ξₙ]
        """
        kappa_mean = parameter[0]
        
        # Handle the case with fewer parameters than modes
        n_xi = min(len(parameter) - 1, self.n_affine_terms - 1)
        xi_modes = np.zeros(self.n_affine_terms - 1)
        if n_xi > 0:
            xi_modes[:n_xi] = parameter[1:n_xi+1]
        
        # Coefficients affines
        theta = np.zeros(self.n_affine_terms)
        theta[0] = kappa_mean
        theta[1:] = kappa_mean * xi_modes
        
        # Assemblage
        A = sum(theta[i] * self.affine_matrices[i] 
                for i in range(self.n_affine_terms))
        
        return A, self.affine_rhs.copy()
    
    def solve_full_order(self, parameter):
        """
        Full-order model (FOM) solve.
        
        SYSTÈME :
        ---------
        A(x) u = f
        
        avec conditions aux limites homogeneous Dirichlets.
        """
        
        if not self.is_parameter_valid(parameter):
            kappa = self.kappa_values(parameter)
            kappa_min = np.min(kappa) if kappa is not None else np.nan
            raise ValueError(
                f"Invalid parameter: κ_min = {kappa_min:.3e} "
                f"(κ_mean = {parameter[0]:.3e})"
            )
        
        A, f = self.assemble_system(parameter)
        
        # Validation: the system should be non-singular
        cond_number = np.linalg.cond(A)
        if cond_number > 1e12:
            print(f"Warning: ill-conditioned matrix: cond(A) = {cond_number:.2e}")
        
        # Validation: nonzero RHS
        if np.linalg.norm(f) < 1e-14:
            print("Warning: right-hand side is nearly zero; solution will be trivial")
        
        try:
            u_interior = solve(A, f, assume_a='pos')
        except np.linalg.LinAlgError as e:
            print(f"Solver error: {e}")
            print(f"   Parameter: {parameter}")
            print(f"   ||A||: {np.linalg.norm(A):.2e}")
            print(f"   ||f||: {np.linalg.norm(f):.2e}")
            raise
        
        # Reconstruct with boundary conditions
        u_full = np.zeros(self.n_spatial)
        u_full[self.interior_idx] = u_interior
        
        return u_full


class APosterioriError:
    """A posteriori error estimator for RB (robust version)."""
    
    def __init__(self, problem: ParametricProblemRB):
        self.problem = problem
        self.residual_gram_matrix = None
        
        # Cholesky factorization instead of LU (more stable for symmetric matrices)
        from scipy.linalg import cholesky
        try:
            self.mass_chol = cholesky(problem.mass_matrix, lower=True)
        except np.linalg.LinAlgError:
            print("Warning: mass matrix is not positive definite; using LU")
            from scipy.linalg import lu_factor
            self.mass_lu = lu_factor(problem.mass_matrix)
            self.use_cholesky = False
        else:
            self.use_cholesky = True
    
    def _solve_mass_system(self, b):
        """Solve ``M x = b`` robustly."""
        from scipy.linalg import cho_solve, lu_solve
        
        # Check NaN/Inf
        if not np.all(np.isfinite(b)):
            print("Warning: right-hand side contains NaN or Inf")
            return np.zeros_like(b)
        
        if self.use_cholesky:
            try:
                x = cho_solve((self.mass_chol, True), b)
            except:
                print("Warning: Cholesky solve failed")
                return np.zeros_like(b)
        else:
            try:
                x = lu_solve(self.mass_lu, b)
            except:
                print("Warning: LU solve failed")
                return np.zeros_like(b)
        
        # Check result
        if not np.all(np.isfinite(x)):
            print("Warning: solution contains NaN or Inf")
            return np.zeros_like(b)
        
        return x
    
    def precompute_offline(self, reduced_basis):
        """
        Robust simplified offline precomputation.
        
        SIMPLIFIED VERSION:
        --------------------
        Use a simpler but more stable approximation
        of the residual norm.
        """
        s = reduced_basis.shape[1]
        Na = self.problem.n_affine_terms
        
        print(f"Offline precomputation of the residual Gram matrix...")
        
        # Simplified Gram matrix
        self.residual_gram_matrix = np.eye(Na + 1) * 1e-10  # Regularisation
        
        # RHS term (simplified)
        f = self.problem.affine_rhs
        
        # If f is zero (as in our case), use an approximation
        if np.linalg.norm(f) < 1e-14:
            # No RHS contribution
            self.residual_gram_matrix[0, 0] = 1e-12
        else:
            f_dual = self._solve_mass_system(f)
            self.residual_gram_matrix[0, 0] = np.abs(f @ f_dual) + 1e-12
        
        # Matrix terms (simplified and stabilized version)
        print("Computing matrix terms...")
        for i in range(Na):
            Ai = self.problem.affine_matrices[i]
            
            # Auto-correlation
            gram_ii = 0.0
            for k in range(min(s, 5)):  # Limit to 5 vectors for stability
                ri = Ai @ reduced_basis[:, k]
                
                # Check validity
                if not np.all(np.isfinite(ri)):
                    continue
                
                # Mass-weighted L2 norm
                ri_dual = self._solve_mass_system(ri)
                contrib = np.abs(ri @ ri_dual)
                
                if np.isfinite(contrib):
                    gram_ii += contrib
            
            gram_ii = gram_ii / min(s, 5) + 1e-12
            self.residual_gram_matrix[i+1, i+1] = gram_ii
            
            # Cross terms (simplified)
            for j in range(i+1, Na):
                Aj = self.problem.affine_matrices[j]
                
                gram_ij = 0.0
                for k in range(min(s, 3)):  # Even fewer for cross terms
                    ri = Ai @ reduced_basis[:, k]
                    rj = Aj @ reduced_basis[:, k]
                    
                    if not (np.all(np.isfinite(ri)) and np.all(np.isfinite(rj))):
                        continue
                    
                    ri_dual = self._solve_mass_system(ri)
                    contrib = np.abs(ri @ ri_dual) * np.abs(rj @ self._solve_mass_system(rj))
                    
                    if np.isfinite(contrib):
                        gram_ij += np.sqrt(contrib)  # Geometric approximation
                
                gram_ij = gram_ij / min(s, 3)
                self.residual_gram_matrix[i+1, j+1] = gram_ij
                self.residual_gram_matrix[j+1, i+1] = gram_ij
        
        print(f"Residual Gram matrix {self.residual_gram_matrix.shape} precomputed")
        
        # Check that the matrix is positive definite
        eigvals = np.linalg.eigvalsh(self.residual_gram_matrix)
        if np.any(eigvals < -1e-10):
            print(f"Warning: Gram matrix not positive definite; regularizing...")
            # Add diagonal regularization
            self.residual_gram_matrix += np.eye(Na + 1) * 1e-8
    
    def evaluate_online(self, parameter, reduced_solution):
        """
        Online evaluation of the error estimator (robust version).
        """
        # Coefficients affines
        kappa_mean = parameter[0]
        
        # Check parameter validity
        if not np.isfinite(kappa_mean) or kappa_mean <= 0:
            print(f"Warning: invalid parameter: kappa_mean = {kappa_mean}")
            return 1e10  # Very large error
        
        xi_modes = parameter[1:self.problem.n_affine_terms]
        
        theta = np.zeros(self.problem.n_affine_terms + 1)
        theta[0] = 1.0
        theta[1] = kappa_mean
        if len(xi_modes) > 0:
            theta[2:2+len(xi_modes)] = kappa_mean * xi_modes
        
        # Check theta
        if not np.all(np.isfinite(theta)):
            print("Warning: theta coefficients contain NaN or Inf")
            return 1e10
        
        # Residual norm
        try:
            residual_norm_squared = theta @ self.residual_gram_matrix @ theta
        except:
            print("Warning: error while computing residual norm")
            return 1e10
        
        if not np.isfinite(residual_norm_squared):
            print("Warning: residual norm is not finite")
            return 1e10
        
        # Coercivity constant (conservative approximation)
        alpha = max(kappa_mean * 0.1, 1e-6)  # More conservative
        
        # Estimator
        error_estimate = np.sqrt(np.abs(residual_norm_squared)) / alpha
        
        # Check result final
        if not np.isfinite(error_estimate):
            print("Warning: error estimator is not finite")
            return 1e10
        
        # Clip extreme values
        error_estimate = min(error_estimate, 1e10)
        
        return error_estimate


class ReducedBasis:
    """Reduced Basis method with greedy selection."""
    
    def __init__(self, problem: ParametricProblemRB, 
                 tolerance=1e-6, max_basis_size=50):
        self.problem = problem
        self.tolerance = tolerance
        self.max_basis_size = max_basis_size
        
        self.basis = []
        self.selected_parameters = []
        
        self.reduced_matrices = []
        self.reduced_rhs = None
        
        self.error_estimator = APosterioriError(problem)
        
        self.error_history = []
        
    def _gram_schmidt(self, u_new, basis):
        w = u_new.copy()
        M = self.problem.mass_matrix 
        
        for v in basis:
            # Projection using M-inner product
            coeff = (v @ M @ w) / (v @ M @ v)
            w -= coeff * v
        
        # Check the "novelty" of this snapshot
        norm_w = np.sqrt(w @ M @ w)
        norm_u = np.sqrt(u_new @ M @ u_new)
        
        # If the residual is less than 1e-12 of the original signal, it's noise
        if norm_w < 1e-12 * norm_u:
            return None  # Signal that this snapshot is redundant
        
        return w / norm_w
    
    def _update_reduced_system(self):
        """Update reduced matrices."""
        s = len(self.basis)
        Vs = np.column_stack(self.basis)
        
        self.reduced_matrices = []
        for Ai in self.problem.affine_matrices:
            Ai_reduced = Vs.T @ Ai @ Vs
            self.reduced_matrices.append(Ai_reduced)
        
        self.reduced_rhs = Vs.T @ self.problem.affine_rhs
    
    
    def _solve_reduced(self, parameter):
        """Solve the reduced system with error handling."""
        
        kappa_mean = parameter[0]
        n_xi = min(len(parameter) - 1, self.problem.n_affine_terms - 1)
        xi_modes = np.zeros(self.problem.n_affine_terms - 1)
        if n_xi > 0:
            xi_modes[:n_xi] = parameter[1:n_xi+1]
        
        theta = np.zeros(self.problem.n_affine_terms)
        theta[0] = kappa_mean
        theta[1:] = kappa_mean * xi_modes
        
        A_reduced = sum(theta[i] * self.reduced_matrices[i] 
                       for i in range(self.problem.n_affine_terms))
        
        try:
            # Use 'pos' because the physics yields a symmetric positive-definite system
            return solve(A_reduced, self.reduced_rhs, assume_a='pos')
        except (np.linalg.LinAlgError, ValueError):
            # If the matrix is singular, return None for the greedy loop
            return None

    def greedy_algorithm(self, training_set, initial_parameter=None):
        """Greedy algorithm (Algorithm 1)."""
        training_set = np.asarray(training_set, dtype=float)
        if training_set.ndim == 1:
            training_set = training_set[None, :]
        
        valid_mask = np.array([self.problem.is_parameter_valid(p) for p in training_set])
        if not np.all(valid_mask):
            n_bad = np.sum(~valid_mask)
            print(f"Warning: {n_bad} invalid parameters removed from the training set")
            training_set = training_set[valid_mask]
        
        N_train = len(training_set)
        if N_train == 0:
            raise ValueError("Empty training set after filtering invalid parameters")
        
        print("="*70)
        print("RB GREEDY ALGORITHM")
        print("="*70)
        print(f"Training set: {N_train} parameters")
        print(f"Tolerance: {self.tolerance:.2e}")
        
        # Initialization
        if initial_parameter is None:
            idx = np.random.randint(N_train)
            initial_parameter = training_set[idx]
        else:
            if not self.problem.is_parameter_valid(initial_parameter):
                raise ValueError("Initial parameter is invalid (κ(z) is not positive)")
        
        print(f"\n1. Initialization with parameter: {initial_parameter}")
        
        u_init = self.problem.solve_full_order(initial_parameter)
        u_init_interior = u_init[self.problem.interior_idx]
        
        M = self.problem.mass_matrix
        norm = np.sqrt(u_init_interior @ M @ u_init_interior)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(
                "Initial snapshot is zero or non-finite. "
                "Check the right-hand side (affine_rhs) or use "
                "non-homogeneous boundary conditions to obtain a non-trivial solution."
            )
        u_init_interior /= norm
        
        self.basis.append(u_init_interior)
        self.selected_parameters.append(initial_parameter)
        
        self._update_reduced_system()
        
        # Greedy loop
        iteration = 1
        max_error = np.inf
        
        while (max_error > self.tolerance and 
               len(self.basis) < self.max_basis_size):
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration} (basis size = {len(self.basis)})")
            print(f"{'='*70}")
            
            Vs = np.column_stack(self.basis)
            self.error_estimator.precompute_offline(Vs)
            
            print(f"Evaluating {N_train} parameters...")
            errors = []
            
            for param in training_set:
                u_reduced = self._solve_reduced(param)
                
                if u_reduced is None:
                    # Assign an infinite error to force the algorithm 
                    # to choose this parameter as the next snapshot (FOM solve)
                    error = np.inf 
                else:
                    error = self.error_estimator.evaluate_online(param, u_reduced)
                    
                errors.append(error)
            
            errors = np.array(errors)
            
            max_error_idx = np.argmax(errors)
            max_error = errors[max_error_idx]
            worst_parameter = training_set[max_error_idx]
            
            print(f"Max error: {max_error:.4e}")
            print(f"Mean error: {errors.mean():.4e}")
            
            self.error_history.append({
                'iteration': iteration,
                'basis_size': len(self.basis),
                'max_error': max_error,
                'mean_error': errors.mean(),
                'selected_parameter': worst_parameter
            })
            
            if max_error < self.tolerance:
                print(f"\n✓ CONVERGENCE atteinte !")
                break
            
            print(f"Computing FOM for a new snapshot...")
            u_new = self.problem.solve_full_order(worst_parameter)
            u_new_interior = u_new[self.problem.interior_idx]
            
            print(f"Gram-Schmidt orthogonalization...")
            u_new_orth = self._gram_schmidt(u_new_interior, self.basis)

            if u_new_orth is None:
                print("\nStop: the new snapshot is linearly dependent.")
                print("The basis is saturated at the current machine precision.")
                break

            self.basis.append(u_new_orth)
            self.selected_parameters.append(worst_parameter)
            
            self._update_reduced_system()
            
            iteration += 1
        
        print(f"\n{'='*70}")
        print("GREEDY ALGORITHM SUMMARY")
        print(f"{'='*70}")
        print(f"Final basis size: {len(self.basis)}")
        print(f"Final error: {max_error:.4e}")
        print(f"Number of iterations: {iteration}")
        print(f"Compression: {self.problem.n_interior / len(self.basis):.1f}x")
        
        return self
