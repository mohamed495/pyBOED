"""PDE solvers for parametric inverse problems.

Implements various spatial-temporal PDE models with Crank-Nicolson and
semi-implicit time-stepping schemes for use in Bayesian optimal experimental design.
"""
import numpy as np
import numpy.linalg as la

# ==============================================================================
# 2. NONLINEAR BURGERS MODEL (Semi-Implicit Crank-Nicolson)
# ==============================================================================

class BurgersNonLinear_CN:
    """
    1D Burgers model: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    Semi-implicit scheme (Crank-Nicolson for diffusion, explicit for advection).
    """
    def __init__(self, N, dt, diffusivity=0.01):
        self.N = N
        self.dt = dt
        self.nu = diffusivity
        self.dx = 1.0 / (N + 1)
        
        # Spatial operators
        # Centered Laplacian
        d2 = (np.diag(-2 * np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)) / (self.dx**2)
        # Centered first derivative
        self.D1 = (np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)) / (2 * self.dx)
        
        # Crank-Nicolson matrices for the diffusion part
        I = np.eye(N)
        self.B_diff = I - 0.5 * dt * self.nu * d2
        self.C_diff = I + 0.5 * dt * self.nu * d2
        self.B_inv = la.inv(self.B_diff)

    def evolve(self, u0, n_steps):
        """Solve the nonlinear update: ``u^{n+1} = B_inv @ (C_diff @ u^n - dt * u^n * (D1 @ u^n))``."""
        U = np.zeros((n_steps + 1, self.N))
        U[0] = u0
        for n in range(n_steps):
            u_n = U[n]
            # Nonlinear advective term: u * ∂u/∂x
            advection = u_n * (self.D1 @ u_n)
            U[n+1] = self.B_inv @ (self.C_diff @ u_n - self.dt * advection)
        return U

    def get_transition_matrix(self, u_ref=None):
        """
        Return the Jacobian (linearized transition matrix) around a reference state ``u_ref``.
        For Burgers, ``J(u) = B_inv @ (C_diff - dt * (diag(D1 @ u) + diag(u) @ D1))``.
        """
        if u_ref is None:
            u_ref = np.zeros(self.N) # Default linearization (reduces to the heat equation)
            
        # Linearization of ``u*ux``: ``δ(u*ux) = ux*δu + u*δux``
        # Matrix form: ``M_adv = diag(D1 @ u_ref) + diag(u_ref) @ D1``
        M_adv = np.diag(self.D1 @ u_ref) + np.diag(u_ref) @ self.D1
        
        # Linearized transition matrix
        M_lin = self.B_inv @ (self.C_diff - self.dt * M_adv)
        return M_lin

    def get_forward_operator(self, n_steps, u0_ref=None):
        """
        Compute the linear propagator ``G`` (forward operator) such that:
        δu_n = G @ δu_0
        ``G`` is the composition of successive Jacobians along the trajectory.
        """
        if u0_ref is None:
            # Without a reference state, the nonlinear term cannot be linearized
            raise ValueError("The nonlinear Burgers model requires u0_ref to compute the forward operator.")

        # 1. Build the reference trajectory (nominal path)
        traj = self.evolve(u0_ref, n_steps)
        
        # 2. Initialize G to the identity
        G = np.eye(self.N)
        
        for n in range(n_steps):
            M_n = self.get_transition_matrix(u_ref=traj[n])
            G = M_n @ G
            
            # SAFETY CHECK
            if np.isnan(G).any() or np.isinf(G).any():
                raise RuntimeError(f"Numerical instability at t={n*self.dt}. Reduce dt or increase diffusivity.")
        return G
            
