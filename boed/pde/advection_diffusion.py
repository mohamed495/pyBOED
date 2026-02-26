"""PDE solvers for parametric inverse problems.

Implements various spatial-temporal PDE models with Crank-Nicolson and
semi-implicit time-stepping schemes for use in Bayesian optimal experimental design.
"""
import numpy as np
from boed.core.base import ForwardModelBase
import numpy.linalg as la
import matplotlib.patheffects as PathEffects


class AdvectionDiffusion1D_CN(ForwardModelBase):
    """1D Advection-Diffusion PDE with Crank-Nicolson time stepping.
    
    Solves the parabolic PDE:
    
    .. math::
        ∂u/∂t + v·∂u/∂x = κ·∂²u/∂x²
    
    with homogeneous Dirichlet or periodic boundary conditions.
    
    The Crank-Nicolson scheme is unconditionally stable for pure diffusion
    but requires CFL condition check when advection is present.
    
    Parameters
    ----------
    N : int
        Number of spatial grid points
    dt : float
        Time step size (must be positive)
    diffusivity : float, default=0.01
        Thermal/mass diffusivity coefficient (κ ≥ 0)
    velocity : float, default=1.0
        Advection velocity (v)
    bc : {'dirichlet', 'periodic'}, default='dirichlet'
        Boundary condition type
        
    Attributes
    ----------
    diffusivity : float
        Diffusivity coefficient κ
    velocity : float
        Advection velocity v
    bc : str
        Boundary condition
    h : float
        Spatial grid spacing
    A_spatial : np.ndarray
        Spatial discretization matrix (shape: N×N)
    B : np.ndarray
        Left-hand side matrix for implicit step
    C : np.ndarray
        Right-hand side matrix for implicit step
        
    Notes
    -----
    Time stepping uses the Crank-Nicolson implicit scheme:
    
    .. math::
        (I - 0.5·dt·A)u^{n+1} = (I + 0.5·dt·A)u^n
    
    This requires solving a linear system at each time step.
    
    Examples
    --------
    >>> from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
    >>> import numpy as np
    >>> 
    >>> # Create model
    >>> N, dt = 100, 0.01
    >>> model = AdvectionDiffusion1D_CN(N, dt, diffusivity=0.01, velocity=0.5)
    >>> 
    >>> # Check stability
    >>> stable, msg = model.check_stability()
    >>> print(msg)
    >>> 
    >>> # Solve
    >>> u0 = np.exp(-100 * np.linspace(0, 1, N)**2)
    >>> solution = model.evolve(u0, n_steps=50)
    >>> print(solution.shape)  # (51, 100) - includes initial condition
    """

    def __init__(self, N, dt, diffusivity=0.01, velocity=1.0, bc="dirichlet"):
        super().__init__(N, dt)
        self.diffusivity = diffusivity
        self.velocity = velocity
        self.bc = bc
        self.h = 1.0 / (N+1)

        self.A_spatial = self._build_spatial_matrix()
        I = np.eye(N)
        self.B = I - 0.5*dt*self.A_spatial
        self.C = I + 0.5*dt*self.A_spatial

    def _build_spatial_matrix(self):
        r = self.diffusivity / (self.h**2)
        Co = self.velocity / (2*self.h)  # centered

        N = self.N
        # Sign convention to obtain a dissipative (negative) operator
        main = -2 * r * np.ones(N)
        upper = (r - Co) * np.ones(N-1)
        lower = (r + Co) * np.ones(N-1)

        A = np.diag(main) + np.diag(upper, 1) + np.diag(lower, -1)

        if self.bc == "periodic":
            A[0, -1] = r + Co
            A[-1, 0] = r - Co
        # For Dirichlet BCs, boundary entries stay zero by default
        
        return A

    def get_transition_matrix(self):
        # Compute B^{-1} C
        return np.linalg.inv(self.B) @ self.C

    # Compute U at each time step
    def evolve(self, u0, n_steps):
        U = np.zeros((n_steps+1, self.N))
        U[0] = u0
        B_inv = np.linalg.inv(self.B)
        for n in range(n_steps):
            U[n+1] = B_inv @ (self.C @ U[n])
        return U
    
    # As in evolve(), compute U at each time step
    def get_forward_operator(self, n_steps):
        """
        Return the matrix G such that u_n = G @ u0
        """
        M = self.get_transition_matrix()
        # Raise the matrix to the power n_steps
        An = np.linalg.matrix_power(M, n_steps)
        return An
    
    def check_stability(self):
        """Crank-Nicolson is unconditionally stable for pure diffusion."""
        if self.velocity != 0:
            # simple CFL check for advection
            courant = abs(self.velocity)*self.dt/self.h
            stable = courant <= 1.0
            msg = f"CFL={courant:.3f}, stable={stable}"
        else:
            stable = True
            msg = "CN scheme unconditional stable (diffusion only)"
        return stable, msg




    




