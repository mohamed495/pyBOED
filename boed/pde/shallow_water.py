"""PDE solvers for parametric inverse problems.

Implements various spatial-temporal PDE models with Crank-Nicolson and
semi-implicit time-stepping schemes for use in Bayesian optimal experimental design.
"""
import numpy as np
from boed.core.base import ForwardModelBase
import numpy.linalg as la
import matplotlib.patheffects as PathEffects

# ==============================================================================
# 2. SHALLOW WATER (New)
# ==============================================================================

class ShallowWater1D_CN(ForwardModelBase):
    """
    Shallow-water equations (1D):

        ∂h/∂t + ∂(hu)/∂x = 0 (continuity)

        ∂u/∂t + u∂u/∂x + g∂h/∂x = -κu (momentum)

        where h = water height, u = velocity, g = gravity, κ = friction

        Simplified (linearized) form:

        [h] [0 D ] [h]

        [u] = [gD -κ ] [u]

        where D is the spatial derivative operator
    """
    def __init__(self, N, dt, kappa=0.1, g=9.81):
        super().__init__(N, dt)
        self.kappa = kappa  # friction
        self.g = g
        self.h = 1.0 / (N + 1)
        
        self.A_spatial = self._build_spatial_matrix()
        I = np.eye(2 * N)  # doubled state (h, u)
        self.B = I - 0.5 * dt * self.A_spatial
        self.C = I + 0.5 * dt * self.A_spatial

    def _build_spatial_matrix(self):
        N = self.N
        # Centered derivative operator D
        D = (np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)) / (2 * self.h)
        
        zeros = np.zeros((N, N))
        gD = self.g * D
        friction = -self.kappa * np.eye(N)
        
        # Block matrix: [ 0  D ]
        #               [ gD -κ]
        return np.block([[zeros, D], [gD, friction]])

    def get_transition_matrix(self):
        return np.linalg.inv(self.B) @ self.C

    def evolve(self, u0, n_steps):
        # u0 must have size 2N: [h0, v0]
        U = np.zeros((n_steps + 1, 2 * self.N))
        U[0] = u0
        M = self.get_transition_matrix()
        for n in range(n_steps):
            U[n+1] = M @ U[n]
        return U
    
    def check_stability(self):
        """Crank-Nicolson is unconditionally stable for pure diffusion."""
        pass