"""Problem definitions used in reduction demos (POD/RB-style toy PDEs)."""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


class DiffusionProblem1D:
    """
    Parametric 1D diffusion model:
    du/dt = kappa(z) d2u/dz2 on z in [0, 1] with homogeneous Dirichlet BC.
    """

    def __init__(self, n_spatial=100, T_final=1.0, n_timesteps=50):
        self.n_spatial = n_spatial
        self.T_final = T_final
        self.n_timesteps = n_timesteps

        self.z = np.linspace(0, 1, n_spatial)
        self.dz = 1.0 / (n_spatial - 1)

        self.t = np.linspace(0, T_final, n_timesteps)
        self.dt = T_final / (n_timesteps - 1)

        self.interior_idx = slice(1, n_spatial - 1)
        self.n_interior = n_spatial - 2

    def _build_mass_matrix(self):
        n = self.n_interior
        dz = self.dz
        main_diag = np.full(n, 2 * dz / 3)
        off_diag = np.full(n - 1, dz / 6)
        return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    def _build_stiffness_matrix(self, kappa_values):
        n = self.n_interior
        dz = self.dz

        kappa_left = kappa_values[:-1]
        kappa_right = kappa_values[1:]
        kappa_interfaces = 2 * kappa_left * kappa_right / (kappa_left + kappa_right + 1e-12)

        kappa_interface_left = kappa_interfaces[:n]
        kappa_interface_right = kappa_interfaces[1 : n + 1]

        main_diag = (kappa_interface_left + kappa_interface_right) / dz
        lower_diag = -kappa_interface_left[1:] / dz
        upper_diag = -kappa_interface_right[:-1] / dz

        return (
            np.diag(main_diag)
            + np.diag(lower_diag, k=-1)
            + np.diag(upper_diag, k=1)
        )

    def solve(self, kappa_func, u0_func, parameter_vector):
        """Solve for a single parameter vector; returns U with shape (nz, nt)."""
        U = np.zeros((self.n_spatial, self.n_timesteps))
        U[:, 0] = u0_func(self.z, parameter_vector)

        kappa = kappa_func(self.z, parameter_vector)
        M = self._build_mass_matrix()
        K = self._build_stiffness_matrix(kappa)

        A = M + self.dt * K
        lu_piv = lu_factor(A)

        for n in range(self.n_timesteps - 1):
            rhs = M @ U[self.interior_idx, n]
            U[self.interior_idx, n + 1] = lu_solve(lu_piv, rhs)
            U[0, n + 1] = 0.0
            U[-1, n + 1] = 0.0

        return U

    def solve_multiple(self, kappa_func, u0_func, parameter_samples):
        """
        Solve for many parameter samples.
        Returns snapshot matrix with shape (n_spatial * n_timesteps, N).
        """
        N = len(parameter_samples)
        snapshots = np.zeros((self.n_spatial * self.n_timesteps, N))

        print(f"Generating {N} snapshots...")
        for i, param in enumerate(parameter_samples):
            if (i + 1) % 10 == 0:
                print(f"  Snapshot {i+1}/{N}")
            U = self.solve(kappa_func, u0_func, param)
            snapshots[:, i] = U.flatten(order="F")

        return snapshots


def kappa_piecewise(z, parameters):
    """Piecewise-constant diffusivity with one interface."""
    kappa_left, kappa_right, z_interface = parameters
    return np.where(z < z_interface, kappa_left, kappa_right)


def kappa_smooth(z, parameters):
    """Smooth sinusoidal diffusivity."""
    kappa_mean, amplitude, frequency = parameters[:3]
    return kappa_mean * (1 + amplitude * np.sin(2 * np.pi * frequency * z))


def u0_gaussian(z, parameters):
    """Gaussian initial condition based on parameters[3:6]."""
    z_center = parameters[3]
    width = parameters[4]
    amplitude_ic = parameters[5]
    return amplitude_ic * np.exp(-0.5 * ((z - z_center) / width) ** 2)
