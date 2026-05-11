"""Example: Nonlinear PDE with advanced features.

This script demonstrates pyCBOED capabilities with nonlinear PDEs,
including Shallow Water equations and nonlinear Burgers equation.

Features:
- Multiple PDE models (Shallow Water, Burgers nonlinear)
- Advanced visualization with BOEDVisualizerPro
- Comparison with linear approximations
- Complex spatial dynamics and shock formation

Note: Nonlinear problems require careful numerical treatment and
may need problem-specific approximations for efficiency.
"""
import numpy as np
import matplotlib.pyplot as plt
from boed.priors.kernels import Gaussian, Matern12, Matern32, Matern52
from boed.priors.gp_priors import GaussianProcessPrior
from boed.core.noise import NoiseModel
from boed.core import make_u0
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.pde.shallow_water import ShallowWater1D_CN
from boed.pde.burgers import Burgers_CN

from boed.viz.boed_visualizer_pro import BOEDVisualizerPro
from boed.inference import LinearGaussianModel
from boed.observations.sensors import SpaceTimeSensors
from boed.design.greedy import run_greedy_oed

SEED = 42
np.random.seed(SEED)

# ===============================================================================
# 1. PROBLEM CONFIGURATION
# ===============================================================================
np.random.seed(42)

N        = 50      # number of interior grid points
dt       = 0.001
n_steps  = 100
sigma    = 0.1    # noise standard deviation
lambda_  = 0.
n_samples = 500

x_grid = np.linspace(0, 1, N + 2)[1:-1]

model   = AdvectionDiffusion1D_CN(N=N, dt=dt, diffusivity=0.02, velocity=0.0)
u0_true = make_u0(x_grid, "gaussian", center=0.25, width=0.07, amplitude=1.5)
noise   = NoiseModel(sigma_noise=sigma)


trajectory = model.evolve(u0_true, n_steps)

# ==============================================================================
# 2. CADRE BAYÉSIEN (PRIOR & NOISE)
# ==============================================================================
kernel = Matern32(length_scale=0.2, sigma=1.0)
prior  = GaussianProcessPrior(kernel, mu=np.zeros(N), nx=N)

y_obs = trajectory + noise.sample(N, n_samples=n_steps+1)  # shape matches trajectory

# Get the forward operator for ALL time steps (0 to n_steps)
A = model.get_forward_operator(n_steps=n_steps+1)
Sigma_noise = noise.get_covariance(N )  # Covariance for ALL observations

print(f"A shape : {A.shape}")
print(f"Sigma_noise shape : {Sigma_noise.shape}")
print(y_obs[-1,:].size)



# # 4. Inférence avec les matrices recalibrées
lgm = LinearGaussianModel(
    model=model,
    Sigma_obs=Sigma_noise,
    mu_prior=prior.mu, 
    Sigma_prior=prior.Sigma,
    n_steps=n_steps+1
)

# # Flatten y_obs to 1D: shape (101, 100) -> (10100,)
y_obs_flat = y_obs[-1,:]
mu_post, Sigma_post = lgm.posterior(y_obs_flat)

print(f"EIG = {lgm.expected_information_gain()}")
print(f"Eig : {lgm.eig_post(Sigma_post=Sigma_post, Sigma_prior=prior.Sigma)}")

# # ==============================================================================
# # 5. VISUALISATION PRO
# # ==============================================================================
# viz = BOEDVisualizerPro(output_dir="results/tutorial_results")

# # A. Plot Spatio-Temporel avec les deux designs pour comparaison
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# im = ax.imshow(trajectory.T, aspect='auto', origin='lower', extent=[0, n_steps, 0, N], cmap="magma")

# # # Tracer OED (Cercles rouges)
# # for i, (xi, ti) in enumerate(design_oed):
# #     ax.scatter(ti, xi, edgecolors='white', facecolors='red', s=100, label="OED D-Opt" if i==0 else "")


# ax.legend()
# ax.set_title("Comparaison des placements de capteurs sur Burgers NL")
# plt.savefig("results/tutorial_results/design_comparison.pdf")

# # B. Plot de la reconstruction de u0
# viz.plot_field(
#     x_grid, u0_true, mu_post, Sigma_post,
#     title="Inférence de la condition initiale (Burgers NL)",
#     filename="reconstruction_final.pdf"
# )

# print("✅ Tutoriel terminé. Résultats dans le dossier 'results/tutorial_results/'")
