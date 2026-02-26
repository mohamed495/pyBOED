# """
# Test complet de NonlinearLaplaceModel et LinearGaussianModel.

# Setup :
#     - BurgersNonLinear_CN comme modèle PDE
#     - Condition initiale u0 = sin(pi x) (vérité terrain)
#     - Prior gaussien GP (lissage)
#     - Observations bruitées à t=5, 10, 20
#     - Test : MAP, Laplace, pCN, comparaison linéaire vs non-linéaire
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import numpy.linalg as la
# from boed.priors.kernels import Gaussian, Matern12, Matern32, Matern52
# from boed.priors.gp_priors import GaussianProcessPrior
# from boed.core.noise import NoiseModel
# from boed.core import make_u0
# from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
# from boed.pde.burgers import BurgersNonLinear_CN

# from boed.viz.boed_visualizer_pro import BOEDVisualizerPro
# from boed.inference import LinearGaussianModel, NonlinearLaplaceModel
# from boed.observations.sensors import SpaceTimeSensors
# from boed.design.greedy import run_greedy_oed


# # ---------------------------------------------------------------------------
# # Mini BurgersNonLinear_CN (standalone, pas besoin d'importer boed)
# # ---------------------------------------------------------------------------

# # ---------------------------------------------------------------------------
# # Setup
# # ---------------------------------------------------------------------------

# np.random.seed(42)

# N      = 100       # spatial points (small for fast test)
# dt     = 0.001
# T      = 15       # total time steps
# sigma  = 0.05     # observation noise std
# n_obs  = 8        # number of observed spatial points

# x_grid = np.linspace(0, 1, N + 2)[1:-1]  # interior points
# # PDE model
# model = BurgersNonLinear_CN(N=N, dt=dt, diffusivity=0.02)
# # True initial condition
# # Condition initiale : Une onde qui va se raidir
# u0_true = make_u0(
#     x_grid,
#     "gaussian",
#     center=0.25,
#     width=0.07,
#     amplitude=1.5,
# )
# trajectory = model.evolve(u0_true, T)

# # ==============================================================================
# # 2. CADRE BAYÉSIEN (PRIOR & NOISE)
# # ==============================================================================
# kernel = Gaussian(length_scale=0.5, sigma=1.0)
# prior = GaussianProcessPrior(kernel, nx=N)

# noise = NoiseModel(sigma_noise=0.000001)

# y_obs = trajectory + noise.sample(N, n_samples=T+1) 
# y_obs_final = y_obs[1]

# # ---------------------------------------------------------------------------
# # 1. NonlinearLaplaceModel
# # ---------------------------------------------------------------------------

# print("=" * 60)
# print("1. NonlinearLaplaceModel")
# print("=" * 60)

# nl_model = NonlinearLaplaceModel(
#     pde_model=model,
#     H=np.eye(N=N),
#     Sigma_obs=noise.get_covariance(N),
#     mu_prior=prior.mu,
#     Sigma_prior=prior.Sigma
# )

# # MAP
# print("Running MAP...")
# res = nl_model.map_estimate(y_obs_final, theta_init=prior.mu.copy(), T=T)
# theta_MAP = res.x
# print(f"  MAP converged: {res.success}  |  iterations: {res.nit}")
# print(f"  ||u0_MAP - u0_true|| = {la.norm(theta_MAP - u0_true):.4f}")
# print(f"  ||u0_prior - u0_true|| = {la.norm(prior.mu - u0_true):.4f}")

# # Laplace
# print("\nRunning Laplace posterior...")
# Sigma_post_NL, Precision_NL, J = nl_model.laplace_posterior(theta_MAP, T=T)
# std_NL = np.sqrt(np.diag(Sigma_post_NL))
# print(f"  Sigma_post diagonal mean : {std_NL.mean():.4f}")
# print(f"  Sigma_prior diagonal mean: {np.sqrt(np.diag(prior.Sigma)).mean():.4f}")
# print(f"  => Uncertainty reduced by {100*(1 - std_NL.mean()/np.sqrt(np.diag(prior.Sigma)).mean()):.1f}%")

# # D-opt score
# d_score = nl_model.d_opt_score(theta_MAP, T=T)
# print(f"\n  D-opt score = {d_score:.4f}  (higher = more informative design)")

# # pCN sampling
# print("\nRunning pCN sampler (1000 samples, 200 burn-in)...")
# samples = nl_model.sample_posterior(
#     y_obs_final, n_samples=1000, T=T,
#     theta_init=theta_MAP,
#     step_size=0.05,
#     n_burnin=200,
# )
# print(f"  Acceptance rate: {nl_model.acceptance_rate_:.3f}  (target ~0.234)")
# mu_pCN = samples.mean(axis=0)
# print(f"  ||u0_pCN - u0_true|| = {la.norm(mu_pCN - u0_true):.4f}")

# # ---------------------------------------------------------------------------
# # 4. Plot
# # ---------------------------------------------------------------------------

# # pCN posterior statistics
# mu_pCN  = samples.mean(axis=0)
# std_pCN = samples.std(axis=0)

# # Laplace posterior mean = MAP (by definition of Laplace approximation)
# mu_laplace = theta_MAP
# std_laplace = std_NL

# fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# # --- Panel 1: Bayesian reconstruction (pCN) ---
# ax = axes[0]
# ax.plot(x_grid, u0_true, 'k-', lw=2, label='True $u_0$')
# ax.plot(x_grid, mu_pCN,  'b-', lw=2, label='Posterior mean (pCN)')
# ax.fill_between(x_grid,
#     mu_pCN - 2*std_pCN,
#     mu_pCN + 2*std_pCN,
#     alpha=0.25, color='blue', label='95% credible interval (pCN)')
# # ax.scatter(x[sensor_idx], np.zeros(m) - 0.15,
# #            c='r', s=40, zorder=5, marker='|', label='Sensor locations')
# ax.set_title('Bayesian reconstruction (pCN)')
# ax.set_xlabel('x')
# ax.set_ylabel('$u_0(x)$')
# ax.legend(fontsize=8)

# # --- Panel 2: Laplace approximation ---
# ax = axes[1]
# ax.plot(x_grid, u0_true,    'k-',  lw=2,   label='True $u_0$')
# ax.plot(x_grid, mu_laplace, 'r-',  lw=2,   label='Posterior mean (Laplace = MAP)')
# ax.fill_between(x_grid,
#     mu_laplace - 2*std_laplace,
#     mu_laplace + 2*std_laplace,
#     alpha=0.25, color='red', label='95% credible interval (Laplace)')
# # ax.scatter(x[sensor_idx], np.zeros(m) - 0.15,
# #            c='r', s=40, zorder=5, marker='|', label='Sensor locations')
# ax.set_title('Bayesian reconstruction (Laplace)')
# ax.set_xlabel('x')
# ax.set_ylabel('$u_0(x)$')
# ax.legend(fontsize=8)

# # --- Panel 3: Reconstruction error ---
# ax = axes[2]
# err_prior   = np.abs(prior.mu - u0_true)
# err_MAP     = np.abs(theta_MAP - u0_true)
# err_pCN     = np.abs(mu_pCN - u0_true)
# ax.plot(x_grid, err_prior, 'g-',  lw=2, label=f'Prior (L2={la.norm(prior.mu - u0_true):.3f})')
# ax.plot(x_grid, err_MAP,   'r--', lw=2, label=f'MAP   (L2={la.norm(theta_MAP - u0_true):.3f})')
# ax.plot(x_grid, err_pCN,   'b-',  lw=2, label=f'pCN   (L2={la.norm(mu_pCN - u0_true):.3f})')
# ax.set_title('Pointwise reconstruction error $|u_0^{est} - u_0^{true}|$')
# ax.set_xlabel('x')
# ax.set_ylabel('absolute error')
# ax.legend(fontsize=8)

# plt.tight_layout()
# plt.savefig('test_inverse_models.png', dpi=120, bbox_inches='tight')
# plt.show()
# print("\nPlot saved: test_inverse_models.png")


"""
Test de NonlinearLaplaceModel — observation en un seul temps.

Setup :
    - BurgersNonLinear_CN
    - u0 = gaussienne centrée en 0.25
    - On observe u(t=T) avec H = I_N (tous les capteurs)
    - On reconstruit u0 par MAP + Laplace + pCN
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from boed.priors.kernels import Gaussian
from boed.priors.gp_priors import GaussianProcessPrior
from boed.core.noise import NoiseModel
from boed.core import make_u0
from boed.pde.burgers import BurgersNonLinear_CN
from boed.inference import LinearGaussianModel, NonlinearLaplaceModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

np.random.seed(42)

N     = 100
dt    = 0.001
T     = 15       # on observe seulement à t=T
sigma = 0.05

x_grid = np.linspace(0, 1, N + 2)[1:-1]

# PDE model
model = BurgersNonLinear_CN(N=N, dt=dt, diffusivity=0.02)

# True initial condition
u0_true = make_u0(x_grid, "gaussian", center=0.25, width=0.07, amplitude=1.5)

# Trajectory + noisy observation at t=T only
trajectory = model.evolve(u0_true, T)
noise = NoiseModel(sigma_noise=sigma)
y_obs = trajectory[T] + noise.sample(N, n_samples=1).flatten()  # shape (N,)

print(f"Setup: N={N}, T={T}, sigma={sigma}")
print(f"y_obs shape: {y_obs.shape}  (single observation at t={T})\n")

# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

kernel = Gaussian(length_scale=0.1, sigma=1.0)
prior  = GaussianProcessPrior(kernel, nx=N)

# ---------------------------------------------------------------------------
# NonlinearLaplaceModel — obs at t=T, H = I_N
# ---------------------------------------------------------------------------

nl_model = NonlinearLaplaceModel(
    pde_model=model,
    H=np.eye(N),                    # observe full state
    Sigma_obs=noise.get_covariance(N),
    mu_prior=prior.mu,
    Sigma_prior=prior.Sigma,
    obs_steps=[T],                  # single observation time
)

# --- MAP ---
print("Running MAP...")
res = nl_model.map_estimate(y_obs, theta_init=prior.mu.copy(), T=T)
theta_MAP = res.x
print(f"  Converged : {res.success}  |  iterations : {res.nit}")
print(f"  ||prior - u0_true|| = {la.norm(prior.mu - u0_true):.4f}")
print(f"  ||MAP   - u0_true|| = {la.norm(theta_MAP - u0_true):.4f}")

# --- Laplace ---
print("\nRunning Laplace posterior...")
Sigma_post, _, _ = nl_model.laplace_posterior(theta_MAP, T=T)
std_laplace = np.sqrt(np.diag(Sigma_post))
reduction = 100 * (1 - std_laplace.mean() / np.sqrt(np.diag(prior.Sigma)).mean())
print(f"  Uncertainty reduced by {reduction:.1f}%")

# --- pCN ---
print("\nRunning pCN sampler...")
samples = nl_model.sample_posterior(
    y_obs, n_samples=1000, T=T,
    theta_init=theta_MAP,
    step_size=0.3,
    n_burnin=500,
)
print(f"  Acceptance rate : {nl_model.acceptance_rate_:.3f}  (target ~0.234)")
mu_pCN  = samples.mean(axis=0)
std_pCN = samples.std(axis=0)
print(f"  ||pCN  - u0_true|| = {la.norm(mu_pCN - u0_true):.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"Burgers inverse problem — single observation at t={T}", fontsize=12)

# Panel 1 : pCN reconstruction
ax = axes[0]
ax.plot(x_grid, u0_true, 'k-', lw=2, label='True $u_0$')
ax.plot(x_grid, mu_pCN,  'b-', lw=2, label='Posterior mean (pCN)')
ax.fill_between(x_grid,
    mu_pCN - 2*std_pCN, mu_pCN + 2*std_pCN,
    alpha=0.25, color='blue', label='95% credible interval')
ax.set_title('pCN reconstruction')
ax.set_xlabel('x')
ax.set_ylabel('$u_0(x)$')
ax.legend(fontsize=8)

# Panel 2 : Laplace reconstruction
ax = axes[1]
ax.plot(x_grid, u0_true,   'k-', lw=2, label='True $u_0$')
ax.plot(x_grid, theta_MAP, 'r-', lw=2, label='Posterior mean (MAP = Laplace)')
ax.fill_between(x_grid,
    theta_MAP - 2*std_laplace, theta_MAP + 2*std_laplace,
    alpha=0.25, color='red', label='95% credible interval')
ax.set_title('Laplace reconstruction')
ax.set_xlabel('x')
ax.set_ylabel('$u_0(x)$')
ax.legend(fontsize=8)

# Panel 3 : reconstruction error
ax = axes[2]
ax.plot(x_grid, np.abs(prior.mu  - u0_true), 'g-',  lw=2,
        label=f'Prior (L2={la.norm(prior.mu - u0_true):.3f})')
ax.plot(x_grid, np.abs(theta_MAP - u0_true), 'r--', lw=2,
        label=f'MAP   (L2={la.norm(theta_MAP - u0_true):.3f})')
ax.plot(x_grid, np.abs(mu_pCN   - u0_true), 'b-',  lw=2,
        label=f'pCN   (L2={la.norm(mu_pCN - u0_true):.3f})')
ax.set_title('Pointwise error $|u_0^{est} - u_0^{true}|$')
ax.set_xlabel('x')
ax.set_ylabel('absolute error')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('test_single_obs.png', dpi=120, bbox_inches='tight')
plt.show()
print("\nPlot saved: test_single_obs.png")