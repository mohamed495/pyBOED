"""
Comparaison de la qualité de reconstruction selon le temps d'observation.

Hypothèse à tester :
    Pour Burgers non-linéaire, observer tard donne-t-il plus ou moins
    d'information sur u0 qu'observer tôt ?

Setup :
    - u0 = gaussienne (onde qui se raidit)
    - On compare : t_obs in [2, 5, 10, 15, 20]
    - Même budget : H = I_N, même bruit
    - Critère : ||MAP - u0_true||, trace(Sigma_post), EIG
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
T_max = 20
sigma = 0.05

x_grid = np.linspace(0, 1, N + 2)[1:-1]

model  = BurgersNonLinear_CN(N=N, dt=dt, diffusivity=0.02)
u0_true = make_u0(x_grid, "gaussian", center=0.25, width=0.07, amplitude=1.5)
trajectory = model.evolve(u0_true, T_max)

kernel = Gaussian(length_scale=0.1, sigma=1.0)
prior  = GaussianProcessPrior(kernel, nx=N)
noise  = NoiseModel(sigma_noise=sigma)

obs_times = [2, 5, 10, 15, 20]

# ---------------------------------------------------------------------------
# Run MAP + Laplace for each observation time
# ---------------------------------------------------------------------------

results = {}

for T_obs in obs_times:
    print(f"\n--- t_obs = {T_obs} ---")

    # Noisy observation at t=T_obs
    y_obs = trajectory[T_obs] + noise.sample(N, n_samples=1).flatten()

    nl_model = NonlinearLaplaceModel(
        pde_model=model,
        H=np.eye(N),
        Sigma_obs=noise.get_covariance(N),
        mu_prior=prior.mu,
        Sigma_prior=prior.Sigma,
        obs_steps=[T_obs],
    )

    # MAP
    res = nl_model.map_estimate(y_obs, theta_init=prior.mu.copy(), T=T_obs)
    theta_MAP = res.x
    err_MAP = la.norm(theta_MAP - u0_true)

    # Laplace
    Sigma_post, _, _ = nl_model.laplace_posterior(theta_MAP, T=T_obs)
    std_post = np.sqrt(np.diag(Sigma_post))
    trace_post = np.trace(Sigma_post)

    # EIG via LinearGaussianModel (linearized around MAP)
    G = model.get_forward_operator(n_steps=T_obs, u0_ref=theta_MAP)
    A = np.eye(N) @ G   # H = I_N
    lin_model = LinearGaussianModel(
        A=A,
        Sigma_noise=noise.get_covariance(N),
        mu_prior=prior.mu,
        Sigma_prior=prior.Sigma,
    )
    eig_val = lin_model.eig_optimal(m=N)

    print(f"  ||MAP - u0_true|| = {err_MAP:.4f}")
    print(f"  trace(Sigma_post) = {trace_post:.4f}")
    print(f"  EIG               = {eig_val:.4f}")

    results[T_obs] = {
        "theta_MAP": theta_MAP,
        "std_post":  std_post,
        "err_MAP":   err_MAP,
        "trace":     trace_post,
        "eig":       eig_val,
        "y_obs":     y_obs,
    }

# ---------------------------------------------------------------------------
# Plot 1 : reconstruction quality vs observation time
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Effect of observation time on Burgers inverse problem", fontsize=12)

times  = obs_times
errors = [results[t]["err_MAP"]  for t in times]
traces = [results[t]["trace"]    for t in times]
eigs   = [results[t]["eig"]      for t in times]

ax = axes[0]
ax.plot(times, errors, 'ro-', lw=2, markersize=8)
ax.set_xlabel('Observation time $t_{obs}$')
ax.set_ylabel('$||u_0^{MAP} - u_0^{true}||_2$')
ax.set_title('Reconstruction error vs $t_{obs}$')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(times, traces, 'bs-', lw=2, markersize=8)
ax.set_xlabel('Observation time $t_{obs}$')
ax.set_ylabel('trace($\\Sigma_{post}$)')
ax.set_title('Posterior uncertainty vs $t_{obs}$')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(times, eigs, 'g^-', lw=2, markersize=8)
ax.set_xlabel('Observation time $t_{obs}$')
ax.set_ylabel('EIG')
ax.set_title('Expected Information Gain vs $t_{obs}$')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_obs_times_metrics.png', dpi=120, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Plot 2 : MAP reconstruction for each observation time
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, len(obs_times), figsize=(4*len(obs_times), 4), sharey=True)
fig.suptitle("MAP reconstruction for different observation times", fontsize=12)

for ax, T_obs in zip(axes, obs_times):
    r = results[T_obs]
    ax.plot(x_grid, u0_true,      'k-',  lw=2,   label='True $u_0$')
    ax.plot(x_grid, r["theta_MAP"],'r-',  lw=2,   label='MAP')
    ax.fill_between(x_grid,
        r["theta_MAP"] - 2*r["std_post"],
        r["theta_MAP"] + 2*r["std_post"],
        alpha=0.25, color='red', label='95% CI (Laplace)')
    ax.set_title(f't_obs={T_obs}\nL2={r["err_MAP"]:.3f}')
    ax.set_xlabel('x')
    if T_obs == obs_times[0]:
        ax.set_ylabel('$u_0(x)$')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('comparison_obs_times_reconstruction.png', dpi=120, bbox_inches='tight')
plt.show()

print("\nPlots saved.")
