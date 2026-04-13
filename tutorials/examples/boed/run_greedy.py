"""Example: Greedy EIG design matrix with run_greedy_oed.

This script:
1. Solves a 1D advection-diffusion trajectory.
2. Selects space-time sensors with greedy OED using criterion_type="EIG".
3. Saves a single annotated design matrix figure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from boed.core import make_u0
from boed.core.noise import NoiseModel
from boed.design.criteria import DesignCriteria
from boed.design.greedy import run_greedy_oed
from boed.observations.sensors import SpaceTimeSensors
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import Matern32

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1) Setup
# ---------------------------------------------------------------------------
N, dt, n_steps = 150, 0.01, 100
diffusivity, velocity = 0.01, 0.5
n_budget = 5
max_per_time = 1  # Force temporal spread: at most one sensor per time index.

x_grid = np.linspace(0.0, 1.0, N)
model = AdvectionDiffusion1D_CN(N, dt, diffusivity=diffusivity, velocity=velocity)

u0 = make_u0(
    x_grid,
    "double_gaussian",
    centers=(0.2, 0.4),
    widths=(0.07, 0.058),
    amplitudes=(1.0, 0.5),
)
trajectory = model.evolve(u0, n_steps)

kernel = Matern32(length_scale=0.05, sigma=1.0)
try:
    prior_process = GaussianProcessPrior(kernel, nx=N, mu=None)
except TypeError:
    prior_process = GaussianProcessPrior(kernel, nx=N)
Sigma_prior = prior_process.Sigma
noise = NoiseModel(sigma_noise=0.01)

candidates_x = np.linspace(10, N - 10, 25, dtype=int)
candidates_t = np.linspace(0, n_steps, 15, dtype=int)

# ---------------------------------------------------------------------------
# 2) Greedy EIG with run_greedy_oed
# ---------------------------------------------------------------------------
print("🔎 Optimisation greedy avec critère EIG (run_greedy_oed)...")
design_eig, history_eig, Sigma_post = run_greedy_oed(
    N,
    model,
    Sigma_prior,
    noise,
    candidates_x,
    candidates_t,
    n_budget,
    "EIG",
    max_per_time=max_per_time,
)

final_eig = DesignCriteria.EIG(Sigma_post, Sigma_prior)
eig_gains = np.diff(np.concatenate(([0.0], np.asarray(history_eig, dtype=float))))

print("\nDesign EIG sélectionné (ordre greedy):")
for k, ((xi, ti), gain, eig_k) in enumerate(zip(design_eig, eig_gains, history_eig), start=1):
    print(f"  #{k}: x={xi:3d}, t={ti:3d} | ΔEIG={gain:.3f} nats | EIG cumulée={eig_k:.3f} nats")
print(f"\nEIG finale (KL prior->posterior): {final_eig:.3f} nats")

# ---------------------------------------------------------------------------
# 3) Annotated design matrix output (single panel)
# ---------------------------------------------------------------------------
os.makedirs("results/results_eig_design", exist_ok=True)

unique_pts = sorted(list(set(design_eig)), key=lambda p: p[1])
sensors = SpaceTimeSensors([p[0] for p in unique_pts], [p[1] for p in unique_pts], N)

fig, ax = plt.subplots(figsize=(11, 6))

# Force a consistent orientation:
# - horizontal axis = space x
# - vertical axis   = time t
if trajectory.shape == (n_steps + 1, N):
    traj_img = trajectory
elif trajectory.shape == (N, n_steps + 1):
    traj_img = trajectory.T
else:
    raise ValueError(
        f"Unexpected trajectory shape {trajectory.shape}; expected ({n_steps + 1}, {N}) or ({N}, {n_steps + 1})."
    )

# Use index coordinates to make axis orientation unambiguous.
x_idx = np.arange(N)
t_idx = np.arange(n_steps + 1)
im = ax.pcolormesh(
    x_idx,
    t_idx,
    traj_img,
    shading="auto",
    cmap="viridis",
)
plt.colorbar(im, ax=ax, label="Amplitude u(x,t)")

# Selected points color-coded by greedy order.
ax.scatter(
    sensors.x_idx,
    np.asarray(sensors.t_idx),
    c="red",
    marker="x",
    s=140,
    linewidths=2.0,
    zorder=4,
    label="Selected sensors",
)

order_map = {pt: k + 1 for k, pt in enumerate(design_eig)}
for xi, ti in unique_pts:
    order = order_map[(xi, ti)]
    gain = eig_gains[order - 1]
    ax.annotate(
        f"#{order}\n+{gain:.2f}",
        (xi, ti),
        textcoords="offset points",
        xytext=(7, 7),
        fontsize=8,
        color="white",
        bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.65, "ec": "none"},
        zorder=5,
    )

ax.set_xlim(0, N - 1)
ax.set_ylim(0, n_steps)
ax.set_xlabel(f"Space x (index 0..{N-1})")
ax.set_ylabel(f"Time t (index 0..{n_steps})")
ax.set_title(
    f"EIG design matrix | x-axis = x, y-axis = t | Final EIG = {final_eig:.2f} nats"
)
ax.text(0.01, 0.02, "t=0", transform=ax.transAxes, color="white", fontsize=9, va="bottom")
ax.text(0.01, 0.98, "t increases upward", transform=ax.transAxes, color="white", fontsize=9, va="top")
ax.legend(frameon=True)
ax.grid(ls=":", alpha=0.25)

out_pdf = "results/results_eig_design/design_matrix_eig_annotated.pdf"
out_png = "results/results_eig_design/design_matrix_eig_annotated.png"
plt.tight_layout()
plt.savefig(out_pdf)
plt.savefig(out_png, dpi=180)
plt.close(fig)

print(f"\n✅ Figure sauvegardée: {out_pdf}")
print(f"✅ Figure sauvegardée: {out_png}")
