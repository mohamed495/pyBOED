"""Example: compare Greedy vs QR-pivot vs Maxvol sensor selection.

Generates a small plot of A-optimal scores versus the measurement budget.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from boed.core.noise import NoiseModel
from boed.design.selection import compare_to_greedy
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import Gaussian


SEED = 7
np.random.seed(SEED)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
N, dt, n_steps = 60, 0.02, 40
diffusivity, velocity = 0.01, 0.5
n_budget = 8

model = AdvectionDiffusion1D_CN(N, dt, diffusivity=diffusivity, velocity=velocity)
kernel = Gaussian(length_scale=0.1, sigma=1.0)
prior = GaussianProcessPrior(kernel, nx=N)
Sigma_prior = prior.Sigma
noise = NoiseModel(sigma_noise=0.02)

# Candidate grid (space/time)
candidates_x = np.linspace(0, N - 1, 20, dtype=int)
candidates_t = np.linspace(0, n_steps, 10, dtype=int)

# ==============================================================================
# 2. COMPARISON (A-OPT)
# ==============================================================================
results = compare_to_greedy(
    model=model,
    Sigma_prior=Sigma_prior,
    noise_model=noise,
    candidates_x=candidates_x,
    candidates_t=candidates_t,
    n_budget=n_budget,
    criterion_type="A",
)

print("Final A-opt scores:")
print(f"  Greedy : {results['greedy']['score']:.4e}")
print(f"  QR     : {results['qr']['score']:.4e}")
print(f"  Maxvol : {results['maxvol']['score']:.4e}")

# ==============================================================================
# 3. PLOT
# ==============================================================================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))
for label, key, style in [
    ("Greedy", "greedy", "o-"),
    ("QR pivot", "qr", "s-"),
    ("Maxvol", "maxvol", "^-"),
]:
    history = results[key]["history"]
    steps = np.arange(1, len(history) + 1)
    plt.plot(steps, history, style, label=label)

plt.xlabel("Budget (k)")
plt.ylabel("A-opt score (trace)")
plt.title("Greedy vs QR-pivot vs Maxvol")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/compare_greedy_qr_maxvol.png", dpi=150)
print("Saved plot to results/compare_greedy_qr_maxvol.png")
