"""Example: Comparison of A-optimal vs D-optimal experimental designs.

This script demonstrates greedy design selection using two different optimality
criteria and compares their performance on a parametric inverse problem governed
by a 1D advection-diffusion PDE.

The example:
1. Sets up an advection-diffusion PDE with Gaussian Process prior
2. Selects measurements using A-optimality criterion (minimize trace)
3. Selects measurements using D-optimality criterion (minimize log-determinant)
4. Compares the two designs visually and in terms of posterior uncertainty

A-optimal designs minimize the average parameter variance, while D-optimal
designs minimize the volume of the uncertainty ellipsoid.
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from boed.priors.kernels import Gaussian, Matern32, Matern52
from boed.priors.gp_priors import GaussianProcessPrior
from boed.core.noise import NoiseModel
from boed.core import make_u0
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.viz.boed_visualizer_pro import BOEDVisualizerPro
from boed.inference import LinearGaussianModel
from boed.observations.sensors import SpaceTimeSensors
from boed.design.greedy import run_greedy_oed

SEED = 42
np.random.seed(SEED)

# ===============================================================================
# 1. CONFIGURATION
# ===============================================================================
# T (time) = n_steps * dt
N, dt, n_steps = 150, 0.01, 100
diffusivity, velocity = 0.01, 0.5
x_grid = np.linspace(0, 1, N)
n_budget = 8

model = AdvectionDiffusion1D_CN(N, dt, diffusivity=diffusivity, velocity=velocity)
u0 = make_u0(
    x_grid,
    "double_gaussian",
    centers=(0.2, 0.4),
    widths=(0.07, 0.058),
    amplitudes=(1.0, 0.5),
)
trajectory = model.evolve(u0, n_steps)

kernel = Matern52(length_scale=0.5, sigma=1.0)
prior_process = GaussianProcessPrior(kernel, nx=N)
Sigma_prior, mu_prior = prior_process.Sigma, prior_process.mu
noise = NoiseModel(sigma_noise=0.001)

# Candidats
candidates_x = np.linspace(10, N-10, 25, dtype=int)
candidates_t = np.linspace(0, n_steps, 15, dtype=int)

# Définition de la QoI pour le critère C (Moyenne sur la zone gauche [0, 0.5])
L_left = np.zeros(N)
L_left[N//4:N//2] = 1.0 / (N//2)

# ------------------- 2. Lancement des Optimisations -------------------
strategies = {}

# A. Critère A
print("🔎 Optimisation A-optimal...")
des_A, _, _ = run_greedy_oed(N, model, Sigma_prior, noise, candidates_x, candidates_t, n_budget, "A")
strategies["A-Opt"] = des_A

# B. Critère D
print("🔎 Optimisation D-optimal...")
des_D, _, _ = run_greedy_oed(N, model, Sigma_prior, noise, candidates_x, candidates_t, n_budget, "D")
strategies["D-Opt"] = des_D

# C. Critère C (Cible : Zone Gauche)
print("🔎 Optimisation C-optimal (Zone Gauche)...")
des_C, _, _ = run_greedy_oed(N, model, Sigma_prior, noise, candidates_x, candidates_t, n_budget, "C", L_qoi=L_left)
strategies["C-Opt"] = des_C

# D. Random (Baseline)
print("🎲 Génération Design Aléatoire...")
rand_idx = np.random.choice(len(candidates_x) * len(candidates_t), n_budget, replace=False)
des_rand = []
for idx in rand_idx:
    ti_idx = idx // len(candidates_x)
    xi_idx = idx % len(candidates_x)
    des_rand.append((candidates_x[xi_idx], candidates_t[ti_idx]))
strategies["Random"] = des_rand

# ------------------- 3. Reconstruction & Métriques Unifiées -------------------
results = {}

def get_posterior_and_eig(design, strategy_name):
    # Gestion des doublons
    unique_pts = sorted(list(set(design)), key=lambda x: x[1])
    sensors = SpaceTimeSensors([p[0] for p in unique_pts], [p[1] for p in unique_pts], N)
    W = sensors.observation_operator(n_steps + 1)
    
    # Opérateur Forward réduit (u0 -> y)
    M = model.get_transition_matrix()
    Traj_Op = np.vstack([np.linalg.matrix_power(M, t) for t in range(n_steps + 1)])
    A_fwd = W @ Traj_Op
    
    # Simulation mesure
    y_obs = A_fwd @ u0 + np.random.normal(0, noise.sigma, size=len(unique_pts))
    
    # Inférence
    Sigma_eps = (noise.sigma**2) * np.eye(len(unique_pts))
    lgm = LinearGaussianModel(A_fwd, Sigma_eps, mu_prior, Sigma_prior)
    mu_post, Sigma_post = lgm.posterior(y_obs)
    
    # Calcul EIG (En nats) : 0.5 * (log|Prior| - log|Post|)
    sign, logdet_prior = la.slogdet(Sigma_prior)
    sign, logdet_post = la.slogdet(Sigma_post)
    eig_val = 0.5 * (logdet_prior - logdet_post)
    
    # Calcul Erreur QoI (Zone Gauche)
    err_global = la.norm(u0 - mu_post)
    qoi_true = L_left @ u0
    qoi_est = L_left @ mu_post
    err_qoi = abs(qoi_true - qoi_est)
    
    return mu_post, Sigma_post, sensors, eig_val, err_global, err_qoi

print("\n📊 CALCUL DES RÉSULTATS...")
for name, design in strategies.items():
    mu, sig, sens, eig, err_g, err_c = get_posterior_and_eig(design, name)
    results[name] = {
        "mu": mu, "sig": sig, "sensors": sens, 
        "eig": eig, "err_global": err_g, "err_qoi": err_c
    }
    print(f" > {name}: EIG={eig:.2f} nats | Err Globale={err_g:.3f} | Err QoI={err_c:.4f}")

# ------------------- 4. Visualisation Comparée -------------------
viz = BOEDVisualizerPro(output_dir="results/results_comparison_final")

# A. Plot des Designs Spatio-Temporels
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    # Utilisation partielle du visualizer ou plot manuel rapide pour la grille
    im = ax.imshow(trajectory.T, aspect='auto', origin='lower', extent=[0, N, 0, n_steps], cmap="viridis")
    
    # Capteurs
    sx = res["sensors"].x_idx
    st = res["sensors"].t_idx
    ax.scatter(sx, st, c='r', marker='x', s=100, linewidth=2)
    ax.set_title(f"{name} (EIG: {res['eig']:.1f})")
    ax.set_xlabel("Espace (x)")
    ax.set_ylabel("Temps (t)")

plt.tight_layout()
plt.savefig("results/results_comparison_final/designs_matrix.pdf")



# B. Plot Reconstruction (Focus A vs C)
plt.figure(figsize=(12, 6))
plt.plot(x_grid, u0, 'k-', lw=2, label="Vrai u0")
plt.plot(x_grid, results["A-Opt"]["mu"], '--', color='tab:blue', label="A-Opt (Global)")
plt.plot(x_grid, results["C-Opt"]["mu"], '-.', color='tab:green', label="C-Opt (Zone Gauche)")
plt.plot(x_grid, results["Random"]["mu"], ':', color='gray', alpha=0.6, label="Random")

# Zone QoI
plt.axvspan(0.25, 0.5, color='green', alpha=0.1, label="Zone d'intérêt (QoI)")
plt.legend()
plt.title("Reconstruction : A-Optimality vs C-Optimality")
plt.savefig("results/results_comparison_final/reconstruction_comparison.pdf")

# C. Analyse Spectrale (Incertitude)
plt.figure(figsize=(10, 5))
for name, res in results.items():
    if name == "Random": continue # On allège le graph
    eigenvals = np.linalg.eigvalsh(res["sig"])
    plt.semilogy(np.sort(eigenvals)[::-1], label=f"{name}")

plt.semilogy(np.sort(np.linalg.eigvalsh(Sigma_prior))[::-1], 'k--', label="Prior", alpha=0.5)
plt.ylabel("Variance (Valeurs Propres)")
plt.xlabel("Modes")
plt.title("Réduction de l'Incertitude par Stratégie")
plt.legend()
plt.savefig("results/results_comparison_final/spectral_analysis.pdf")

print("\n✅ Analyse terminée. Ouvrez 'results/results_comparison_final/designs_matrix.pdf' pour voir les différences.")
