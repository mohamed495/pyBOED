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
from boed.pde.burgers import BurgersNonLinear_CN

from boed.viz.boed_visualizer_pro import BOEDVisualizerPro
from boed.inference import LinearGaussianModel
from boed.observations.sensors import SpaceTimeSensors
from boed.design.greedy import run_greedy_oed

SEED = 42
np.random.seed(SEED)

# ===============================================================================
# 1. PROBLEM CONFIGURATION
# ===============================================================================
N = 150
dt = 0.01
n_steps = 1000
x_grid = np.linspace(0, 1, N)

# Choix du modèle : Burgers Non-Linéaire pour voir la formation de choc
model = BurgersNonLinear_CN(N, dt=0.001, diffusivity=0.005)

# Condition initiale : Une onde qui va se raidir
u_true = make_u0(
    x_grid,
    "gaussian",
    center=0.25,
    width=0.07,
    amplitude=1.5,
)
trajectory = model.evolve(u_true, n_steps)

# ==============================================================================
# 2. CADRE BAYÉSIEN (PRIOR & NOISE)
# ==============================================================================
kernel = Gaussian(length_scale=0.5, sigma=1.0)
prior = GaussianProcessPrior(kernel, nx=N)
noise = NoiseModel(sigma_noise=0.0001)

# # ==============================================================================
# # 3. OPTIMISATION DU DESIGN (OED vs MAXVOLUME)
# # ==============================================================================
# print("🔍 Recherche du design optimal...")

# # Définition des candidats (Espace-Temps)
# cand_x = np.linspace(10, N-10, 20, dtype=int)
# cand_t = np.linspace(0, n_steps, 10, dtype=int)
# n_budget = 8

# # A. Approche Statistique (OED D-Optimal)
# # On linéarise autour de la trajectoire pour le modèle non-linéaire
# design_oed, hist_oed, sigma_oed = run_greedy_oed(
#     model, prior.Sigma, noise, cand_x, cand_t, 
#     n_budget=n_budget, criterion_type="D"
# )


# # ==============================================================================
# # 4. INFERENCE & RECONSTRUCTION
# # ==============================================================================
# # On utilise le design OED pour reconstruire la condition initiale
# sensors = SpaceTimeSensors([p[0] for p in design_oed], [p[1] for p in design_oed], N)
# W = sensors.observation_operator(n_steps + 1)

# # Simuler les mesures bruitées
# y_obs = W @ trajectory.flatten() + noise.sample(len(design_oed))

# # G est l'opérateur de propagation complet (150x150)
# G_full = model.get_forward_operator(n_steps, u0_ref=u_true)

# # Il faut extraire uniquement les lignes correspondant aux capteurs du design
# # On construit la matrice d'observation W (8 x 150)
# W = sensors.observation_operator(n_steps + 1) 

# # 1. Initialiser la matrice avec les bonnes dimensions (8 mesures x 150 points d'espace)
# A_effective = np.zeros((len(design_oed), N))

# # 2. Remplir proprement chaque ligne
# for i, (xi, ti) in enumerate(design_oed):
#     # Calcul du propagateur de 0 à ti
#     G_ti = np.eye(N)
#     for n in range(ti):
#         M_n = model.get_transition_matrix(u_ref=trajectory[n])
#         G_ti = M_n @ G_ti
    
#     # On assigne la ligne xi (capteur) à la ligne i de notre matrice d'observation
#     A_effective[i, :] = G_ti[xi, :]

# # 3. Vérification de sécurité avant l'inférence




# # Force y_obs à être un vecteur de taille (8,)
# y_obs_flat = y_obs.flatten() 
# # 1. Nombre de capteurs réellement utilisés
# n_obs = len(y_obs_flat) 

# # 2. Créer la matrice de bruit à la bonne taille (8x8 et non 150x150)
# # Si votre bruit est i.i.d (indépendant), c'est une matrice diagonale
# Sigma_noise_fixed = (noise.sigma**2) * np.eye(n_obs)

# # 3. Vérification finale des dimensions avant l'appel
# # Debug prints removed for clean example output.

y_obs = trajectory + noise.sample(N, n_samples=n_steps+1)  # shape matches trajectory

A = model.get_forward_operator(n_steps=n_steps)
y_final = y_obs[n_steps-1, :]
Sigma_noise = noise.get_covariance(N)


# 4. Inférence avec les matrices recalibrées
lgm = LinearGaussianModel(
    A=model.get_forward_operator(n_steps=n_steps,u0_ref=u_true), 
    Sigma_noise=Sigma_noise, 
    mu_prior=prior.mu, 
    Sigma_prior=prior.Sigma
)

mu_post, Sigma_post = lgm.posterior(y_obs)


# ==============================================================================
# 5. VISUALISATION PRO
# ==============================================================================
viz = BOEDVisualizerPro(output_dir="results/tutorial_results")

# A. Plot Spatio-Temporel avec les deux designs pour comparaison
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
im = ax.imshow(trajectory.T, aspect='auto', origin='lower', extent=[0, n_steps, 0, N], cmap="magma")

# # Tracer OED (Cercles rouges)
# for i, (xi, ti) in enumerate(design_oed):
#     ax.scatter(ti, xi, edgecolors='white', facecolors='red', s=100, label="OED D-Opt" if i==0 else "")


ax.legend()
ax.set_title("Comparaison des placements de capteurs sur Burgers NL")
plt.savefig("results/tutorial_results/design_comparison.pdf")

# B. Plot de la reconstruction de u0
viz.plot_field(
    x_grid, u_true, mu_post, Sigma_post,
    title="Inférence de la condition initiale (Burgers NL)",
    filename="reconstruction_final.pdf"
)

print("✅ Tutoriel terminé. Résultats dans le dossier 'results/tutorial_results/'")
