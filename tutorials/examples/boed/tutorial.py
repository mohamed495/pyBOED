# import numpy as np
# import matplotlib.pyplot as plt
# import numpy.linalg as la
# from boed.priors.kernels import Gaussian
# from boed.priors.gp_priors import GaussianProcessPrior
# from boed.core.noise import NoiseModel
# from boed.core import make_u0
# from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
# from boed.inference import LinearGaussianModel
# from boed.observations.sensors import SpaceTimeSensors
# from boed.design.greedy import run_greedy_oed

# # ===============================================================================
# # 1. FONCTIONS SUPPLÉMENTAIRES (SBOED & VIZ)
# # ===============================================================================

# def run_sboed_step(model, Sigma_prior, noise, cand_x, cand_t, budget_per_t, L_qoi=None):
#     """Logique Séquentielle : 1 décision par pas de temps (Style Go & Chen)"""
#     curr_sig = Sigma_prior.copy()
#     design = []
#     M = model.get_transition_matrix()
#     sigma2 = noise.sigma_noise**2
    
#     for ti in sorted(cand_t):
#         T_ti = np.linalg.matrix_power(M, ti)
#         for _ in range(budget_per_t):
#             best_score, best_x, best_sig = np.inf, None, None
#             for xi in cand_x:
#                 if (xi, ti) in design: continue
#                 g = T_ti[xi, :].reshape(1, -1)
#                 S = (g @ curr_sig @ g.T).item() + sigma2
#                 # Mise à jour de la covariance (Sherman-Morrison)
#                 sig_tmp = curr_sig - (curr_sig @ g.T @ g @ curr_sig) / S
                
#                 # Critère C (QoI) ou A (Global)
#                 if L_qoi is not None:
#                     # Variance de la QoI : trace(L * Sigma * L^T)
#                     score = np.trace(L_qoi @ sig_tmp @ L_qoi.T) if L_qoi.ndim > 1 else (L_qoi @ sig_tmp @ L_qoi.T)
#                 else:
#                     score = np.trace(sig_tmp)
                
#                 if score < best_score:
#                     best_score, best_x, best_sig = score, xi, sig_tmp
#             if best_x is not None:
#                 design.append((best_x, ti))
#                 curr_sig = best_sig
#     return design

# def plot_sensor_trajectories(selected_design, n_spatial_points, x_grid=None, title=None):
#     """Visualisation des trajectoires avec couleurs distinctes par capteur"""
#     design_array = np.array(selected_design)
#     t_vals = design_array[:, 1]
#     x_indices = design_array[:, 0]
#     unique_times = np.unique(t_vals)
#     n_per_step = np.sum(t_vals == unique_times[0])
#     y_vals = x_grid[x_indices] if x_grid is not None else x_indices
    
#     plt.figure(figsize=(10, 6))
#     colors = plt.cm.get_cmap('tab10', n_per_step)
    
#     for i in range(n_per_step):
#         path_t = unique_times
#         path_y = y_vals[i::n_per_step]
#         plt.plot(path_t, path_y, color=colors(i), linestyle='--', alpha=0.6, label=f"Capteur {i+1}")
#         plt.scatter(path_t, path_y, color=[colors(i)], s=50, edgecolors='k')

#     plt.title(title or "Trajectoires des capteurs")
#     plt.xlabel("Temps (t)")
#     plt.ylabel("Position (x)")
#     plt.legend()
#     plt.grid(True, alpha=0.2)
#     plt.show()

# # ===============================================================================
# # 2. CONFIGURATION ET SIMULATION
# # ===============================================================================
# N, dt, n_steps = 150, 0.01, 100
# x_grid = np.linspace(0, 1, N)
# model = AdvectionDiffusion1D_CN(N, dt, diffusivity=0.01, velocity=0.5)

# u0 = make_u0(x_grid, "double_gaussian", centers=(0.2, 0.4))
# trajectory = model.evolve(u0, n_steps)

# prior_process = GaussianProcessPrior(Gaussian(0.05, 1.0), nx=N)
# Sigma_prior = prior_process.Sigma
# noise = NoiseModel(sigma_noise=0.01)

# candidates_x = np.linspace(10, N-10, 25, dtype=int)
# candidates_t = np.linspace(0, n_steps, 15, dtype=int)

# # Définition de la QoI (Zone gauche [0.15, 0.35] par exemple)
# L_qoi = np.zeros(N)
# L_qoi[int(0.15*N):int(0.35*N)] = 1.0

# # ===============================================================================
# # 3. COMPARAISON DES STRATÉGIES
# # ===============================================================================

# # A. Greedy Global (Statique)
# print("🔎 Optimisation C-Opt Statique...")
# des_static, _, _ = run_greedy_oed(N, model, Sigma_prior, noise, candidates_x, candidates_t, 8, "C", L_qoi=L_qoi)

# # B. SBOED (Séquentiel / Trajectoires)
# print("🏃 Optimisation SBOED (1 capteur mobile)...")
# des_sboed = run_sboed_step(model, Sigma_prior, noise, candidates_x, candidates_t, budget_per_t=1, L_qoi=L_qoi)

# # ===============================================================================
# # 4. AFFICHAGE DES RÉSULTATS
# # ===============================================================================

# # Visualisation des trajectoires SBOED
# plot_sensor_trajectories(des_sboed, N, x_grid=x_grid, title="Trajectoire SBOED (Focus QoI)")

# # Comparaison des designs sur le champ physique
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ax1.imshow(trajectory.T, aspect='auto', origin='lower', extent=[0, n_steps, 0, 1], cmap='viridis', alpha=0.8)
# ax1.scatter([p[1] for p in des_static], x_grid[[p[0] for p in des_static]], c='red', marker='x', label='Static C-Opt')
# ax1.set_title("Design Statique (Omniscient)")
# ax1.legend()

# ax2.imshow(trajectory.T, aspect='auto', origin='lower', extent=[0, n_steps, 0, 1], cmap='viridis', alpha=0.8)
# ax2.plot([p[1] for p in des_sboed], x_grid[[p[0] for p in des_sboed]], 'r--')
# ax2.scatter([p[1] for p in des_sboed], x_grid[[p[0] for p in des_sboed]], c='white', edgecolors='red', label='SBOED Path')
# ax2.set_title("Design Séquentiel (Trajectoire)")
# ax2.legend()

# plt.show()


"""
Bayesian Optimal Experimental Design for 1D Burgers Equation
============================================================

Problem setup:
- PDE: u_t + u*u_x = nu*u_xx on [0,1]
- Initial condition: u0(x) = sin(2*pi*x)
- Boundary conditions: periodic
- Parameter to infer: theta = nu (viscosity)
- Prior: nu ~ Uniform(nu_min, nu_max)
- Observations: y(x) = u(x, T; nu) + epsilon, epsilon ~ N(0, sigma^2)
- Design variable: xi = T (observation time)
- Goal: Find T* that maximizes I(theta; y | xi) = EIG
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class BurgersSolver:
    """
    Solver for 1D Burgers equation using finite differences.
    Method: Crank-Nicolson for diffusion + upwind for advection
    """
    
    def __init__(self, nx=100, L=1.0):
        """
        Parameters:
        -----------
        nx : int
            Number of spatial points
        L : float
            Domain length [0, L]
        """
        self.nx = nx
        self.L = L
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
    def initial_condition(self):
        """Sinusoidal initial condition"""
        return np.sin(2 * np.pi * self.x / self.L)
    
    def solve(self, nu, T, dt=None, n_snapshots=1):
        """
        Solve Burgers equation
        
        Parameters:
        -----------
        nu : float
            Viscosity parameter
        T : float
            Final time
        dt : float, optional
            Time step (auto-computed if None)
        n_snapshots : int
            Number of snapshots to return (including t=0 and t=T)
            
        Returns:
        --------
        times : array
            Times at which solution is returned
        solutions : array (n_snapshots, nx)
            Solutions at specified times
        """
        # Stability criterion: CFL for advection and diffusion
        if dt is None:
            u_max = 1.0  # maximum of sin
            dt_advection = 0.5 * self.dx / u_max
            dt_diffusion = 0.5 * self.dx**2 / nu if nu > 0 else dt_advection
            dt = min(dt_advection, dt_diffusion)
        
        nt = int(np.ceil(T / dt))
        dt = T / nt  # adjust to hit T exactly
        
        # Initialize
        u = self.initial_condition()
        
        # Storage for snapshots
        snapshot_times = np.linspace(0, T, n_snapshots)
        solutions = np.zeros((n_snapshots, self.nx))
        solutions[0] = u.copy()
        snapshot_idx = 1
        
        # Time stepping
        alpha = nu * dt / (2 * self.dx**2)
        
        for n in range(nt):
            t = (n + 1) * dt
            
            # Simple explicit scheme for stability
            # Upwind for advection, central for diffusion
            u_new = u.copy()
            
            for i in range(1, self.nx - 1):
                # Advection term (upwind)
                if u[i] > 0:
                    du_dx = (u[i] - u[i-1]) / self.dx
                else:
                    du_dx = (u[i+1] - u[i]) / self.dx
                
                # Diffusion term (central)
                d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / self.dx**2
                
                # Update
                u_new[i] = u[i] - dt * u[i] * du_dx + dt * nu * d2u_dx2
            
            # Periodic boundary conditions
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
            
            u = u_new
            
            # Store snapshots
            if snapshot_idx < n_snapshots and t >= snapshot_times[snapshot_idx]:
                solutions[snapshot_idx] = u.copy()
                snapshot_idx += 1
        
        return snapshot_times, solutions


class BurgersBOED:
    """
    Bayesian Optimal Experimental Design for Burgers equation
    """
    
    def __init__(self, nu_min=0.001, nu_max=0.1, nx=100, sigma_obs=0.01):
        """
        Parameters:
        -----------
        nu_min, nu_max : float
            Prior bounds for viscosity: nu ~ Uniform(nu_min, nu_max)
        nx : int
            Spatial resolution
        sigma_obs : float
            Observation noise standard deviation
        """
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.sigma_obs = sigma_obs
        self.solver = BurgersSolver(nx=nx)
        
    def sample_prior(self, n_samples):
        """Sample from prior p(nu)"""
        return np.random.uniform(self.nu_min, self.nu_max, n_samples)
    
    def forward_model(self, nu, T):
        """
        Run forward model: solve Burgers and return solution at time T
        
        Returns:
        --------
        u : array
            Solution at time T
        """
        _, solutions = self.solver.solve(nu, T, n_snapshots=2)
        return solutions[-1]  # Return solution at t=T
    
    def generate_observation(self, nu, T):
        """
        Generate noisy observation: y = u(x, T; nu) + epsilon
        """
        u_clean = self.forward_model(nu, T)
        noise = np.random.normal(0, self.sigma_obs, size=u_clean.shape)
        return u_clean + noise
    
    def log_likelihood(self, y_obs, nu, T):
        """
        Compute log p(y | nu, T)
        
        Assumes Gaussian observation noise: y ~ N(u(x,T;nu), sigma^2*I)
        """
        u_pred = self.forward_model(nu, T)
        residual = y_obs - u_pred
        return -0.5 * np.sum(residual**2) / self.sigma_obs**2
    
    def estimate_eig_nmc(self, T, N_outer=100, N_inner=50, verbose=True):
        """
        Estimate Expected Information Gain using Nested Monte Carlo
        
        EIG = E_{theta,y}[log p(y|theta,xi)] - E_y[log p(y|xi)]
            = E_{theta,y}[log p(y|theta,xi)] - E_y[log E_theta[p(y|theta,xi)]]
        
        Parameters:
        -----------
        T : float
            Design variable (observation time)
        N_outer : int
            Number of outer Monte Carlo samples (theta, y pairs)
        N_inner : int
            Number of inner Monte Carlo samples (for marginal likelihood)
        verbose : bool
            Show progress bar
            
        Returns:
        --------
        eig : float
            Estimated Expected Information Gain
        """
        # Term 1: E_{theta,y}[log p(y|theta,xi)]
        term1_samples = []
        marginal_log_likelihood_samples = []
        
        if verbose:
            print(f"  Computing EIG at T={T:.3f}...", end='', flush=True)
        
        for i in range(N_outer):
            # Sample from prior and generate observation
            nu_true = self.sample_prior(1)[0]
            y_obs = self.generate_observation(nu_true, T)
            
            # Term 1: log p(y | theta_true, T)
            ll = self.log_likelihood(y_obs, nu_true, T)
            term1_samples.append(ll)
            
            # Term 2: estimate log p(y | T) using importance sampling
            # log p(y|T) ≈ log(1/N * sum_i p(y|theta_i,T))
            nu_samples = self.sample_prior(N_inner)
            log_likelihoods = np.array([
                self.log_likelihood(y_obs, nu, T) for nu in nu_samples
            ])
            
            # Numerically stable log-sum-exp
            max_ll = np.max(log_likelihoods)
            log_marginal = max_ll + np.log(np.mean(np.exp(log_likelihoods - max_ll)))
            marginal_log_likelihood_samples.append(log_marginal)
        
        # Compute EIG
        term1 = np.mean(term1_samples)
        term2 = np.mean(marginal_log_likelihood_samples)
        eig = term1 - term2
        
        if verbose:
            print(" done!")
        
        return eig
    
    def compute_eig_vs_time(self, T_values, N_outer=100, N_inner=50):
        """
        Compute EIG for multiple observation times
        
        Parameters:
        -----------
        T_values : array
            Array of observation times to evaluate
        N_outer, N_inner : int
            Monte Carlo sample sizes
            
        Returns:
        --------
        eig_values : array
            EIG at each time point
        """
        eig_values = []
        
        print(f"Computing EIG for {len(T_values)} time points...")
        print(f"Prior: nu ~ U({self.nu_min}, {self.nu_max})")
        print(f"Observation noise: sigma = {self.sigma_obs}")
        print(f"Monte Carlo: N_outer={N_outer}, N_inner={N_inner}")
        print("-" * 60)
        
        for T in T_values:
            eig = self.estimate_eig_nmc(T, N_outer=N_outer, N_inner=N_inner, verbose=True)
            eig_values.append(eig)
            print(f"T = {T:.3f}: EIG = {eig:.4f}")
        
        return np.array(eig_values)
    
    def find_optimal_design(self, T_min=0.1, T_max=2.0, n_grid=20, N_outer=100, N_inner=50):
        """
        Find optimal observation time T* that maximizes EIG
        
        Uses coarse grid search followed by refinement
        
        Returns:
        --------
        T_opt : float
            Optimal observation time
        eig_opt : float
            Maximum EIG value
        T_grid : array
            Grid of evaluated times
        eig_grid : array
            EIG values at grid points
        """
        # Coarse grid search
        T_grid = np.linspace(T_min, T_max, n_grid)
        eig_grid = self.compute_eig_vs_time(T_grid, N_outer=N_outer, N_inner=N_inner)
        
        # Find optimum
        idx_opt = np.argmax(eig_grid)
        T_opt = T_grid[idx_opt]
        eig_opt = eig_grid[idx_opt]
        
        print("\n" + "=" * 60)
        print(f"Optimal design found:")
        print(f"  T* = {T_opt:.4f}")
        print(f"  EIG* = {eig_opt:.4f}")
        print("=" * 60)
        
        return T_opt, eig_opt, T_grid, eig_grid


def visualize_burgers_evolution(nu_values=[0.01, 0.05, 0.1], T_values=[0.2, 0.5, 1.0]):
    """
    Visualize how Burgers solution evolves for different viscosities and times
    """
    solver = BurgersSolver(nx=100)
    
    fig, axes = plt.subplots(len(nu_values), len(T_values), 
                             figsize=(12, 3*len(nu_values)), sharex=True, sharey=True)
    
    for i, nu in enumerate(nu_values):
        for j, T in enumerate(T_values):
            ax = axes[i, j] if len(nu_values) > 1 else axes[j]
            
            # Solve
            times, solutions = solver.solve(nu, T, n_snapshots=2)
            u0 = solutions[0]
            uT = solutions[1]
            
            # Plot
            ax.plot(solver.x, u0, 'k--', label='t=0', alpha=0.5)
            ax.plot(solver.x, uT, 'r-', label=f't={T}', linewidth=2)
            ax.set_title(f'ν={nu}, T={T}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if j == 0:
                ax.set_ylabel('u(x,t)')
            if i == len(nu_values) - 1:
                ax.set_xlabel('x')
    
    plt.tight_layout()
    return fig


def plot_eig_results(T_grid, eig_grid, T_opt, eig_opt):
    """
    Plot EIG vs observation time
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(T_grid, eig_grid, 'bo-', linewidth=2, markersize=8, label='EIG')
    ax.axvline(T_opt, color='r', linestyle='--', linewidth=2, label=f'T* = {T_opt:.3f}')
    ax.scatter([T_opt], [eig_opt], color='r', s=200, zorder=5, marker='*', 
               edgecolors='darkred', linewidth=2, label=f'EIG* = {eig_opt:.3f}')
    
    ax.set_xlabel('Observation Time T', fontsize=12)
    ax.set_ylabel('Expected Information Gain', fontsize=12)
    ax.set_title('Optimal Experimental Design for Burgers Equation\n(Inferring Viscosity ν)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    return fig


def main_example():
    """
    Main example: find optimal observation time for viscosity inference
    """
    print("=" * 70)
    print("BAYESIAN OPTIMAL EXPERIMENTAL DESIGN FOR BURGERS EQUATION")
    print("=" * 70)
    print("\nProblem Setup:")
    print("  - PDE: u_t + u*u_x = ν*u_xx on x ∈ [0,1]")
    print("  - Initial condition: u₀(x) = sin(2πx)")
    print("  - Parameter to infer: θ = ν (viscosity)")
    print("  - Design variable: ξ = T (observation time)")
    print("  - Goal: maximize I(θ; y | ξ)")
    print("\n" + "=" * 70 + "\n")
    
    # Step 1: Visualize forward model behavior
    print("Step 1: Visualizing Burgers equation solutions...")
    fig1 = visualize_burgers_evolution()
    plt.savefig('burgers_evolution.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: burgers_evolution.png\n")
    
    # Step 2: Setup BOED problem
    print("Step 2: Setting up BOED problem...")
    boed = BurgersBOED(
        nu_min=0.01,
        nu_max=0.1,
        nx=100,
        sigma_obs=0.01
    )
    print("  ✓ BOED problem initialized\n")
    
    # Step 3: Compute EIG over time range
    print("Step 3: Computing optimal design...")
    T_opt, eig_opt, T_grid, eig_grid = boed.find_optimal_design(
        T_min=0.1,
        T_max=2.0,
        n_grid=15,  # Use 15 points for reasonable computation time
        N_outer=50,  # Reduced for faster computation
        N_inner=30
    )
    
    # Step 4: Visualize results
    print("\nStep 4: Creating visualizations...")
    fig2 = plot_eig_results(T_grid, eig_grid, T_opt, eig_opt)
    plt.savefig('/home/claude/boed_results.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: boed_results.png\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Optimal observation time: T* = {T_opt:.4f}")
    print(f"Maximum information gain: EIG* = {eig_opt:.4f}")
    print("\nInterpretation:")
    print(f"  - Observing at t = {T_opt:.4f} provides the most information")
    print(f"    about the viscosity parameter ν")
    print(f"  - Too early: insufficient diffusion to distinguish ν values")
    print(f"  - Too late: over-diffused, less sensitivity to ν")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main example
    main_example()
