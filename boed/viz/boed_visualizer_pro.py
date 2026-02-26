# ============================================================================
# boed/viz/boed_visualizer_pro.py
# ============================================================================
"""
Publication-ready visualization for BOED pipelines with PDEs.

Features:
- Field reconstruction (posterior mean + uncertainty)
- Multi-temporal PDE trajectories
- Covariance heatmaps (prior, posterior, QoI)
- QoI visualization
- Eigenvalue spectrum and cumulative variance
- Optimization history / design convergence
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple

class BOEDVisualizerPro:
    """Advanced visualizer for Bayesian Optimal Experimental Design pipelines."""

    def __init__(
        self,
        output_dir: str = "figures",
        context: str = "paper",
        style: str = "seaborn-v0_8-whitegrid",
        palette: Optional[List] = None,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        try: plt.style.use(style)
        except: pass
        sns.set_context(context)
        self.colors = palette or sns.color_palette("colorblind")

    def _save(self, fig, filename: str):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if filename.endswith(".pdf"):
            fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------- Field reconstruction -------------------
    def plot_field(
        self,
        x_grid: np.ndarray,
        true_field: np.ndarray,
        posterior_mean: np.ndarray,
        posterior_cov: np.ndarray,
        sensor_indices: Optional[List[int]] = None,
        title: Optional[str] = None,
        filename: str = "field.png",
        ci: float = 1,
        figsize: Tuple[int,int] = (12,6)
    ):
        """Plot 1D field reconstruction with posterior uncertainty."""
        std = np.sqrt(np.diag(posterior_cov))
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x_grid, true_field, lw=2.5, label="True Field", color=self.colors[0])
        ax.plot(x_grid, posterior_mean, lw=2, ls="--", color=self.colors[1], label="Posterior Mean")
        ax.fill_between(
            x_grid,
            posterior_mean - ci*std,
            posterior_mean + ci*std,
            color=self.colors[1],
            alpha=0.25,
            label=f"{100*(1-2*(0.025)):.0f}% CI"
        )

        if sensor_indices is not None:
            ax.scatter(
                x_grid[sensor_indices],
                true_field[sensor_indices],
                s=80, edgecolors="black", facecolors="none", lw=1.5, zorder=5,
                label="Sensors"
            )
            for idx in sensor_indices:
                ax.axvline(x_grid[idx], color="gray", alpha=0.2, ls=":")

        ax.set_xlabel("Spatial coordinate x")
        ax.set_ylabel("Field value")
        ax.set_title(title or "Field Reconstruction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, frameon=True)
        self._save(fig, filename)

    # ------------------- Multi-temporal PDE -------------------
    def plot_temporal_trajectory(
        self,
        x_grid: np.ndarray,
        trajectory: np.ndarray,
        timesteps: Optional[List[int]] = None,
        sensor_indices: Optional[List[int]] = None,
        title: Optional[str] = None,
        filename: str = "temporal.png",
        figsize: Tuple[int,int]=(12,6)
    ):
        """Plot PDE trajectories at selected timesteps with optional sensor overlay."""
        fig, ax = plt.subplots(figsize=figsize)
        n_steps = trajectory.shape[0]
        timesteps = timesteps or [0, n_steps//2, n_steps-1]

        for i, t in enumerate(timesteps):
            ax.plot(
                x_grid, trajectory[t],
                lw=2, ls="-" if i==0 else "--",
                color=self.colors[i % len(self.colors)],
                label=f"t={t}"
            )

        if sensor_indices is not None:
            ax.scatter(
                x_grid[sensor_indices],
                trajectory[0, sensor_indices],
                s=60, facecolors="none", edgecolors="black", lw=1.5,
                label="Sensors at t=0"
            )

        ax.set_xlabel("Spatial coordinate x")
        ax.set_ylabel("Field value")
        ax.set_title(title or "PDE Temporal Trajectories")
        ax.grid(alpha=0.3)
        ax.legend()
        self._save(fig, filename)

    # ------------------- Covariance heatmaps -------------------
    def plot_covariance(
        self,
        cov: np.ndarray,
        x_grid: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        filename: str = "covariance.png",
        figsize: Tuple[int,int]=(8,6)
    ):
        """Heatmap of covariance matrix (prior, posterior, or QoI)."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cov,
            cmap="viridis",
            xticklabels=x_grid if x_grid is not None else False,
            yticklabels=x_grid if x_grid is not None else False,
            ax=ax
        )
        ax.set_title(title or "Covariance Matrix")
        self._save(fig, filename)

    # ------------------- QoI visualization -------------------
    def plot_qoi(
        self,
        q_true: np.ndarray,
        q_post_mean: np.ndarray,
        q_post_std: np.ndarray,
        title: Optional[str] = None,
        filename: str = "qoi.png",
        ci: float = 2,
        figsize: Tuple[int,int]=(8,5)
    ):
        """Plot a Quantity of Interest with uncertainty."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(q_true, label="True QoI", lw=2, color=self.colors[0])
        ax.plot(q_post_mean, lw=2, ls="--", color=self.colors[1], label="Posterior Mean")
        ax.fill_between(
            np.arange(len(q_post_mean)),
            q_post_mean - ci*q_post_std,
            q_post_mean + ci*q_post_std,
            color=self.colors[1],
            alpha=0.25,
            label=f"{100*(1-2*0.025):.0f}% CI"
        )
        ax.set_xlabel("Index")
        ax.set_ylabel("QoI value")
        ax.set_title(title or "Quantity of Interest")
        ax.grid(alpha=0.3)
        ax.legend()
        self._save(fig, filename)

    # ------------------- Eigenvalue spectrum -------------------
    def plot_eigen_spectrum(
        self,
        eigvals: np.ndarray,
        n_eff: Optional[int] = None,
        filename: str = "spectrum.png",
        figsize: Tuple[int,int]=(14,5)
    ):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=figsize)

        # Log-spectrum
        ax1.semilogy(eigvals, "o-", lw=1.5, color=self.colors[0])
        ax1.axhline(1.0, color="gray", ls="--", alpha=0.5)
        if n_eff: ax1.axvline(n_eff, color="red", ls="--", label=f"n_eff={n_eff}")
        ax1.set_title("Eigenvalue Spectrum")
        ax1.set_xlabel("Mode index")
        ax1.set_ylabel("Eigenvalue")
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        if n_eff: ax1.legend()

        # Cumulative explained variance
        cumsum = np.cumsum(eigvals)/np.sum(eigvals)
        ax2.plot(cumsum, lw=2, color=self.colors[1])
        ax2.axhline(0.95, color="red", ls="--", alpha=0.5)
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_xlabel("Number of Modes")
        ax2.set_ylabel("Fraction Variance")
        ax2.grid(True, alpha=0.3)

        self._save(fig, filename)

    # ------------------- Optimization history -------------------
    def plot_optimization_history(
        self,
        history: List[float],
        filename: str = "opt_history.png",
        figsize: Tuple[int,int]=(10,5)
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(history, "o-", lw=2, color=self.colors[0])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Criterion value")
        ax.set_title("Optimization Convergence")
        ax.grid(True, alpha=0.3)
        ax.axhline(np.min(history), ls="--", color="green", alpha=0.5, label="Best Value")
        ax.legend()
        self._save(fig, filename)


    # ------------------- Spacetime Visualization -------------------
    def plot_spacetime_observations(
            self, 
            trajectory: np.ndarray, 
            sensors, 
            x_grid: np.ndarray, 
            n_steps: int,
            filename: str = "spacetime_obs.png",
            figsize: Tuple[int, int] = (10, 7)
        ):
        """
        Plot the PDE trajectory (Hovmöller diagram) with spacetime sensor overlay.
        
        Convention:
            - trajectory[t, x] = u(x, t)
            - x-axis : space
            - y-axis : time (t=0 at bottom, increasing upward)
        """

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Hovmöller diagram u(x, t)
        extent = [x_grid[0], x_grid[-1], 0, n_steps]

        im = ax.imshow(
            trajectory,
            aspect='auto',
            extent=extent,
            origin='lower',      # clé pour une lecture physique correcte
            cmap='viridis',
            alpha=0.85
        )

        fig.colorbar(im, ax=ax, label='Amplitude u(x,t)')

        # 2. Capteurs spatio-temporels
        sensor_x = x_grid[sensors.x_idx]
        sensor_t = sensors.t_idx

        ax.scatter(
            sensor_x,
            sensor_t,
            color='red',
            edgecolors='white',
            s=100,
            marker='X',
            label='Capteurs (x, t)',
            zorder=5
        )

        # 3. Mise en forme
        ax.set_xlabel("Espace (x)")
        ax.set_ylabel("Temps (t)")
        ax.set_title("Trajectoire PDE et placement des capteurs spatio-temporels")
        ax.legend(frameon=True)
        ax.grid(ls=':', alpha=0.3)

        # ❌ PAS de ax.invert_yaxis()

        self._save(fig, filename)
    
    # ------------------- SBOED Adaptive Trajectories -------------------
    def plot_adaptive_trajectories(
        self,
        selected_design: List[Tuple[int, int]],
        n_spatial_points: int,
        x_grid: Optional[np.ndarray] = None,
        background_trajectory: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        filename: str = "adaptive_trajectories.png",
        figsize: Tuple[int, int] = (12, 7)
    ):
        """
        Plot the paths of individual sensors over time.
        If background_trajectory is provided, plots over the field heatmap.
        """
        design_array = np.array(selected_design)
        # Sort by time to ensure lines follow chronology
        design_array = design_array[design_array[:, 1].argsort()]
        
        t_vals = design_array[:, 1]
        x_indices = design_array[:, 0]
        unique_times = np.unique(t_vals)
        n_per_step = np.sum(t_vals == unique_times[0])
        
        y_vals = x_grid[x_indices] if x_grid is not None else x_indices
        y_label = "Spatial coordinate (x)" if x_grid is not None else "Spatial Index"

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Optionnel : Fond Heatmap
        if background_trajectory is not None:
            extent = [unique_times.min(), unique_times.max(), 
                      y_vals.min() if x_grid is not None else 0, 
                      y_vals.max() if x_grid is not None else n_spatial_points]
            ax.imshow(background_trajectory.T, aspect='auto', origin='lower', 
                      extent=extent, cmap='Greys', alpha=0.3)

        # 2. Tracé par capteur (couleurs distinctes)
        # On utilise ta palette de couleurs self.colors définie dans __init__
        for i in range(n_per_step):
            color = self.colors[i % len(self.colors)]
            path_t = unique_times
            path_y = y_vals[i::n_per_step]
            
            ax.plot(path_t, path_y, color=color, linestyle='--', alpha=0.6, lw=1.5, zorder=2)
            ax.scatter(path_t, path_y, color=[color], s=60, edgecolors='black', 
                       label=f"Sensor {i+1}", zorder=3)

        ax.set_title(title or f"Adaptive SBOED Trajectories ({n_per_step} sensors)")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.2, ls=':')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self._save(fig, filename)

    # ------------------- Sequential QoI Uncertainty -------------------
    def plot_sequential_qoi_learning(
        self,
        x_qoi: np.ndarray,
        true_qoi: np.ndarray,
        history_mu: List[np.ndarray],
        history_sigma: List[np.ndarray],
        filename: str = "qoi_learning.png",
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plots how the QoI uncertainty shrinks over sequential design steps.
        Shows the 'narrowing tube' effect.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color gradient from light blue to dark blue
        n_steps = len(history_mu)
        
        # Plot True QoI
        ax.plot(x_qoi, true_qoi, color='red', lw=2.5, label="Ground Truth", zorder=5)
        
        for k in range(n_steps):
            mu_k = history_mu[k]
            std_k = np.sqrt(np.diag(history_sigma[k]))
            
            # Plus on avance, plus la couleur est intense
            alpha_fill = 0.05 + 0.2 * (k / n_steps)
            alpha_line = 0.3 + 0.7 * (k / n_steps)
            
            ax.plot(x_qoi, mu_k, color=self.colors[0], alpha=alpha_line, lw=1)
            ax.fill_between(x_qoi, mu_k - 2*std_k, mu_k + 2*std_k, 
                             color=self.colors[0], alpha=alpha_fill)

        # Label for the last (most certain) step
        ax.plot([], [], color=self.colors[0], lw=2, label="Final Posterior Mean")
        
        ax.set_title("Sequential Reduction of QoI Uncertainty")
        ax.set_xlabel("Space (x) within QoI zone")
        ax.set_ylabel("QoI Value")
        ax.legend()
        ax.grid(alpha=0.3)
        
        self._save(fig, filename)
