# ============================================================================
# boed/viz/boed_visualizer.py
# ============================================================================
"""
Advanced visualization toolkit for BOED pipelines.

Supports field reconstruction, covariance heatmaps, temporal evolution, 
QoI, eigenvalue spectrum, and optimization history.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple

class BOEDVisualizer:
    """Comprehensive visualizer for Bayesian Optimal Experimental Design."""

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
        filename: str = "field.pdf",
        ci: float = 2,
        figsize: Tuple[int,int] = (12,6)
    ):
        """Plot 1D field reconstruction with posterior uncertainty."""
        std = np.sqrt(np.diag(posterior_cov))
        fig, ax = plt.subplots(figsize=figsize)

        # True field and posterior
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

        # Sensor locations
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

    # ------------------- Temporal evolution -------------------
    def plot_temporal_evolution(
        self,
        x_grid: np.ndarray,
        trajectory: np.ndarray,
        timesteps: Optional[List[int]] = None,
        title: Optional[str] = None,
        filename: str = "temporal.pdf",
        figsize: Tuple[int,int]=(12,6)
    ):
        """Plot temporal evolution of PDE solution."""
        fig, ax = plt.subplots(figsize=figsize)
        n_steps = trajectory.shape[0]
        timesteps = timesteps or [0, n_steps//2, n_steps-1]

        for i, t in enumerate(timesteps):
            ax.plot(
                x_grid,
                trajectory[t],
                lw=2,
                ls="-" if i==0 else "--",
                color=self.colors[i % len(self.colors)],
                label=f"t={t}"
            )

        ax.set_xlabel("Spatial coordinate x")
        ax.set_ylabel("Field value")
        ax.set_title(title or "Temporal Evolution")
        ax.grid(alpha=0.3)
        ax.legend()
        self._save(fig, filename)

    # ------------------- Covariance heatmaps -------------------
    def plot_covariance(
        self,
        cov: np.ndarray,
        x_grid: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        filename: str = "covariance.pdf",
        figsize: Tuple[int,int]=(8,6)
    ):
        """Heatmap of covariance matrix."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cov, cmap="viridis", xticklabels=x_grid, yticklabels=x_grid, ax=ax)
        ax.set_title(title or "Covariance Matrix")
        self._save(fig, filename)

    # ------------------- Eigenvalue spectrum -------------------
    def plot_eigen_spectrum(
        self,
        eigvals: np.ndarray,
        n_eff: Optional[int] = None,
        filename: str = "spectrum.pdf",
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
        filename: str = "opt_history.pdf",
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
