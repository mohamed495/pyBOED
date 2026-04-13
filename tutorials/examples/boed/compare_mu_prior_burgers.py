"""Compare Burgers EIG bounds across lambda values and repeated runs.

This standalone script reproduces the core pipeline from
`tuto_our_proposal_burgers.ipynb` without modifying the notebook:
- estimate l_theta and H_theta (Monte Carlo over prior)
- estimate Sigma_Y
- build Sigma_signal
- greedy sensor selection from LB criterion
- compare LB/UB distributions for several lambda values and mu_prior values
"""

import os
import argparse
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.linalg as sla
from math import ceil
from matplotlib.lines import Line2D

from boed.priors.kernels import Matern32
from boed.priors.gp_priors import GaussianProcessPrior
from boed.core.noise import ColoredNoise
from boed.core import make_u0
from boed.pde.burgers import Burgers_CN
from boed.utils.observation import build_selection_matrices

DEFAULT_BASE_SEED = 42

def make_prior(kernel_obj, nx, mu):
    """Compatibility wrapper for constructor variants."""
    try:
        return GaussianProcessPrior(kernel_obj, nx=nx, mu=mu)
    except TypeError:
        prior_obj = GaussianProcessPrior(kernel_obj, nx=nx)
        prior_obj.mu = mu
        return prior_obj


def make_forward_model(pde_model, n_steps_local):
    """Forward map G : theta -> u(T)."""

    def G(theta):
        u_final = pde_model.evolve(u0=theta, n_steps=n_steps_local)
        return u_final[-1, :]

    return G

def jacobian_fd(G_fun, theta, h=None):
    """Centered finite-difference Jacobian of G at theta."""
    d = theta.shape[0]
    if h is None:
        h = np.finfo(float).eps ** (1 / 3) * (la.norm(theta) + 1e-8)

    m = G_fun(theta).shape[0]
    J = np.zeros((m, d))

    for j in range(d):
        e_j = np.zeros(d)
        e_j[j] = 1.0
        J[:, j] = (G_fun(theta + h * e_j) - G_fun(theta - h * e_j)) / (2 * h)

    return J


def estimate_H_and_l_consistent(G_fun, prior, Sigma_obs_mat, n_samples):
    """Consistent MC estimator of l_theta and H_theta."""
    Sigma_inv = la.inv(Sigma_obs_mat)
    Sigma_inv_sqrt = sla.sqrtm(Sigma_inv)

    theta_samples = np.random.multivariate_normal(
        mean=prior.mu, cov=prior.Sigma, size=n_samples
    )

    J_all = np.array([jacobian_fd(G_fun, theta) for theta in theta_samples])

    # l_theta = E[J]^T
    l_theta = J_all.mean(axis=0).T

    # X_k = J_k^T Sigma^{-1/2}
    X_all = np.einsum("kmd,mn->kdn", J_all, Sigma_inv_sqrt)
    X_mean = X_all.mean(axis=0)
    X_centered = X_all - X_mean

    # Covariance over (sample, obs) dimensions
    H_theta = np.einsum("kdm,knm->dn", X_centered, X_centered) / (n_samples - 1)

    return l_theta, H_theta


def estimate_Sigma_Y(G_fun, prior, Sigma_obs_mat, n_samples):
    """MC estimator of Sigma_Y = Sigma_obs + Cov[G(theta)]."""
    theta_samples = np.random.multivariate_normal(
        mean=prior.mu, cov=prior.Sigma, size=n_samples
    )

    U = np.array([G_fun(theta) for theta in theta_samples])
    U = U[np.all(np.isfinite(U), axis=1)]
    if U.shape[0] < 2:
        raise RuntimeError("Not enough finite forward samples to estimate Sigma_Y.")

    return Sigma_obs_mat + np.cov(U, rowvar=False, bias=False)


def compute_Sigma_signal(l_theta, H_theta, prior, Sigma_obs_mat):
    """Sigma_signal = Sigma_obs + l^T (Sigma_prior^{-1} + H)^{-1} l."""
    A = la.inv(prior.Sigma) + H_theta
    X = la.solve(A, l_theta)
    return Sigma_obs_mat + l_theta.T @ X


def delta_UB(Sigma_signal, Sigma_obs_mat, W):
    """Upper-bound style estimator used in the notebook."""

    def log_ratio(A, B, M):
        _, la_ = la.slogdet(M.T @ A @ M)
        _, lb_ = la.slogdet(M.T @ B @ M)
        return la_ - lb_

    eye = np.eye(Sigma_obs_mat.shape[0])
    return 0.5 * (
        log_ratio(Sigma_signal, Sigma_obs_mat, W)
        - log_ratio(Sigma_signal, Sigma_obs_mat, eye)
    )


def delta_LB(Sigma_Y, Sigma_obs_mat, W):
    """Lower-bound style estimator used in the notebook."""

    def log_ratio(A, B, M):
        _, la_ = la.slogdet(M.T @ A @ M)
        _, lb_ = la.slogdet(M.T @ B @ M)
        return la_ - lb_

    eye = np.eye(Sigma_obs_mat.shape[0])
    return 0.5 * (log_ratio(Sigma_Y, Sigma_obs_mat, W) - log_ratio(Sigma_Y, Sigma_obs_mat, eye))


def greedy_maximize_LB(Sigma_Y, Sigma_obs_mat, n_sensors):
    """Greedy subset maximizing the LB proxy."""
    n_obs = Sigma_Y.shape[0]
    selected = []
    remaining = list(range(n_obs))

    for _ in range(n_sensors):
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            S = selected + [idx]
            ix = np.ix_(S, S)
            score = la.slogdet(Sigma_Y[ix])[1] - la.slogdet(Sigma_obs_mat[ix])[1]
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(selected)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Burgers LB/UB bounds across lambda values with repeated runs."
    )
    parser.add_argument("--n", type=int, default=100, help="Spatial size N.")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step.")
    parser.add_argument("--n-steps", type=int, default=100, help="Total steps (for metadata).")
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise std.")
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=None,
        help="List of Burgers nonlinearity values to compare (e.g. 0 0.25 0.5 0.75 1).",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=None,
        help="Single Burgers nonlinearity (legacy option). Used only if --lambda-values is not set.",
    )
    parser.add_argument(
        "--n-samples-lh",
        type=int,
        default=120,
        help="MC samples for l_theta and H_theta.",
    )
    parser.add_argument(
        "--n-samples-sigma-y",
        type=int,
        default=800,
        help="MC samples for Sigma_Y.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeated runs per (mu, lambda) configuration.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Base random seed used to generate repeat-specific seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/results_mu_prior_burgers",
        help="Output directory for figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    N = int(args.n)
    dt = float(args.dt)
    n_steps = int(args.n_steps)
    sigma = float(args.sigma)
    if args.lambda_values is not None:
        lambda_values = [float(v) for v in args.lambda_values]
    elif args.lambda_ is not None:
        lambda_values = [float(args.lambda_)]
    else:
        lambda_values = [0.0, 0.5, 1.0]

    # Keep ordering stable while dropping duplicates.
    lambda_values = list(dict.fromkeys(lambda_values))
    n_samples_lh = int(args.n_samples_lh)
    n_samples_sigma_y = int(args.n_samples_sigma_y)
    n_repeats = int(args.n_repeats)
    base_seed = int(args.base_seed)

    if n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1.")

    # Budgets for LB/UB comparison
    sensor_budgets = [b for b in [5, 10, 15, 20, 25] if b <= N]
    if not sensor_budgets:
        raise ValueError(f"N={N} is too small for the configured sensor budgets.")

    x_grid = np.linspace(0.0, 1.0, N + 2)[1:-1]
    u0_true = make_u0(x_grid, "gaussian", center=0.25, width=0.07, amplitude=1.5)

    kernel = Matern32(length_scale=0.1, sigma=1.0)
    noise = ColoredNoise(sigma_noise=sigma, correlation_length=20.3)
    Sigma_obs = noise.get_covariance(N)

    # -----------------------------------------------------------------------
    # Define three prior means mu_prior
    # -----------------------------------------------------------------------
    mu_configs = [
        ("alpha=0", np.zeros(N), "mu0"),
        # ("alpha=+4", np.ones(N) * 4.0, "mu_pos4"),
        # ("alpha=-4", np.ones(N) * (-4.0), "mu_neg4"),
    ]

    results = {}

    for mu_idx, (name, mu_vec, mu_slug) in enumerate(mu_configs):
        print("\n" + "=" * 72)
        print(f"Running case: {name}")

        prior = make_prior(kernel, N, mu_vec)
        results[name] = {"mu": mu_vec, "mu_slug": mu_slug, "by_lambda": {}}

        for lam_idx, lambda_ in enumerate(lambda_values):
            print(f"  -> lambda={lambda_:g} | repeats={n_repeats}")
            model = Burgers_CN(N=N, dt=dt, diffusivity=0.02, lambda_=lambda_)
            G = make_forward_model(model, n_steps)
            max_budget = max(sensor_budgets)

            lb_runs = []
            ub_runs = []
            gap_runs = []
            sigma_y_runs = []
            sigma_signal_runs = []
            indices_path_last = None

            for repeat_idx in range(n_repeats):
                # Deterministic but unique seed for each (mu, lambda, repeat) tuple.
                seed = base_seed + 100000 * mu_idx + 1000 * lam_idx + repeat_idx
                np.random.seed(seed)

                l_theta, H_theta = estimate_H_and_l_consistent(
                    G_fun=G,
                    prior=prior,
                    Sigma_obs_mat=Sigma_obs,
                    n_samples=n_samples_lh,
                )
                Sigma_signal = compute_Sigma_signal(l_theta, H_theta, prior, Sigma_obs)

                Sigma_Y = estimate_Sigma_Y(
                    G_fun=G,
                    prior=prior,
                    Sigma_obs_mat=Sigma_obs,
                    n_samples=n_samples_sigma_y,
                )

                indices_path = greedy_maximize_LB(
                    Sigma_Y=Sigma_Y,
                    Sigma_obs_mat=Sigma_obs,
                    n_sensors=max_budget,
                )
                indices_path_last = indices_path

                lb_vals = []
                ub_vals = []
                for budget in sensor_budgets:
                    indices = indices_path[:budget]
                    W, _ = build_selection_matrices(N, indices)
                    lb_vals.append(delta_LB(Sigma_Y, Sigma_obs, W))
                    ub_vals.append(delta_UB(Sigma_signal, Sigma_obs, W))

                lb_arr = np.array(lb_vals, dtype=float)
                ub_arr = np.array(ub_vals, dtype=float)
                lb_runs.append(lb_arr)
                ub_runs.append(ub_arr)
                gap_runs.append(ub_arr - lb_arr)
                sigma_y_runs.append(Sigma_Y)
                sigma_signal_runs.append(Sigma_signal)

            lb_runs = np.array(lb_runs)
            ub_runs = np.array(ub_runs)
            gap_runs = np.array(gap_runs)
            sigma_y_runs = np.array(sigma_y_runs)
            sigma_signal_runs = np.array(sigma_signal_runs)

            lb_median = np.median(lb_runs, axis=0)
            ub_median = np.median(ub_runs, axis=0)
            gap_median = np.median(gap_runs, axis=0)

            results[name]["by_lambda"][lambda_] = {
                "indices_path_last": indices_path_last,
                "LB_runs": lb_runs,
                "UB_runs": ub_runs,
                "GAP_runs": gap_runs,
                "LB_median": lb_median,
                "UB_median": ub_median,
                "GAP_median": gap_median,
                "LB_q1": np.quantile(lb_runs, 0.25, axis=0),
                "LB_q3": np.quantile(lb_runs, 0.75, axis=0),
                "UB_q1": np.quantile(ub_runs, 0.25, axis=0),
                "UB_q3": np.quantile(ub_runs, 0.75, axis=0),
                "GAP_q1": np.quantile(gap_runs, 0.25, axis=0),
                "GAP_q3": np.quantile(gap_runs, 0.75, axis=0),
                "Sigma_Y_mean": sigma_y_runs.mean(axis=0),
                "Sigma_signal_mean": sigma_signal_runs.mean(axis=0),
            }

            print(
                f"     final budget {sensor_budgets[-1]} | "
                f"LB median={lb_median[-1]:.3f}, UB median={ub_median[-1]:.3f}, "
                f"gap median={gap_median[-1]:.3f}"
            )

    # -----------------------------------------------------------------------
    # Visualization: for each mu, show LB/UB across lambda
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    saved_files = []
    n_lambda = len(lambda_values)

    for name, _, mu_slug in mu_configs:
        case = results[name]

        all_lb = np.concatenate([case["by_lambda"][lam]["LB_runs"].ravel() for lam in lambda_values])
        all_ub = np.concatenate([case["by_lambda"][lam]["UB_runs"].ravel() for lam in lambda_values])
        ymin_bounds = min(float(all_lb.min()), float(all_ub.min()))
        ymax_bounds = max(float(all_lb.max()), float(all_ub.max()))
        margin = 0.05 * (ymax_bounds - ymin_bounds + 1e-12)

        if n_lambda <= 3:
            n_plot_rows, n_plot_cols = 1, n_lambda
        elif n_lambda <= 6:
            n_plot_rows, n_plot_cols = 2, ceil(n_lambda / 2)
        else:
            n_plot_rows, n_plot_cols = 3, ceil(n_lambda / 3)

        fig, axes = plt.subplots(
            n_plot_rows,
            n_plot_cols,
            figsize=(4.4 * n_plot_cols, 4.2 * n_plot_rows),
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        if len(sensor_budgets) > 1:
            min_step = float(np.min(np.diff(sensor_budgets)))
        else:
            min_step = 1.0
        box_width = 0.30 * min_step
        offset = 0.20 * min_step

        for col, lambda_ in enumerate(lambda_values):
            r = case["by_lambda"][lambda_]
            ax = axes[col]
            positions = np.array(sensor_budgets, dtype=float)
            lb_data = [r["LB_runs"][:, i] for i in range(len(sensor_budgets))]
            ub_data = [r["UB_runs"][:, i] for i in range(len(sensor_budgets))]

            lb_box = ax.boxplot(
                lb_data,
                positions=positions - offset,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops=dict(color="tab:blue"),
                whiskerprops=dict(color="tab:blue"),
                capprops=dict(color="tab:blue"),
                medianprops=dict(color="tab:blue", linewidth=1.6),
            )
            ub_box = ax.boxplot(
                ub_data,
                positions=positions + offset,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops=dict(color="tab:orange"),
                whiskerprops=dict(color="tab:orange"),
                capprops=dict(color="tab:orange"),
                medianprops=dict(color="tab:orange", linewidth=1.6),
            )
            for box in lb_box["boxes"]:
                box.set_facecolor("tab:blue")
                box.set_alpha(0.22)
            for box in ub_box["boxes"]:
                box.set_facecolor("tab:orange")
                box.set_alpha(0.22)

            ax.plot(positions, r["LB_median"], color="tab:blue", marker="o", lw=1.8)
            ax.plot(positions, r["UB_median"], color="tab:orange", marker="s", lw=1.8)
            ax.set_title(f"lambda={lambda_:g}")
            ax.set_ylim(ymin_bounds - margin, ymax_bounds + margin)
            ax.set_xlabel("Number of sensors")
            ax.set_xticks(sensor_budgets)
            ax.grid(True, alpha=0.3)
            if col % n_plot_cols == 0:
                ax.set_ylabel("Value")
            if col == 0:
                legend_handles = [
                    Line2D([0], [0], color="tab:blue", marker="o", lw=1.8, label="LB median"),
                    Line2D([0], [0], color="tab:orange", marker="s", lw=1.8, label="UB median"),
                ]
                ax.legend(
                    handles=legend_handles,
                    fontsize=9,
                    title=f"Boxes: {n_repeats} repeats",
                    loc="best",
                )

        for ax in axes[n_lambda:]:
            ax.axis("off")

        fig.suptitle(
            f"Impact of lambda on EIG bounds (boxplots, {n_repeats} repeats) - {name}",
            fontsize=13,
        )
        plt.tight_layout()

        out_pdf = os.path.join(args.output_dir, f"lambda_bounds_boxplot_{mu_slug}.pdf")
        out_png = os.path.join(args.output_dir, f"lambda_bounds_boxplot_{mu_slug}.png")
        plt.savefig(out_pdf)
        plt.savefig(out_png, dpi=180)
        plt.close(fig)
        saved_files.extend([out_pdf, out_png])

        # Gap figure for the same mu: median + interquartile band across repeats
        fig_gap, ax_gap = plt.subplots(1, 1, figsize=(7.6, 4.6))
        for lambda_ in lambda_values:
            r = case["by_lambda"][lambda_]
            ax_gap.plot(sensor_budgets, r["GAP_median"], marker="o", label=f"lambda={lambda_:g}")
            ax_gap.fill_between(sensor_budgets, r["GAP_q1"], r["GAP_q3"], alpha=0.18)
        ax_gap.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax_gap.set_xlabel("Number of sensors")
        ax_gap.set_ylabel("Gap (UB - LB)")
        ax_gap.set_title(f"Bounds gap across lambda ({n_repeats} repeats, median + IQR) - {name}")
        ax_gap.grid(True, alpha=0.3)
        ax_gap.legend(fontsize=9)
        plt.tight_layout()

        out_gap_pdf = os.path.join(args.output_dir, f"lambda_gap_iqr_{mu_slug}.pdf")
        out_gap_png = os.path.join(args.output_dir, f"lambda_gap_iqr_{mu_slug}.png")
        plt.savefig(out_gap_pdf)
        plt.savefig(out_gap_png, dpi=180)
        plt.close(fig_gap)
        saved_files.extend([out_gap_pdf, out_gap_png])

        # Sigma_Y and Sigma_signal heatmaps across lambda for the same mu
        matrix_specs = [
            ("$\\Sigma_Y$ (mean)", "Sigma_Y_mean"),
            ("$\\Sigma_{signal}$ (mean)", "Sigma_signal_mean"),
        ]
        n_rows = len(matrix_specs)
        fig_sigma, axes_sigma = plt.subplots(
            n_rows,
            n_lambda,
            figsize=(4.2 * n_lambda, 3.8 * n_rows),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes_sigma = np.array([axes_sigma])
        if n_lambda == 1:
            axes_sigma = axes_sigma.reshape(n_rows, 1)

        for row, (row_label, key) in enumerate(matrix_specs):
            mats = [case["by_lambda"][lam][key] for lam in lambda_values]
            vmin = min(float(m.min()) for m in mats)
            vmax = max(float(m.max()) for m in mats)
            if abs(vmax - vmin) < 1e-15:
                vmax = vmin + 1e-15

            im_row = None
            for col, lambda_ in enumerate(lambda_values):
                ax = axes_sigma[row, col]
                im_row = ax.imshow(mats[col], cmap="viridis", vmin=vmin, vmax=vmax)
                if row == 0:
                    ax.set_title(f"lambda={lambda_:g}")
                if col == 0:
                    ax.set_ylabel(row_label)
                ax.set_xticks([])
                ax.set_yticks([])

            fig_sigma.colorbar(im_row, ax=axes_sigma[row, :], fraction=0.02, pad=0.01)

        fig_sigma.suptitle(
            f"Mean sigma matrices across lambda ({n_repeats} repeats) - {name}",
            fontsize=13,
        )
        out_sigma_pdf = os.path.join(args.output_dir, f"lambda_sigmas_mean_{mu_slug}.pdf")
        out_sigma_png = os.path.join(args.output_dir, f"lambda_sigmas_mean_{mu_slug}.png")
        plt.savefig(out_sigma_pdf)
        plt.savefig(out_sigma_png, dpi=180)
        plt.close(fig_sigma)
        saved_files.extend([out_sigma_pdf, out_sigma_png])

    # Optional quick check: true initial condition vs prior means
    fig_mu, ax_mu = plt.subplots(1, 1, figsize=(8.2, 4.6))
    ax_mu.plot(x_grid, u0_true, color="black", lw=2.0, label="true u0")
    for name, mu_vec, _ in mu_configs:
        ax_mu.plot(x_grid, mu_vec, lw=1.8, ls="--", label=name)
    ax_mu.set_xlabel("x")
    ax_mu.set_ylabel("Amplitude")
    ax_mu.set_title("Initial conditions: true u0 and mu_prior")
    ax_mu.grid(True, alpha=0.3)
    ax_mu.legend(fontsize=9)
    plt.tight_layout()
    out_mu_pdf = os.path.join(args.output_dir, "mu_prior_profiles.pdf")
    out_mu_png = os.path.join(args.output_dir, "mu_prior_profiles.png")
    plt.savefig(out_mu_pdf)
    plt.savefig(out_mu_png, dpi=180)
    plt.close(fig_mu)
    saved_files.extend([out_mu_pdf, out_mu_png])

    print("\nSaved:")
    for path in saved_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
