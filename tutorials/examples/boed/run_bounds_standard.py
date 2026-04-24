from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyboed_mpl")

import numpy as np
import numpy.linalg as la

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boed.core import make_u0
from boed.core.noise import NoiseModel
from boed.pde.burgers import Burgers_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import Matern32
from boed.utils.observation import build_selection_matrices

from bounds_sweep_utils import (
    append_rows,
    build_setups,
    eig_BI,
    eig_BS,
    greedy_maximize_lb,
    incremental_bounds,
    initialize_results_file,
    make_seed,
    refresh_outputs,
    sanitize_sensor_budgets,
    write_run_config,
)


DEFAULT_RUN_CONFIG = {
    "mode": "incremental",
    "n_repeats": 5,
    "base_seed": 42,
    "refresh_plots_each_setup": True,
    "max_setups": None,
    "output_dir": "results/bounds_standard",
    "incremental_sigma_source": "free",
    "sensor_budgets": [5, 10, 15, 20, 25, 30],
    "base_setup": {
        "N": 100,
        "dt": 0.001,
        "n_steps": 100,
        "diffusivity": 0.02,
        "sigma": 1.0,
        "lambda_": 0.25,
        "kernel_length_scale": 0.2,
        "kernel_sigma": 1.0,
        "u0_center": 0.25,
        "u0_width": 0.07,
        "u0_amplitude": 1.5,
        # Lower these Monte Carlo counts if you want a quick local test.
        "n_samples": 100,
        "n_samples_sigma_y": 500,
        "n_samples_regression": 2000,
    },
    "sweep": {
        "lambda_": [0.0, 0.25, 0.5, 1.0],
    },
}

INCREMENTAL_SIGMA_SOURCES = ("free", "signal")


def make_prior(kernel_obj, nx, mu):
    try:
        return GaussianProcessPrior(kernel_obj, mu=mu, nx=nx)
    except TypeError:
        prior = GaussianProcessPrior(kernel_obj, nx=nx)
        prior.mu = mu
        return prior


def make_forward_model(pde_model, n_steps):
    def G(theta):
        return pde_model.evolve(u0=theta, n_steps=n_steps)[-1, :]

    return G


def jacobian_fd(G, theta, h=None):
    d = theta.shape[0]
    if h is None:
        h = np.finfo(float).eps ** (1 / 3) * (la.norm(theta) + 1e-8)

    G_theta = G(theta)
    m = G_theta.shape[0]
    J = np.zeros((m, d))
    for j in range(d):
        e_j = np.zeros(d)
        e_j[j] = 1.0
        J[:, j] = (G(theta + h * e_j) - G_theta) / h
    return J


def estimate_E_JT(G, prior, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    theta_samples = rng.multivariate_normal(
        mean=np.asarray(prior.mu),
        cov=np.asarray(prior.Sigma),
        size=n_samples,
    )
    J0 = jacobian_fd(G, theta_samples[0])
    d = theta_samples.shape[1]
    m = J0.shape[0]
    EJ_theta_T_sum = np.zeros((d, m))
    EJ_theta_T_sum += J0.T

    for k in range(1, n_samples):
        EJ_theta_T_sum += jacobian_fd(G, theta_samples[k]).T

    return EJ_theta_T_sum / n_samples


def estimate_jacobian_covariances_mc(
    G,
    prior,
    Sigma_obs,
    n_samples,
    h=None,
    unbiased=False,
    rng=None,
):
    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    evals, evecs = la.eigh(Sigma_obs)
    if np.any(evals <= 0):
        raise ValueError("Sigma_obs must be symmetric positive definite.")
    Sigma_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    if rng is None:
        rng = np.random.default_rng()

    theta_samples = rng.multivariate_normal(
        mean=np.asarray(prior.mu),
        cov=np.asarray(prior.Sigma),
        size=n_samples,
    )

    Z_theta_list = []
    for k in range(n_samples):
        J_theta = jacobian_fd(G, theta_samples[k], h=h)
        Z_theta_list.append(J_theta.T @ Sigma_inv_sqrt)

    Z_theta_arr = np.asarray(Z_theta_list, dtype=float)
    Z_theta_mean = np.mean(Z_theta_arr, axis=0)
    d_theta = Z_theta_mean.shape[0]
    Cov_theta = np.zeros((d_theta, d_theta))

    for k in range(n_samples):
        Dt = Z_theta_arr[k] - Z_theta_mean
        Cov_theta += Dt @ Dt.T

    denom = (n_samples - 1) if (unbiased and n_samples > 1) else n_samples
    Cov_theta /= denom
    return 0.5 * (Cov_theta + Cov_theta.T)


def estimate_Sigma_Y(G, prior, Sigma_obs, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    theta_samples = rng.multivariate_normal(
        mean=np.asarray(prior.mu),
        cov=np.asarray(prior.Sigma),
        size=n_samples,
    )
    U = np.asarray([G(theta_samples[k]) for k in range(n_samples)])
    SY = np.asarray(Sigma_obs, dtype=float) + np.cov(U, rowvar=False, bias=False)
    return 0.5 * (SY + SY.T)


def compute_Sigma_signal(l_theta, H_theta, Sigma_theta, Sigma_obs):
    A = la.inv(Sigma_theta) + H_theta
    X = la.solve(A, l_theta)
    return np.asarray(Sigma_obs) + l_theta.T @ X


def solve_linear_matrix_regression_minimization(
    G,
    prior,
    Sigma_obs,
    n_samples=10000,
    random_state=0,
):
    rng = np.random.default_rng(random_state)
    Sigma_obs = np.asarray(Sigma_obs)
    theta = rng.multivariate_normal(
        np.asarray(prior.mu),
        np.asarray(prior.Sigma),
        size=n_samples,
    )

    X = np.asarray([G(th) for th in theta])
    if X.ndim != 2:
        raise ValueError("u(theta) must return a 1D array with fixed length p.")

    n, p = X.shape
    if Sigma_obs.shape != (p, p):
        raise ValueError(f"Sigma_obs must have shape ({p}, {p}).")

    eps = rng.multivariate_normal(np.zeros(p), Sigma_obs, size=n_samples)
    Y = X + eps

    m_X = X.mean(axis=0)
    m_Y = Y.mean(axis=0)
    Xc = X - m_X
    Yc = Y - m_Y

    Sigma_XY = (Xc.T @ Yc) / n
    Sigma_YY = (Yc.T @ Yc) / n
    Sigma_X = (Xc.T @ Xc) / n

    A_star = Sigma_XY @ np.linalg.inv(Sigma_YY)
    b_star = m_X - A_star @ m_Y
    R = X - (Y @ A_star.T + b_star)
    M_emp = (R.T @ R) / n
    M_formula = Sigma_X - Sigma_X @ np.linalg.inv(Sigma_X + Sigma_obs) @ Sigma_X
    stats = {
        "m_X": m_X,
        "m_Y": m_Y,
        "Sigma_X": Sigma_X,
        "Sigma_XY": Sigma_XY,
        "Sigma_YY": Sigma_YY,
        "X": X,
        "Y": Y,
        "residuals": R,
    }
    return A_star, b_star, M_emp, M_formula, stats


def compute_incremental_sigma_signal_free(G, prior, Sigma_obs, n_samples, random_state):
    _, _, _, M_formula, _ = solve_linear_matrix_regression_minimization(
        G=G,
        prior=prior,
        Sigma_obs=Sigma_obs,
        n_samples=int(n_samples),
        random_state=random_state,
    )
    Sobs_inv = np.linalg.inv(Sigma_obs)
    Iy = Sobs_inv @ (np.eye(Sigma_obs.shape[0]) - M_formula @ Sobs_inv)
    Sigma_signal_free = np.linalg.inv(Iy)
    return 0.5 * (Sigma_signal_free + Sigma_signal_free.T)


def build_standard_objects(setup):
    N = int(setup["N"])
    x_grid = np.linspace(0.0, 1.0, N + 2)[1:-1]
    make_u0(
        x_grid,
        "gaussian",
        center=float(setup["u0_center"]),
        width=float(setup["u0_width"]),
        amplitude=float(setup["u0_amplitude"]),
    )

    model = Burgers_CN(
        N=N,
        dt=float(setup["dt"]),
        diffusivity=float(setup["diffusivity"]),
        lambda_=float(setup["lambda_"]),
    )
    kernel = Matern32(
        length_scale=float(setup["kernel_length_scale"]),
        sigma=float(setup["kernel_sigma"]),
    )
    prior = make_prior(kernel, N, np.zeros(N))
    noise = NoiseModel(sigma_noise=float(setup["sigma"]))
    Sigma_obs = noise.get_covariance(N)

    return {
        "N": N,
        "prior": prior,
        "Sigma_obs": Sigma_obs,
        "G": make_forward_model(model, int(setup["n_steps"])),
    }


def build_phase_rows(
    *,
    run_mode,
    phase,
    incremental_sigma_source,
    setup_meta,
    repeat_index,
    seed,
    budgets,
    lb_values,
    ub_values,
    elapsed_phase_s,
):
    rows = []
    setup = setup_meta["setup"]
    for budget, lb, ub in zip(budgets, lb_values, ub_values):
        row = {
            "mode": run_mode,
            "phase": phase,
            "setup_id": setup_meta["setup_id"],
            "setup_label": setup_meta["setup_label"],
            "repeat": repeat_index,
            "seed": seed,
            "budget": budget,
            "incremental_sigma_source": incremental_sigma_source,
            "lb": float(lb),
            "ub": float(ub),
            "gap": float(ub - lb),
            "elapsed_phase_s": float(elapsed_phase_s),
        }
        for name, value in setup.items():
            row[name] = value
        rows.append(row)
    return rows


def run_conservative_repeat(
    setup_meta,
    run_mode,
    repeat_index,
    seed,
    sensor_budgets,
    incremental_sigma_source,
):
    setup = setup_meta["setup"]
    objects = build_standard_objects(setup)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    l_theta = estimate_E_JT(
        G=objects["G"],
        prior=objects["prior"],
        n_samples=int(setup["n_samples"]),
        rng=rng,
    )
    H_theta = estimate_jacobian_covariances_mc(
        G=objects["G"],
        prior=objects["prior"],
        Sigma_obs=objects["Sigma_obs"],
        n_samples=int(setup["n_samples"]),
        rng=rng,
    )
    Sigma_Y = estimate_Sigma_Y(
        G=objects["G"],
        prior=objects["prior"],
        Sigma_obs=objects["Sigma_obs"],
        n_samples=int(setup["n_samples_sigma_y"]),
        rng=rng,
    )
    Sigma_signal = compute_Sigma_signal(
        l_theta=l_theta,
        H_theta=H_theta,
        Sigma_theta=np.asarray(objects["prior"].Sigma),
        Sigma_obs=objects["Sigma_obs"],
    )

    elapsed_phase_s = time.perf_counter() - start
    max_budget = sensor_budgets[-1]
    indices_path = greedy_maximize_lb(
        Sigma_Y=Sigma_Y,
        Sigma_noise=objects["Sigma_obs"],
        n_sensors=max_budget,
    )

    lb_values = []
    ub_values = []
    for budget in sensor_budgets:
        W_opt, _ = build_selection_matrices(objects["N"], indices_path[:budget])
        lb_values.append(eig_BI(Sigma_Y, objects["Sigma_obs"], W_opt))
        ub_values.append(eig_BS(Sigma_signal, objects["Sigma_obs"], W_opt))

    rows = build_phase_rows(
        run_mode=run_mode,
        phase="conservative",
        incremental_sigma_source=incremental_sigma_source,
        setup_meta=setup_meta,
        repeat_index=repeat_index,
        seed=seed,
        budgets=sensor_budgets,
        lb_values=lb_values,
        ub_values=ub_values,
        elapsed_phase_s=elapsed_phase_s,
    )
    context = {
        "setup_meta": setup_meta,
        "repeat_index": repeat_index,
        "seed": seed,
        "N": objects["N"],
        "G": objects["G"],
        "prior": objects["prior"],
        "Sigma_obs": objects["Sigma_obs"],
        "Sigma_signal": Sigma_signal,
        "Sigma_Y": Sigma_Y,
        "incremental_sigma_source": incremental_sigma_source,
    }
    return rows, context


def run_incremental_repeat(context, run_mode, sensor_budgets):
    start = time.perf_counter()
    setup = context["setup_meta"]["setup"]
    sigma_source = context["incremental_sigma_source"]
    if sigma_source == "signal":
        incremental_sigma_signal = context["Sigma_signal"]
    elif sigma_source == "free":
        incremental_sigma_signal = compute_incremental_sigma_signal_free(
            G=context["G"],
            prior=context["prior"],
            Sigma_obs=context["Sigma_obs"],
            n_samples=setup["n_samples_regression"],
            random_state=context["seed"],
        )
    else:
        raise ValueError(
            f"Unsupported incremental sigma source: {sigma_source!r}. "
            f"Expected one of {INCREMENTAL_SIGMA_SOURCES}."
        )

    result = incremental_bounds(
        Sigma_signal=incremental_sigma_signal,
        Sigma_Y_theta=context["Sigma_obs"],
        Sigma_Y=context["Sigma_Y"],
        Sigma_noise=context["Sigma_obs"],
        n_sensors=sensor_budgets[-1],
    )
    elapsed_phase_s = time.perf_counter() - start

    lb_values = [result["scores_inf"][budget - 1] for budget in sensor_budgets]
    ub_values = [result["scores_sup"][budget - 1] for budget in sensor_budgets]
    return build_phase_rows(
        run_mode=run_mode,
        phase="incremental",
        incremental_sigma_source=sigma_source,
        setup_meta=context["setup_meta"],
        repeat_index=context["repeat_index"],
        seed=context["seed"],
        budgets=sensor_budgets,
        lb_values=lb_values,
        ub_values=ub_values,
        elapsed_phase_s=elapsed_phase_s,
    )


def build_config_from_args(args):
    config = copy.deepcopy(DEFAULT_RUN_CONFIG)
    config["mode"] = args.mode
    config["incremental_sigma_source"] = args.incremental_sigma_source
    if args.n_repeats is not None:
        config["n_repeats"] = args.n_repeats
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    if args.max_setups is not None:
        config["max_setups"] = args.max_setups
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.refresh_plots_each_setup is not None:
        config["refresh_plots_each_setup"] = args.refresh_plots_each_setup
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeated conservative/incremental bounds runs for the standard Burgers tutorial.",
    )
    parser.add_argument(
        "--mode",
        choices=["conservative", "incremental"],
        default=DEFAULT_RUN_CONFIG["mode"],
    )
    parser.add_argument(
        "--incremental-sigma-source",
        choices=INCREMENTAL_SIGMA_SOURCES,
        default=DEFAULT_RUN_CONFIG["incremental_sigma_source"],
        help=(
            "Matrix used as Sigma_signal in incremental mode. "
            "'free' reproduces the current regression-based construction; "
            "'signal' reuses the conservative Sigma_signal."
        ),
    )
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--max-setups", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--refresh-plots-each-setup",
        dest="refresh_plots_each_setup",
        action="store_true",
    )
    parser.add_argument(
        "--no-refresh-plots-each-setup",
        dest="refresh_plots_each_setup",
        action="store_false",
    )
    parser.set_defaults(refresh_plots_each_setup=None)
    return parser.parse_args()


def run_experiments(config):
    if int(config["n_repeats"]) < 1:
        raise ValueError("n_repeats must be >= 1.")
    if config["incremental_sigma_source"] not in INCREMENTAL_SIGMA_SOURCES:
        raise ValueError(
            "incremental_sigma_source must be one of "
            f"{INCREMENTAL_SIGMA_SOURCES}."
        )

    setups = build_setups(config["base_setup"], config["sweep"])
    if config["max_setups"] is not None:
        setups = setups[: int(config["max_setups"])]
    if not setups:
        raise ValueError("No setup available to run.")

    output_dir = Path(config["output_dir"]) / str(config["mode"])
    parameter_names = list(config["base_setup"].keys())
    output_path, raw_csv_path, fieldnames = initialize_results_file(
        output_dir=output_dir,
        parameter_names=parameter_names,
        metadata_fields=["incremental_sigma_source"],
    )
    write_run_config(
        output_path,
        {
            "script": "run_bounds_standard.py",
            "resolved_output_dir": str(output_path),
            **config,
        },
    )

    print(f"Running {len(setups)} setup(s) in {output_path}")
    for setup_index, setup_meta in enumerate(setups):
        sensor_budgets = sanitize_sensor_budgets(
            config["sensor_budgets"],
            n_grid=int(setup_meta["setup"]["N"]),
        )
        print(
            f"[{setup_index + 1}/{len(setups)}] {setup_meta['setup_label']} | "
            f"budgets={sensor_budgets}"
        )

        conservative_rows = []
        contexts = []
        for repeat_index in range(int(config["n_repeats"])):
            seed = make_seed(config["base_seed"], setup_index, repeat_index)
            rows, context = run_conservative_repeat(
                setup_meta=setup_meta,
                run_mode=config["mode"],
                repeat_index=repeat_index,
                seed=seed,
                sensor_budgets=sensor_budgets,
                incremental_sigma_source=config["incremental_sigma_source"],
            )
            conservative_rows.extend(rows)
            contexts.append(context)

        append_rows(raw_csv_path, fieldnames, conservative_rows)
        if config["refresh_plots_each_setup"]:
            refresh_outputs(output_path, parameter_names)

        if config["mode"] == "incremental":
            incremental_rows = []
            for context in contexts:
                incremental_rows.extend(
                    run_incremental_repeat(
                        context=context,
                        run_mode=config["mode"],
                        sensor_budgets=sensor_budgets,
                    )
                )
            append_rows(raw_csv_path, fieldnames, incremental_rows)
            if config["refresh_plots_each_setup"]:
                refresh_outputs(output_path, parameter_names)

    if not config["refresh_plots_each_setup"]:
        refresh_outputs(output_path, parameter_names)

    print(f"Done. Results saved in {output_path}")


def main():
    args = parse_args()
    config = build_config_from_args(args)
    run_experiments(config)


if __name__ == "__main__":
    main()
