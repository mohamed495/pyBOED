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
from boed.core.noise import ColoredNoise
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
    "mode": "conservative",
    "n_repeats": 5,
    "base_seed": 42,
    "refresh_plots_each_setup": True,
    "max_setups": None,
    "output_dir": "results/bounds_go",
    "incremental_sigma_source": "signal",
    "sensor_budgets": [5, 10, 15, 20, 25, 30],
    "base_setup": {
        "N": 100,
        "dt": 0.001,
        "n_steps": 100,
        "diffusivity": 0.02,
        "sigma": 0.05,
        "correlation_length": 20.3,
        "lambda_": 0.5,
        "kernel_length_scale": 0.2,
        "kernel_sigma": 1.0,
        "u0_center": 0.25,
        "u0_width": 0.07,
        "u0_amplitude": 1.5,
        # Lower these Monte Carlo counts if you want a quick local test.
        "n_samples": 500,
        "n_samples_sigma_y": 500,
        "n_theta_cov": 500,
        "n_eta_cov": 500,
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


def sample_joint_prior(joint_prior, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    d = joint_prior["d"]
    z = rng.multivariate_normal(
        mean=joint_prior["mu"],
        cov=joint_prior["Sigma"],
        size=n_samples,
    )
    return z[:, :d], z[:, d:]


def make_forward_model(pde_model, n_steps):
    def G(theta, eta):
        u0 = np.concatenate((theta, eta), axis=None) if eta is not None else theta
        return pde_model.evolve(u0=u0, n_steps=n_steps)[-1, :]

    return G


def jacobian_fd_theta(G, theta, eta, h=None):
    d = theta.shape[0]
    if h is None:
        h = np.finfo(float).eps ** (1 / 3) * (la.norm(theta) + 1e-8)

    m = G(theta, eta).shape[0]
    J = np.zeros((m, d))
    for j in range(d):
        e_j = np.zeros(d)
        e_j[j] = 1.0
        J[:, j] = (G(theta + h * e_j, eta) - G(theta - h * e_j, eta)) / (2 * h)
    return J


def jacobian_fd_eta(G, theta, eta, h=None):
    q = eta.shape[0]
    if h is None:
        h = np.finfo(float).eps ** (1 / 3) * (la.norm(eta) + 1e-8)

    m = G(theta, eta).shape[0]
    J = np.zeros((m, q))
    for j in range(q):
        e_j = np.zeros(q)
        e_j[j] = 1.0
        J[:, j] = (G(theta, eta + h * e_j) - G(theta, eta - h * e_j)) / (2 * h)
    return J


def estimate_E_JT(G, joint_prior, n_samples, rng=None):
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)

    EJ_theta_T_sum = None
    EJ_eta_T_sum = None
    for k in range(n_samples):
        J_theta_T = jacobian_fd_theta(G, theta_samples[k], eta_samples[k]).T
        J_eta_T = jacobian_fd_eta(G, theta_samples[k], eta_samples[k]).T

        if EJ_theta_T_sum is None:
            EJ_theta_T_sum = np.zeros_like(J_theta_T, dtype=float)
            EJ_eta_T_sum = np.zeros_like(J_eta_T, dtype=float)

        EJ_theta_T_sum += J_theta_T
        EJ_eta_T_sum += J_eta_T

    EJ_theta_T = EJ_theta_T_sum / n_samples
    EJ_eta_T = EJ_eta_T_sum / n_samples
    EJ_full_T = np.concatenate((EJ_theta_T, EJ_eta_T), axis=0)
    return {
        "EJ_theta_T": EJ_theta_T,
        "EJ_eta_T": EJ_eta_T,
        "EJ_full_T": EJ_full_T,
    }


def estimate_jacobian_covariances_mc(
    G,
    joint_prior,
    Sigma_obs,
    n_samples,
    h_theta=None,
    h_eta=None,
    unbiased=False,
    rng=None,
):
    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    evals, evecs = la.eigh(Sigma_obs)
    if np.any(evals <= 0):
        raise ValueError("Sigma_obs must be symmetric positive definite.")
    Sigma_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    Z_theta_list = []
    Z_eta_list = []
    for k in range(n_samples):
        J_theta = jacobian_fd_theta(G, theta_samples[k], eta_samples[k], h=h_theta)
        J_eta = jacobian_fd_eta(G, theta_samples[k], eta_samples[k], h=h_eta)
        Z_theta_list.append(J_theta.T @ Sigma_inv_sqrt)
        Z_eta_list.append(J_eta.T @ Sigma_inv_sqrt)

    Z_theta_arr = np.asarray(Z_theta_list, dtype=float)
    Z_eta_arr = np.asarray(Z_eta_list, dtype=float)
    Z_theta_mean = np.mean(Z_theta_arr, axis=0)
    Z_eta_mean = np.mean(Z_eta_arr, axis=0)

    d_theta = Z_theta_mean.shape[0]
    d_eta = Z_eta_mean.shape[0]
    Cov_theta = np.zeros((d_theta, d_theta))
    Cov_eta = np.zeros((d_eta, d_eta))
    Cov_theta_eta = np.zeros((d_theta, d_eta))

    for k in range(n_samples):
        Dt = Z_theta_arr[k] - Z_theta_mean
        De = Z_eta_arr[k] - Z_eta_mean
        Cov_theta += Dt @ Dt.T
        Cov_eta += De @ De.T
        Cov_theta_eta += Dt @ De.T

    denom = (n_samples - 1) if (unbiased and n_samples > 1) else n_samples
    Cov_theta /= denom
    Cov_eta /= denom
    Cov_theta_eta /= denom

    Cov_theta = 0.5 * (Cov_theta + Cov_theta.T)
    Cov_eta = 0.5 * (Cov_eta + Cov_eta.T)
    Cov_full = np.block(
        [
            [Cov_theta, Cov_theta_eta],
            [Cov_theta_eta.T, Cov_eta],
        ]
    )
    return {
        "Cov_theta": Cov_theta,
        "Cov_eta": Cov_eta,
        "Cov_theta_eta": Cov_theta_eta,
        "Cov_full": Cov_full,
    }


def estimate_Sigma_Y(G, joint_prior, Sigma_obs, n_samples, rng=None):
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    U = np.asarray([G(theta_samples[k], eta_samples[k]) for k in range(n_samples)])
    SY = np.asarray(Sigma_obs, dtype=float) + np.cov(U, rowvar=False, bias=False)
    return 0.5 * (SY + SY.T)


def compute_Sigma_noise(L_eta, H_eta, Sigma_eta_given_theta, Sigma_obs):
    A = la.inv(Sigma_eta_given_theta) + H_eta
    X = la.solve(A, L_eta)
    return np.asarray(Sigma_obs) + L_eta.T @ X


def compute_Sigma_signal(l_theta, H_theta, Sigma_theta, Sigma_obs):
    A = la.inv(Sigma_theta) + H_theta
    X = la.solve(A, l_theta)
    return np.asarray(Sigma_obs) + l_theta.T @ X


def solve_linear_matrix_regression_minimization(
    G,
    joint_prior,
    Sigma_obs,
    n_samples=10000,
    random_state=0,
):
    Sigma_obs = np.asarray(Sigma_obs)
    rng = np.random.default_rng(random_state)
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)

    X = np.asarray([G(theta_samples[k], eta_samples[k]) for k in range(n_samples)])
    if X.ndim != 2:
        raise ValueError("u(theta, eta) must return a 1D array with fixed length p.")

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


def compute_incremental_sigma_signal_free(
    G,
    joint_prior,
    Sigma_obs,
    n_samples,
    random_state,
):
    _, _, _, M_formula, _ = solve_linear_matrix_regression_minimization(
        G=G,
        joint_prior=joint_prior,
        Sigma_obs=Sigma_obs,
        n_samples=int(n_samples),
        random_state=random_state,
    )
    Sobs_inv = np.linalg.inv(Sigma_obs)
    Iy = Sobs_inv @ (np.eye(Sigma_obs.shape[0]) - M_formula @ Sobs_inv)
    Sigma_signal_free = np.linalg.inv(Iy)
    return 0.5 * (Sigma_signal_free + Sigma_signal_free.T)


def estimate_E_cov_Y_given_theta(G, joint_prior, Sigma_obs, n_theta, n_eta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_eta < 2:
        raise ValueError("n_eta must be >= 2 to estimate a covariance.")

    d = joint_prior["d"]
    q = joint_prior["q"]
    mu = joint_prior["mu"]
    Sigma = joint_prior["Sigma"]

    mu_theta = mu[:d]
    mu_eta = mu[d:]
    Sigma_theta = Sigma[:d, :d]
    Sigma_eta = Sigma[d:, d:]
    Sigma_et = Sigma[d:, :d]

    Sigma_theta_inv_Sigma_te = la.solve(Sigma_theta, Sigma_et.T)
    Sigma_eta_given_theta = Sigma_eta - Sigma_et @ Sigma_theta_inv_Sigma_te
    Sigma_eta_given_theta = 0.5 * (
        Sigma_eta_given_theta + Sigma_eta_given_theta.T
    )
    Sigma_eta_given_theta += 1e-12 * np.eye(q)

    theta_samples = rng.multivariate_normal(
        mean=mu_theta,
        cov=Sigma_theta,
        size=n_theta,
    )
    inner_cov_list = []
    for i in range(n_theta):
        mu_eta_given_theta_i = mu_eta + Sigma_et @ la.solve(
            Sigma_theta,
            theta_samples[i] - mu_theta,
        )
        eta_cond = rng.multivariate_normal(
            mean=mu_eta_given_theta_i,
            cov=Sigma_eta_given_theta,
            size=n_eta,
        )
        values = np.asarray([G(theta_samples[i], eta_cond[j]) for j in range(n_eta)])
        if values.ndim == 1:
            values = values[:, None]
        inner_cov_list.append(np.cov(values, rowvar=False, ddof=1))

    result = np.asarray(Sigma_obs) + np.mean(np.asarray(inner_cov_list), axis=0)
    return 0.5 * (result + result.T)


def build_go_objects(setup):
    N = int(setup["N"])
    if N % 2 != 0:
        raise ValueError("GO experiments require an even N so theta and eta split evenly.")

    x_grid = np.linspace(0.0, 1.0, N + 2)[1:-1]
    u0_true = make_u0(
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
    noise = ColoredNoise(
        sigma_noise=float(setup["sigma"]),
        correlation_length=float(setup["correlation_length"]),
    )
    Sigma_obs = noise.get_covariance(N)

    d = q = N // 2
    joint_prior = {
        "mu": np.asarray(prior.mu),
        "Sigma": np.asarray(prior.Sigma),
        "d": d,
        "q": q,
    }
    Sigma_theta = joint_prior["Sigma"][:d, :d]
    Sigma_eta = joint_prior["Sigma"][d:, d:]
    Sigma_theta_eta = joint_prior["Sigma"][:d, d:]
    Sigma_eta_theta = joint_prior["Sigma"][d:, :d]
    Sigma_eta_given_theta = Sigma_eta - Sigma_eta_theta @ la.solve(
        Sigma_theta,
        Sigma_theta_eta,
    )
    Sigma_eta_given_theta = 0.5 * (
        Sigma_eta_given_theta + Sigma_eta_given_theta.T
    )

    theta_true, eta_true = np.split(u0_true, 2)
    G = make_forward_model(model, int(setup["n_steps"]))
    G(theta_true, eta_true)

    return {
        "N": N,
        "prior": prior,
        "joint_prior": joint_prior,
        "Sigma_obs": Sigma_obs,
        "Sigma_eta_given_theta": Sigma_eta_given_theta,
        "G": G,
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
    objects = build_go_objects(setup)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    res_l = estimate_E_JT(
        G=objects["G"],
        joint_prior=objects["joint_prior"],
        n_samples=int(setup["n_samples"]),
        rng=rng,
    )
    res_H = estimate_jacobian_covariances_mc(
        G=objects["G"],
        joint_prior=objects["joint_prior"],
        Sigma_obs=objects["Sigma_obs"],
        n_samples=int(setup["n_samples"]),
        rng=rng,
    )
    Sigma_Y = estimate_Sigma_Y(
        G=objects["G"],
        joint_prior=objects["joint_prior"],
        Sigma_obs=objects["Sigma_obs"],
        n_samples=int(setup["n_samples_sigma_y"]),
        rng=rng,
    )
    Sigma_noise = compute_Sigma_noise(
        L_eta=res_l["EJ_eta_T"],
        H_eta=res_H["Cov_eta"],
        Sigma_eta_given_theta=objects["Sigma_eta_given_theta"],
        Sigma_obs=objects["Sigma_obs"],
    )
    Sigma_signal = compute_Sigma_signal(
        l_theta=res_l["EJ_full_T"],
        H_theta=res_H["Cov_full"],
        Sigma_theta=np.asarray(objects["prior"].Sigma),
        Sigma_obs=objects["Sigma_obs"],
    )
    Sigma_Y_given_theta = estimate_E_cov_Y_given_theta(
        G=objects["G"],
        joint_prior=objects["joint_prior"],
        Sigma_obs=objects["Sigma_obs"],
        n_theta=int(setup["n_theta_cov"]),
        n_eta=int(setup["n_eta_cov"]),
        rng=rng,
    )
    elapsed_phase_s = time.perf_counter() - start

    max_budget = sensor_budgets[-1]
    indices_path = greedy_maximize_lb(
        Sigma_Y=Sigma_Y,
        Sigma_noise=Sigma_noise,
        n_sensors=max_budget,
    )

    lb_values = []
    ub_values = []
    for budget in sensor_budgets:
        W_opt, _ = build_selection_matrices(objects["N"], indices_path[:budget])
        lb_values.append(eig_BI(Sigma_Y, Sigma_noise, W_opt))
        ub_values.append(eig_BS(Sigma_signal, Sigma_Y_given_theta, W_opt))

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
        "G": objects["G"],
        "joint_prior": objects["joint_prior"],
        "Sigma_obs": objects["Sigma_obs"],
        "Sigma_signal": Sigma_signal,
        "Sigma_Y_given_theta": Sigma_Y_given_theta,
        "Sigma_Y": Sigma_Y,
        "Sigma_noise": Sigma_noise,
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
            joint_prior=context["joint_prior"],
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
        Sigma_Y_theta=context["Sigma_Y_given_theta"],
        Sigma_Y=context["Sigma_Y"],
        Sigma_noise=context["Sigma_noise"],
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
        description="Repeated conservative/incremental bounds runs for the GO Burgers tutorial.",
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
            "'signal' reproduces the current GO behavior; "
            "'free' uses the regression-based construction from the notebook."
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
            "script": "run_bounds_go.py",
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
