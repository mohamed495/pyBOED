"""
run_bounds_GO.py
----------------
Répète l'expérience du tuto_bounds_GO pour plusieurs valeurs de lambda_,
plusieurs seeds, et produit deux figures :

  - Figure 1 : sélection incrémentale  (incremental_bounds)
  - Figure 2 : sélection conservative  (greedy_maximize_LB)

Chaque figure contient 4 subplots (un par lambda) avec les boxplots des 4 bornes
(conservative LB/UB, incremental LB/UB) en fonction du budget.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyboed_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from joblib import Parallel, delayed

# --- ajout du repo au path si nécessaire ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from boed.core.noise import NoiseModel
from boed.pde.burgers import Burgers_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import Matern32
from boed.utils.observation import build_selection_matrices

# ============================================================
# Paramètres de l'expérience
# ============================================================

LAMBDAS        = [0.0, 0.25, 0.5, 1.0]
SENSOR_BUDGETS = [5, 10, 15, 20, 25]
N_REPEATS      = 10
BASE_SEED      = 42
N_JOBS         = -1

# Hyperparamètres du modèle (identiques au tuto GO)
N           = 100
DT          = 0.001
N_STEPS     = 100
SIGMA       = 0.01
DIFFUSIVITY = 0.02

KERNEL_LS   = 0.2
KERNEL_SIG  = 1.0

# Nombre d'échantillons MC
N_SAMPLES               = 500
N_SAMPLES_SIGMA_Y       = 2000
N_SAMPLES_Y_GIVEN_THETA = 500
N_SAMPLES_EIG           = 100000
N_ETA_INNER            = 200

OUTPUT_DIR = Path("results_sweep")

# ============================================================
# Monte Carlo helpers
# ============================================================

def sample_joint_prior(joint_prior, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    d = joint_prior["d"]
    z = rng.multivariate_normal(mean=joint_prior["mu"], cov=joint_prior["Sigma"], size=n_samples)
    return z[:, :d], z[:, d:]


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


def _one_sample(G, theta, eta):
    J_theta_T = jacobian_fd_theta(G, theta, eta).T
    J_eta_T = jacobian_fd_eta(G, theta, eta).T
    return J_theta_T, J_eta_T


def estimate_E_JT(G, joint_prior, n_samples, rng=None):
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_one_sample)(G, theta_samples[k], eta_samples[k])
        for k in range(n_samples)
    )
    J_theta_T_0, J_eta_T_0 = results[0]
    EJ_theta_T_sum = np.zeros_like(J_theta_T_0, dtype=float)
    EJ_eta_T_sum = np.zeros_like(J_eta_T_0, dtype=float)
    for Jt, Je in results:
        EJ_theta_T_sum += Jt
        EJ_eta_T_sum += Je
    EJ_theta_T = EJ_theta_T_sum / n_samples
    EJ_eta_T = EJ_eta_T_sum / n_samples
    EJ_full_T = np.concatenate((EJ_theta_T, EJ_eta_T), axis=0)
    return {
        "EJ_theta_T": EJ_theta_T,
        "EJ_eta_T": EJ_eta_T,
        "EJ_full_T": EJ_full_T,
    }


def _compute_z(theta, eta, G, h_theta, h_eta, Sigma_inv_sqrt):
    J_theta = jacobian_fd_theta(G, theta, eta, h=h_theta)
    J_eta = jacobian_fd_eta(G, theta, eta, h=h_eta)
    Z_theta = J_theta.T @ Sigma_inv_sqrt
    Z_eta = J_eta.T @ Sigma_inv_sqrt
    return Z_theta, Z_eta


def estimate_jacobian_covariances_mc(
    G, joint_prior, Sigma_obs, n_samples,
    h_theta=None, h_eta=None, unbiased=False, rng=None, n_jobs=-1,
):
    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    evals, evecs = la.eigh(Sigma_obs)
    if np.any(evals <= 0):
        raise ValueError("Sigma_obs must be symmetric positive definite.")
    Sigma_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_z)(theta_samples[k], eta_samples[k], G, h_theta, h_eta, Sigma_inv_sqrt)
        for k in range(n_samples)
    )
    Z_theta_list, Z_eta_list = zip(*results)
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
    Cov_full = np.block([
        [Cov_theta, Cov_theta_eta],
        [Cov_theta_eta.T, Cov_eta],
    ])
    return {
        "Cov_theta": Cov_theta,
        "Cov_eta": Cov_eta,
        "Cov_theta_eta": Cov_theta_eta,
        "Cov_full": Cov_full,
    }


def estimate_Sigma_Y(G, joint_prior, Sigma_obs, n_samples, rng=None):
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    U = Parallel(n_jobs=N_JOBS)(delayed(G)(theta_samples[k], eta_samples[k]) for k in range(n_samples))
    U = np.asarray(U)
    SY = np.asarray(Sigma_obs) + np.cov(U, rowvar=False, bias=False)
    return 0.5 * (SY + SY.T)


def estimate_E_cov_Y_given_theta(G, joint_prior, Sigma_obs, n_theta, n_eta, rng=None, n_jobs=-1):
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
    Sigma_eta_given_theta = 0.5 * (Sigma_eta_given_theta + Sigma_eta_given_theta.T)
    Sigma_eta_given_theta += 1e-12 * np.eye(q)
    theta_samples = rng.multivariate_normal(mean=mu_theta, cov=Sigma_theta, size=n_theta)
    base_seed = rng.integers(0, 2**30)
    inner_cov_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_theta)(i, theta_samples[i], mu_theta, Sigma_theta, mu_eta,
                                Sigma_et, Sigma_eta_given_theta, G, n_eta, base_seed)
        for i in range(n_theta)
    )
    result = Sigma_obs + np.mean(np.asarray(inner_cov_list), axis=0)
    return 0.5 * (result + result.T)


def _process_theta(i, theta_i, mu_theta, Sigma_theta, mu_eta, Sigma_et, Sigma_eta_given_theta, G, n_eta, base_seed):
    rng = np.random.default_rng(base_seed + i)
    mu_eta_given_theta_i = mu_eta + Sigma_et @ la.solve(Sigma_theta, theta_i - mu_theta)
    eta_cond = rng.multivariate_normal(mean=mu_eta_given_theta_i, cov=Sigma_eta_given_theta, size=n_eta)
    vals = np.asarray([G(theta_i, eta_cond[j]) for j in range(n_eta)])
    if vals.ndim == 1:
        vals = vals[:, None]
    return np.cov(vals, rowvar=False, ddof=1)


def compute_Sigma_signal(l_theta, H_theta, Sigma_theta, Sigma_obs):
    HS = H_theta @ Sigma_theta
    M = Sigma_theta - Sigma_theta @ la.solve(np.eye(Sigma_theta.shape[0]) + HS, HS)
    return Sigma_obs + l_theta.T @ M @ l_theta


def logsumexp_rows(A):
    amax = A.max(axis=1, keepdims=True)
    return amax.squeeze() + np.log(np.exp(A - amax).sum(axis=1))


def estimate_eig_memory_efficient(G, joint_prior, Sigma_obs, n_samples, batch_size=500, seed=None):
    if n_samples < 2:
        raise ValueError("n_samples doit être >= 2")
    rng = np.random.default_rng(seed)
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    U = np.asarray(Parallel(n_jobs=N_JOBS)(delayed(G)(theta_samples[k], eta_samples[k]) for k in range(n_samples)))
    m = U.shape[1]
    Sinv = la.inv(Sigma_obs)
    _, ldet = la.slogdet(Sigma_obs)
    log_const = -0.5 * (ldet + m * np.log(2 * np.pi))
    noise_mat = rng.multivariate_normal(np.zeros(m), Sigma_obs, size=n_samples)
    Y = U + noise_mat
    Y_M = Y @ Sinv
    U_M = U @ Sinv
    normY = np.einsum("ij,ij->i", Y, Y_M)
    normU = np.einsum("ij,ij->i", U, U_M)
    eig_estimate = 0.0
    for i in range(0, n_samples, batch_size):
        i_end = min(i + batch_size, n_samples)
        batch_size_i = i_end - i
        cross_batch = Y_M[i:i_end, :] @ U.T
        normY_batch = normY[i:i_end]
        quad_batch = normY_batch[:, None] + normU[None, :] - 2 * cross_batch
        log_liks_batch = log_const - 0.5 * quad_batch
        log_num_batch = log_liks_batch[np.arange(batch_size_i), i + np.arange(batch_size_i)]
        log_den_batch = logsumexp_rows(log_liks_batch) - np.log(n_samples)
        eig_estimate += (log_num_batch - log_den_batch).sum()
    return eig_estimate / n_samples

# ============================================================
# EIG bounds and selection
# ============================================================

def _log_ratio(A, B, M):
    _, la_ = la.slogdet(M.T @ A @ M)
    _, lb_ = la.slogdet(M.T @ B @ M)
    return 0.5 * (la_ - lb_)


def eig_BI(Sigma_Y, Sigma_noise, W):
    eye = np.eye(Sigma_noise.shape[0])
    return _log_ratio(Sigma_Y, Sigma_noise, W) - _log_ratio(Sigma_Y, Sigma_noise, eye)


def eig_BS(Sigma_signal, Sigma_obs, W):
    eye = np.eye(Sigma_obs.shape[0])
    return _log_ratio(Sigma_signal, Sigma_obs, W) - _log_ratio(Sigma_signal, Sigma_obs, eye)


def greedy_maximize_LB(Sigma_Y, Sigma_noise, n_sensors):
    N_grid = Sigma_Y.shape[0]
    selected = []
    remaining = list(range(N_grid))
    for _ in range(n_sensors):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            S = selected + [idx]
            ix = np.ix_(S, S)
            sign_y, logdet_y = np.linalg.slogdet(Sigma_Y[ix])
            sign_n, logdet_n = np.linalg.slogdet(Sigma_noise[ix])
            if sign_y <= 0 or sign_n <= 0:
                continue
            score = 0.5 * (logdet_y - logdet_n)
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return np.asarray(selected, dtype=int)


def schur(Sigma, Wm):
    if Wm.size == 0:
        return Sigma
    G = Wm.T @ Sigma @ Wm
    return Sigma - Sigma @ Wm @ np.linalg.inv(G) @ Wm.T @ Sigma


def incremental_bounds(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, n_sensors):
    N_grid = Sigma_Y.shape[0]
    candidates = [np.eye(N_grid)[:, i] for i in range(N_grid)]
    selected, remaining = [], list(range(N_grid))
    scores_lb, scores_ub = [], []
    eig_lb = eig_ub = 0.0
    for _ in range(n_sensors):
        Wm = np.column_stack([candidates[i] for i in selected]) if selected else np.empty((N_grid, 0))
        Ss = schur(Sigma_signal,  Wm)
        Syt = schur(Sigma_Y_theta, Wm)
        SY = schur(Sigma_Y,       Wm)
        Sn = schur(Sigma_noise,   Wm)
        best_idx, best_lb, best_ub = None, -np.inf, None
        for idx in remaining:
            w = candidates[idx]
            n_lb = float(w @ Ss @ w)
            d_lb = float(w @ Syt @ w)
            n_ub = float(w @ SY @ w)
            d_ub = float(w @ Sn @ w)
            if min(n_lb, d_lb, n_ub, d_ub) <= 0:
                continue
            dlb = 0.5 * np.log(n_lb / d_lb)
            dub = 0.5 * np.log(n_ub / d_ub)
            if dlb > best_lb:
                best_lb, best_ub, best_idx = dlb, dub, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        eig_lb += best_lb
        eig_ub += best_ub
        scores_lb.append(eig_lb)
        scores_ub.append(eig_ub)
    return np.asarray(selected, dtype=int), scores_lb, scores_ub


def incremental_bounds_given_W(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, W):
    _, ls = np.linalg.slogdet(W.T @ Sigma_signal @ W)
    _, lt = np.linalg.slogdet(W.T @ Sigma_Y_theta @ W)
    _, ly = np.linalg.slogdet(W.T @ Sigma_Y @ W)
    _, ln = np.linalg.slogdet(W.T @ Sigma_noise @ W)
    return 0.5 * (ls - lt), 0.5 * (ly - ln)

# ============================================================
# Modèle GO / prior
# ============================================================

def make_forward_model(pde_model, n_steps):
    def G(theta, eta):
        u0 = np.concatenate((theta, eta), axis=None)
        return pde_model.evolve(u0=u0, n_steps=n_steps)[-1, :]
    return G


def build_objects(lambda_):
    model = Burgers_CN(N=N, dt=DT, diffusivity=DIFFUSIVITY, lambda_=lambda_)
    kernel = Matern32(length_scale=KERNEL_LS, sigma=KERNEL_SIG)
    try:
        prior = GaussianProcessPrior(kernel, mu=np.zeros(N), nx=N)
    except TypeError:
        prior = GaussianProcessPrior(kernel, nx=N)
        prior.mu = np.zeros(N)
    # jitter for numerical stability
    eigvals = np.linalg.eigvalsh(prior.Sigma)
    prior.Sigma += eigvals[-1] / 1000.0
    d = q = N // 2
    joint_prior = {
        "mu": prior.mu,
        "Sigma": prior.Sigma,
        "d": d,
        "q": q,
    }
    noise = NoiseModel(sigma_noise=SIGMA)
    Sigma_obs = noise.get_covariance(N)
    G = make_forward_model(model, n_steps=N_STEPS)
    return prior, joint_prior, Sigma_obs, G

# ============================================================
# Une répétition
# ============================================================

def run_one_repeat(lambda_, seed, eig_offset):
    prior, joint_prior, Sigma_obs, G = build_objects(lambda_)
    rng = np.random.default_rng(seed)
    l_theta = estimate_E_JT(G, joint_prior, N_SAMPLES, rng=rng)["EJ_full_T"]
    H_theta = estimate_jacobian_covariances_mc(
        G=G,
        joint_prior=joint_prior,
        Sigma_obs=Sigma_obs,
        n_samples=N_SAMPLES,
        rng=rng,
        n_jobs=N_JOBS,
    )["Cov_full"]
    Sigma_Y = estimate_Sigma_Y(G, joint_prior, Sigma_obs, N_SAMPLES_SIGMA_Y, rng=rng)
    Sigma_Y_theta = estimate_E_cov_Y_given_theta(
        G=G,
        joint_prior=joint_prior,
        Sigma_obs=Sigma_obs,
        n_theta=N_SAMPLES_Y_GIVEN_THETA,
        n_eta=N_ETA_INNER,
        rng=rng,
        n_jobs=N_JOBS,
    )
    Sigma_signal = compute_Sigma_signal(l_theta, H_theta, prior.Sigma, Sigma_obs)
    max_budget = max(SENSOR_BUDGETS)
    indices_inc, scores_lb, scores_ub = incremental_bounds(
        Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_obs, max_budget
    )
    inc_results = {"cons_lb": [], "cons_ub": [], "inc_lb": [], "inc_ub": []}
    for budget in SENSOR_BUDGETS:
        W, _ = build_selection_matrices(N, indices_inc[:budget])
        inc_results["cons_lb"].append(eig_BI(Sigma_Y, Sigma_obs, W) + eig_offset)
        inc_results["cons_ub"].append(eig_BS(Sigma_signal, Sigma_obs, W) + eig_offset)
        inc_results["inc_lb"].append(scores_lb[budget - 1])
        inc_results["inc_ub"].append(scores_ub[budget - 1])
    indices_cons = greedy_maximize_LB(Sigma_Y, Sigma_obs, max_budget)
    cons_results = {"cons_lb": [], "cons_ub": [], "inc_lb": [], "inc_ub": []}
    for budget in SENSOR_BUDGETS:
        W, _ = build_selection_matrices(N, indices_cons[:budget])
        cons_results["cons_lb"].append(eig_BI(Sigma_Y, Sigma_obs, W) + eig_offset)
        cons_results["cons_ub"].append(eig_BS(Sigma_signal, Sigma_obs, W) + eig_offset)
        lb_inc, ub_inc = incremental_bounds_given_W(
            Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_obs, W
        )
        cons_results["inc_lb"].append(lb_inc)
        cons_results["inc_ub"].append(ub_inc)
    return inc_results, cons_results

# ============================================================
# Plotting
# ============================================================

BOUND_STYLES = {
    "cons_lb": dict(color="tab:blue",   marker="o", linestyle="--", label="Conservative LB"),
    "cons_ub": dict(color="tab:cyan",   marker="s", linestyle="--", label="Conservative UB"),
    "inc_lb":  dict(color="tab:orange", marker="o", linestyle="-",  label="Incremental LB"),
    "inc_ub":  dict(color="tab:red",    marker="s", linestyle="-",  label="Incremental UB"),
}


def _fill_common_region(ax, positions, lb_common, ub_common, mask):
    from matplotlib.patches import Patch
    label = "Common certified region"
    handle = Patch(
        facecolor="tab:green",
        edgecolor="tab:green",
        alpha=0.38,
        hatch="//",
        label=label,
    )
    positions = np.asarray(positions, dtype=float)
    lb_common = np.asarray(lb_common, dtype=float)
    ub_common = np.asarray(ub_common, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(positions) == 1:
        if mask[0]:
            width = 0.35
            ax.fill_between(
                [positions[0] - width, positions[0] + width],
                [lb_common[0], lb_common[0]],
                [ub_common[0], ub_common[0]],
                facecolor="tab:green",
                edgecolor="tab:green",
                linewidth=0.0,
                alpha=0.38,
                hatch="//",
                zorder=1,
            )
        return handle
    half_width = 0.35 * float(np.min(np.diff(positions)))
    for idx, is_valid in enumerate(mask):
        if not is_valid:
            continue
        ax.fill_between(
            [positions[idx] - half_width, positions[idx] + half_width],
            [lb_common[idx], lb_common[idx]],
            [ub_common[idx], ub_common[idx]],
            facecolor="tab:green",
            edgecolor="tab:green",
            linewidth=0.0,
            alpha=0.38,
            hatch="//",
            zorder=1,
        )
    for idx in range(len(positions) - 1):
        if not (mask[idx] and mask[idx + 1]):
            continue
        ax.fill_between(
            positions[idx : idx + 2],
            lb_common[idx : idx + 2],
            ub_common[idx : idx + 2],
            facecolor="tab:green",
            edgecolor="tab:green",
            linewidth=0.0,
            alpha=0.38,
            hatch="//",
            zorder=1,
        )
    return handle


def _draw_boxplot_subplot(ax, all_repeats, lambda_, budgets):
    positions = np.asarray(budgets, dtype=float)
    step = float(np.min(np.diff(positions))) if len(positions) > 1 else 1.0
    offsets = {
        "cons_lb": -0.30 * step,
        "cons_ub": -0.10 * step,
        "inc_lb":   0.10 * step,
        "inc_ub":   0.30 * step,
    }
    box_width = 0.14 * step
    legend_handles = []
    data_by_key = {}
    medians_by_key = {}
    for key in BOUND_STYLES:
        data = [[rep[key][b_idx] for rep in all_repeats] for b_idx in range(len(budgets))]
        data_by_key[key] = data
        medians_by_key[key] = np.asarray([float(np.median(d)) for d in data])
    cons_gap = ax.fill_between(
        positions,
        medians_by_key["cons_lb"],
        medians_by_key["cons_ub"],
        color="tab:blue",
        alpha=0.14,
        label="Conservative gap",
        zorder=0,
    )
    inc_gap = ax.fill_between(
        positions,
        medians_by_key["inc_lb"],
        medians_by_key["inc_ub"],
        color="tab:orange",
        alpha=0.16,
        label="Incremental gap",
        zorder=0,
    )
    lb_common = np.maximum(medians_by_key["cons_lb"], medians_by_key["inc_lb"])
    ub_common = np.minimum(medians_by_key["cons_ub"], medians_by_key["inc_ub"])
    common_mask = lb_common <= ub_common
    common_region = _fill_common_region(ax, positions, lb_common, ub_common, common_mask)
    for key, style in BOUND_STYLES.items():
        data = data_by_key[key]
        color = style["color"]
        bp = ax.boxplot(
            data,
            positions=positions + offsets[key],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(color=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            medianprops=dict(color=color, linewidth=1.6),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
        ax.plot(
            positions + offsets[key], medians_by_key[key],
            color=color, marker=style["marker"],
            linestyle=style["linestyle"], linewidth=1.6,
            zorder=3,
        )
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D(
            [0], [0], color=color, marker=style["marker"],
            linestyle=style["linestyle"], linewidth=1.6,
            label=style["label"],
        ))
    legend_handles.extend([cons_gap, inc_gap, common_region])
    ax.set_title(f"λ = {lambda_}", fontsize=11)
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Information gain")
    ax.set_xticks(budgets)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend(handles=legend_handles, frameon=False, fontsize=8)


def make_figure(results_by_lambda, selection_method, output_path):
    lambdas = list(results_by_lambda.keys())
    n = len(lambdas)
    n_cols = min(2, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()
    for ax, lam in zip(axes, lambdas):
        _draw_boxplot_subplot(ax, results_by_lambda[lam], lam, SENSOR_BUDGETS)
    for ax in axes[len(lambdas):]:
        ax.axis("off")
    fig.suptitle(f"GO EIG bounds — {selection_method} selection  ({N_REPEATS} repeats)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inc_by_lambda = {}
    cons_by_lambda = {}
    for lam in LAMBDAS:
        print(f"\n=== lambda = {lam} ===")
        prior, joint_prior, Sigma_obs, G = build_objects(lam)
        print(f"  Estimating eig_mem (n={N_SAMPLES_EIG})...")
        eig_offset = estimate_eig_memory_efficient(
            G=G,
            joint_prior=joint_prior,
            Sigma_obs=Sigma_obs,
            n_samples=N_SAMPLES_EIG,
            batch_size=500,
            seed=BASE_SEED,
        )
        print(f"  eig_mem = {eig_offset:.4f}")
        seeds = [BASE_SEED + 1000 * LAMBDAS.index(lam) + r for r in range(N_REPEATS)]
        print(f"  Running {N_REPEATS} repeats...")
        repeat_outputs = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_repeat)(lam, s, eig_offset) for s in seeds
        )
        inc_by_lambda[lam] = [out[0] for out in repeat_outputs]
        cons_by_lambda[lam] = [out[1] for out in repeat_outputs]
    print("\nGenerating figures...")
    make_figure(
        inc_by_lambda,
        selection_method="incremental",
        output_path=OUTPUT_DIR / "bounds_incremental_selection_GO.png",
    )
    make_figure(
        cons_by_lambda,
        selection_method="conservative",
        output_path=OUTPUT_DIR / "bounds_conservative_selection_GO.png",
    )
    print("Done.")

if __name__ == "__main__":
    main()
