"""
run_bounds_GO.py
----------------
Répétition scriptable de tuto_bounds_GO (notebook).
Structure : multi-lambda × multi-seed, sauvegarde .npz par lambda.

Une fois par lambda :
  - Sigma_Y_given_theta   (MC imbriqué)
  - Sigma_signal_free     (régression Y → G)
  - Sigma_noise_free      (régression (theta,Y) → G)
  - eig_offset            (estimateur KL sur theta, bruit = Sigma_Y_given_theta)

Par répétition (seed variable) :
  - E[J^T], Cov(J)        (différences finies)
  - Sigma_Y               (MC marginal)
  - Sigma_signal, Sigma_signal_misfit
  - Sigma_noise,  Sigma_noise_misfit
  → bornes {cons,inc} × {fd,free}

Clés .npz : {method}_{variant}_{lb,ub,inc_lb,inc_ub}
  method  : cons (greedy BI), inc (glouton incrémental)
  variant : fd (jacobiens FD), free (gradient-free)
"""

from __future__ import annotations

import os
import sys
import warnings
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyboed_mpl")

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2]
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from boed.core.noise import NoiseModel
from boed.pde.burgers import Burgers_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import Matern32
from boed.utils.observation import build_selection_matrices

# =============================================================================
# Paramètres
# =============================================================================

LAMBDAS        = [0.0]
SENSOR_BUDGETS = [5, 10, 15, 20, 25]
N_REPEATS      = 1
BASE_SEED      = 42
N_JOBS         = -1

N           = 100
DT          = 0.001
N_STEPS     = 100
SIGMA       = 0.01
DIFFUSIVITY = 0.02

KERNEL_LS   = 0.2
KERNEL_SIG  = 1.0

N_SAMPLES         = 500    # FD jacobiens par repeat
N_SAMPLES_SIGMA_Y = 500    # Sigma_Y par repeat
N_THETA_YTH       = 500    # n_theta pour estimate_E_cov_Y_given_theta
N_ETA_INNER       = 200    # n_eta  pour estimate_E_cov_Y_given_theta
N_SAMPLES_FREE    = 20000  # régression gradient-free (une fois)
N_SAMPLES_EIG     = 2000   # estimateur KL EIG (une fois)

OUTPUT_DIR = Path("results_sweep")

# =============================================================================
# Prior joint et forward model
# =============================================================================

def build_objects(lambda_):
    model  = Burgers_CN(N=N, dt=DT, diffusivity=DIFFUSIVITY, lambda_=lambda_)
    kernel = Matern32(length_scale=KERNEL_LS, sigma=KERNEL_SIG)
    try:
        prior = GaussianProcessPrior(kernel, mu=np.zeros(N), nx=N)
    except TypeError:
        prior = GaussianProcessPrior(kernel, nx=N)
        prior.mu = np.zeros(N)

    eigvals = la.eigvalsh(prior.Sigma)
    prior.Sigma += eigvals[-1] / 1000.0

    d = q = N // 2
    joint_prior = {"mu": prior.mu, "Sigma": prior.Sigma, "d": d, "q": q}

    noise     = NoiseModel(sigma_noise=SIGMA)
    Sigma_obs = noise.get_covariance(N)

    def G(theta, eta):
        u0 = np.concatenate((theta, eta), axis=None)
        return model.evolve(u0=u0, n_steps=N_STEPS)[-1, :]

    return prior, joint_prior, Sigma_obs, G

# =============================================================================
# MC : prior joint
# =============================================================================

def sample_joint_prior(joint_prior, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    d = joint_prior["d"]
    z = rng.multivariate_normal(mean=joint_prior["mu"], cov=joint_prior["Sigma"], size=n_samples)
    return z[:, :d], z[:, d:]

# =============================================================================
# Jacobiens par différences finies
# =============================================================================

def jacobian_fd_theta(G, theta, eta, h=None):
    d = theta.shape[0]
    h = h or np.finfo(float).eps ** (1/3) * (la.norm(theta) + 1e-8)
    m = G(theta, eta).shape[0]
    J = np.zeros((m, d))
    for j in range(d):
        ej = np.zeros(d); ej[j] = 1.0
        J[:, j] = (G(theta + h*ej, eta) - G(theta - h*ej, eta)) / (2*h)
    return J


def jacobian_fd_eta(G, theta, eta, h=None):
    q = eta.shape[0]
    h = h or np.finfo(float).eps ** (1/3) * (la.norm(eta) + 1e-8)
    m = G(theta, eta).shape[0]
    J = np.zeros((m, q))
    for j in range(q):
        ej = np.zeros(q); ej[j] = 1.0
        J[:, j] = (G(theta, eta + h*ej) - G(theta, eta - h*ej)) / (2*h)
    return J


def _one_sample(G, theta, eta):
    return jacobian_fd_theta(G, theta, eta).T, jacobian_fd_eta(G, theta, eta).T


def estimate_E_JT(G, joint_prior, n_samples, rng=None):
    """E[J_theta^T] (d,m), E[J_eta^T] (q,m) et leur concaténation (d+q,m)."""
    theta_s, eta_s = sample_joint_prior(joint_prior, n_samples, rng=rng)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_one_sample)(G, theta_s[k], eta_s[k]) for k in range(n_samples)
    )
    Jt0, Je0 = results[0]
    Jt_sum = np.zeros_like(Jt0); Je_sum = np.zeros_like(Je0)
    for Jt, Je in results:
        Jt_sum += Jt; Je_sum += Je
    EJt = Jt_sum / n_samples
    EJe = Je_sum / n_samples
    return {"EJ_theta_T": EJt, "EJ_eta_T": EJe, "EJ_full_T": np.vstack([EJt, EJe])}


def _compute_z(theta, eta, G, Sinv_sqrt):
    Zt = jacobian_fd_theta(G, theta, eta).T @ Sinv_sqrt
    Ze = jacobian_fd_eta(G,  theta, eta).T @ Sinv_sqrt
    return Zt, Ze


def estimate_jacobian_covariances_mc(G, joint_prior, Sigma_obs, n_samples, rng=None):
    """Cov_theta, Cov_eta, Cov_theta_eta, Cov_full."""
    evals, evecs = la.eigh(np.asarray(Sigma_obs, dtype=float))
    Sinv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    theta_s, eta_s = sample_joint_prior(joint_prior, n_samples, rng=rng)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_compute_z)(theta_s[k], eta_s[k], G, Sinv_sqrt) for k in range(n_samples)
    )
    Zt_arr = np.asarray([r[0] for r in results])
    Ze_arr = np.asarray([r[1] for r in results])
    Zt_m = Zt_arr.mean(axis=0); Ze_m = Ze_arr.mean(axis=0)
    Ct = np.zeros((Zt_m.shape[0],)*2)
    Ce = np.zeros((Ze_m.shape[0],)*2)
    Cte = np.zeros((Zt_m.shape[0], Ze_m.shape[0]))
    for k in range(n_samples):
        dt = Zt_arr[k] - Zt_m; de = Ze_arr[k] - Ze_m
        Ct += dt @ dt.T; Ce += de @ de.T; Cte += dt @ de.T
    Ct  = 0.5*(Ct  + Ct.T) / n_samples
    Ce  = 0.5*(Ce  + Ce.T) / n_samples
    Cte /= n_samples
    return {
        "Cov_theta":     Ct,
        "Cov_eta":       Ce,
        "Cov_theta_eta": Cte,
        "Cov_full":      np.block([[Ct, Cte], [Cte.T, Ce]]),
    }


def estimate_Sigma_Y(G, joint_prior, Sigma_obs, n_samples, rng=None):
    theta_s, eta_s = sample_joint_prior(joint_prior, n_samples, rng=rng)
    U = np.asarray(Parallel(n_jobs=N_JOBS)(
        delayed(G)(theta_s[k], eta_s[k]) for k in range(n_samples)
    ))
    SY = np.asarray(Sigma_obs) + np.cov(U, rowvar=False, bias=False)
    return 0.5*(SY + SY.T)


def _process_theta(i, theta_i, mu_theta, Sigma_theta, mu_eta, Sigma_et,
                   Sigma_eta_given_theta, G, n_eta, base_seed):
    rng = np.random.default_rng(base_seed + i)
    mu_eta_i = mu_eta + Sigma_et @ la.solve(Sigma_theta, theta_i - mu_theta)
    eta_cond  = rng.multivariate_normal(mean=mu_eta_i, cov=Sigma_eta_given_theta, size=n_eta)
    vals = np.array([G(theta_i, eta_cond[j]) for j in range(n_eta)])
    if vals.ndim == 1:
        vals = vals[:, None]
    return np.cov(vals, rowvar=False, ddof=1)


def estimate_E_cov_Y_given_theta(G, joint_prior, Sigma_obs, n_theta, n_eta, rng=None, n_jobs=-1):
    """E_theta[Cov(Y|theta)] = Sigma_obs + E_theta[Cov(G(theta,eta)|theta)]."""
    if rng is None:
        rng = np.random.default_rng()
    d = joint_prior["d"]
    mu, Sigma = joint_prior["mu"], joint_prior["Sigma"]
    mu_theta   = mu[:d];  Sigma_theta = Sigma[:d, :d]
    mu_eta     = mu[d:];  Sigma_eta   = Sigma[d:, d:]
    Sigma_et   = Sigma[d:, :d]
    Sigma_eGt  = Sigma_eta - Sigma_et @ la.solve(Sigma_theta, Sigma_et.T)
    Sigma_eGt  = 0.5*(Sigma_eGt + Sigma_eGt.T) + 1e-12*np.eye(Sigma_eta.shape[0])
    theta_s    = rng.multivariate_normal(mean=mu_theta, cov=Sigma_theta, size=n_theta)
    base_seed  = int(rng.integers(0, 2**30))
    cov_list   = Parallel(n_jobs=n_jobs)(
        delayed(_process_theta)(i, theta_s[i], mu_theta, Sigma_theta, mu_eta,
                                Sigma_et, Sigma_eGt, G, n_eta, base_seed)
        for i in range(n_theta)
    )
    result = np.asarray(Sigma_obs) + np.mean(np.asarray(cov_list), axis=0)
    return 0.5*(result + result.T)

# =============================================================================
# Matrices de covariance signal / bruit
# =============================================================================

def compute_Sigma_signal(l_theta_eta, H_theta_eta, Sigma_theta_eta, Sigma_obs):
    N_loc = Sigma_theta_eta.shape[0]
    HS = H_theta_eta @ Sigma_theta_eta
    M  = Sigma_theta_eta - Sigma_theta_eta @ la.solve(np.eye(N_loc) + HS, HS)
    return np.asarray(Sigma_obs) + l_theta_eta.T @ M @ l_theta_eta


def compute_Sigma_signal_misfit(l_theta_eta, H_theta_eta, Sigma_theta_eta, Sigma_obs):
    eigvals, eigvects = la.eigh(np.asarray(Sigma_theta_eta))
    eigvals = np.clip(eigvals, 0.0, None)
    Ssqrt = eigvects @ np.diag(np.sqrt(eigvals)) @ eigvects.T
    H_m   = Ssqrt @ H_theta_eta @ Ssqrt
    A     = Ssqrt @ la.solve(np.eye(H_m.shape[0]) + H_m, Ssqrt)  # = (Sigma^{-1}+H)^{-1}
    X     = A @ l_theta_eta
    return np.asarray(Sigma_obs) + l_theta_eta.T @ X


def compute_Sigma_noise(L_eta, H_eta, Sigma_eta_given_theta, Sigma_obs):
    q  = Sigma_eta_given_theta.shape[0]
    HS = H_eta @ Sigma_eta_given_theta
    M  = Sigma_eta_given_theta - Sigma_eta_given_theta @ la.solve(np.eye(q) + HS, HS)
    return np.asarray(Sigma_obs) + L_eta.T @ M @ L_eta


def compute_Sigma_noise_misfit(L_eta, H_eta, Sigma_eta_given_theta, Sigma_obs):
    eigvals, eigvects = la.eigh(np.asarray(Sigma_eta_given_theta))
    eigvals = np.clip(eigvals, 1e-14, None)
    Ssqrt = eigvects @ np.diag(np.sqrt(eigvals)) @ eigvects.T
    H_m   = Ssqrt @ H_eta @ Ssqrt
    A     = Ssqrt @ la.solve(np.eye(H_m.shape[0]) + H_m, Ssqrt)  # = (Sigma^{-1}+H)^{-1}
    X     = A @ L_eta
    return np.asarray(Sigma_obs) + L_eta.T @ X

# =============================================================================
# Gradient-free (régression linéaire)
# =============================================================================

def solve_linear_regression_theta_Y(G, joint_prior, Sigma_obs, n_samples=10000,
                                     random_state=0, n_jobs=-1):
    """Régresse X = G(theta,eta) sur Z = (theta, Y). Retourne A, b, M_emp, stats."""
    rng = np.random.default_rng(random_state)
    theta_s, eta_s = sample_joint_prior(joint_prior, n_samples, rng=rng)
    X = np.asarray(Parallel(n_jobs=n_jobs)(
        delayed(G)(theta_s[k], eta_s[k]) for k in range(n_samples)
    ), dtype=float)
    n, p = X.shape
    Sigma_obs = np.asarray(Sigma_obs)
    eps = rng.multivariate_normal(np.zeros(p), Sigma_obs, size=n)
    Y   = X + eps
    Z   = np.hstack([theta_s, Y])
    mX  = X.mean(axis=0); mZ = Z.mean(axis=0)
    Xc  = X - mX;         Zc = Z - mZ
    SXZ = (Xc.T @ Zc) / n
    SZZ = (Zc.T @ Zc) / n
    A   = SXZ @ np.linalg.pinv(SZZ)
    b   = mX - A @ mZ
    R   = X - (Z @ A.T + b)
    M   = (R.T @ R) / n
    stats = {"m_X": mX, "m_Z": mZ, "X": X, "Y": Y, "Z": Z, "residuals": R}
    return A, b, M, stats


def solve_linear_matrix_regression_minimization(G, joint_prior, Sigma_obs,
                                                 n_samples=10000, random_state=0):
    """Résout min_{A,b} E[(X-AY-b)(...)^T]. Retourne A_star, b_star, M_emp, M_formula, stats."""
    Sigma_obs = np.asarray(Sigma_obs)
    rng = np.random.default_rng(random_state)
    theta_s, eta_s = sample_joint_prior(joint_prior, n_samples, rng=rng)
    X = np.asarray(Parallel(n_jobs=N_JOBS)(
        delayed(G)(theta_s[k], eta_s[k]) for k in range(n_samples)
    ), dtype=float)
    n, p = X.shape
    eps = rng.multivariate_normal(np.zeros(p), Sigma_obs, size=n)
    Y   = X + eps
    mX  = X.mean(axis=0); mY = Y.mean(axis=0)
    Xc  = X - mX;         Yc = Y - mY
    SXY = (Xc.T @ Yc) / n
    SYY = (Yc.T @ Yc) / n
    SX  = (Xc.T @ Xc) / n
    A   = SXY @ np.linalg.inv(SYY)
    b   = mX - A @ mY
    R   = X - (Y @ A.T + b)
    M_emp     = (R.T @ R) / n
    M_formula = SX - SX @ np.linalg.inv(SX + Sigma_obs) @ SX
    stats = {"m_X": mX, "m_Y": mY, "X": X, "Y": Y, "residuals": R}
    return A, b, M_emp, M_formula, stats

# =============================================================================
# KLPrior + estimateur EIG KL
# =============================================================================

class KLPrior:
    """Prior N(mu, Sigma) → représentation KL tronquée. xi ~ N(0, I_k)."""

    def __init__(self, mu, Sigma, energy=0.99999):
        self.mu_full = np.asarray(mu)
        ev, V = la.eigh(np.asarray(Sigma))
        ev = ev[::-1]; V = V[:, ::-1]
        cumvar = np.cumsum(ev) / ev.sum()
        self.k        = int(np.searchsorted(cumvar, energy)) + 1
        self.V_k      = V[:, :self.k]
        self.lam_k    = ev[:self.k]
        self.sqrt_lam = np.sqrt(self.lam_k)
        self.mu    = np.zeros(self.k)
        self.Sigma = np.eye(self.k)
        print(f"    KL tronquée : {self.k} modes, {cumvar[self.k-1]*100:.3f}% variance")

    def xi_to_z(self, xi):
        return self.mu_full + self.V_k @ (self.sqrt_lam * xi)

    def make_G_reduced(self, G_1arg):
        return lambda xi: G_1arg(self.xi_to_z(xi))

    def make_J_reduced(self, J_G_1arg):
        def J_r(xi):
            z = self.xi_to_z(xi)
            J = np.asarray(J_G_1arg(z))        # (m, d)
            return J @ self.V_k @ np.diag(self.sqrt_lam)  # (m, k)
        return J_r


def estimate_eig_kl(G_j, J_G_j, kl_prior, Sigma_obs, n_outer, seed=None, n_jobs=-1):
    """Estimateur KL de l'EIG. Sigma_obs est le bruit effectif (Sigma_Y_given_theta en GO)."""
    G_r = kl_prior.make_G_reduced(G_j)
    J_r = kl_prior.make_J_reduced(J_G_j)
    k   = kl_prior.k
    m   = len(Sigma_obs)

    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    L_obs    = sla.cholesky(Sigma_obs, lower=True)
    Sinv_obs = sla.cho_solve((L_obs, True), np.eye(m))
    _, ld    = la.slogdet(Sigma_obs)
    log_c    = -0.5 * (ld + m * np.log(2*np.pi))

    ss    = np.random.SeedSequence(seed)
    seeds = ss.spawn(n_outer)

    def _step(seed_i):
        rng = np.random.default_rng(seed_i)
        xi  = rng.standard_normal(k)
        Gr  = np.asarray(G_r(xi))
        Jr  = np.asarray(J_r(xi))              # (m, k)
        S      = Jr @ Jr.T + Sigma_obs
        L_S    = sla.cholesky(S, lower=True)
        Sinv_S = sla.cho_solve((L_S, True), np.eye(m))
        _, ld_S = la.slogdet(S)
        log_c_S = -0.5 * (ld_S + m * np.log(2*np.pi))
        bias_corr = 0.5 * (np.trace(Sinv_S @ Sigma_obs) - m)
        eps    = rng.multivariate_normal(np.zeros(m), Sigma_obs)
        y      = Gr + eps
        log_lik = log_c - 0.5 * np.dot(eps, Sinv_obs @ eps)
        r       = y - Gr
        log_py  = log_c_S - 0.5 * r @ Sinv_S @ r
        return float(log_lik - log_py - bias_corr)

    results = Parallel(n_jobs=n_jobs)(delayed(_step)(seeds[i]) for i in range(n_outer))
    return float(np.mean(results))

# =============================================================================
# Bornes EIG
# =============================================================================

def _log_ratio(A, B, M):
    _, la_ = la.slogdet(M.T @ A @ M)
    _, lb_ = la.slogdet(M.T @ B @ M)
    return 0.5 * (la_ - lb_)


def eig_LB(Sigma_Y, Sigma_noise, W):
    """Borne inférieure BI : log|W^T Sigma_Y W| / |W^T Sigma_noise W| (relatif au plein)."""
    eye = np.eye(Sigma_noise.shape[0])
    return _log_ratio(Sigma_Y, Sigma_noise, W) - _log_ratio(Sigma_Y, Sigma_noise, eye)


def eig_UB(Sigma_signal, Sigma_Y_given_theta, W):
    """Borne supérieure BS : log|W^T Sigma_signal W| / |W^T Sigma_Y_given_theta W| (relatif)."""
    eye = np.eye(Sigma_Y_given_theta.shape[0])
    return _log_ratio(Sigma_signal, Sigma_Y_given_theta, W) - _log_ratio(Sigma_signal, Sigma_Y_given_theta, eye)


def greedy_maximize_LB(Sigma_Y, Sigma_noise, n_sensors):
    Sigma_Y     = 0.5*(Sigma_Y     + Sigma_Y.T)
    Sigma_noise = 0.5*(Sigma_noise + Sigma_noise.T)
    N_g = Sigma_Y.shape[0]
    selected = []; remaining = set(range(N_g))

    for _ in range(n_sensors):
        best_idx, best_sc = None, -np.inf
        for idx in remaining:
            S = selected + [idx]; ix = np.ix_(S, S)
            sy, ldy = la.slogdet(Sigma_Y[ix])
            sn, ldn = la.slogdet(Sigma_noise[ix])
            if sy <= 0 or sn <= 0:
                continue
            sc = 0.5*(ldy - ldn)
            if sc > best_sc:
                best_sc, best_idx = sc, idx
        if best_idx is None:
            warnings.warn("greedy_maximize_LB : aucun capteur SPD trouvé.", RuntimeWarning, stacklevel=2)
            break
        selected.append(best_idx); remaining.remove(best_idx)
    return np.array(selected, dtype=int)


def incremental_bounds(Sigma_signal, Sigma_Y_given_theta, Sigma_Y, Sigma_noise, n_sensors):
    """Sélection gloutonne par bornes incrémentales (Schur rang-1, O(N²) par étape).

    LB increment : ½ log [Σ_signal(Wm)]_ii / [Σ_{Y|θ}(Wm)]_ii
    UB increment : ½ log [Σ_Y(Wm)]_ii     / [Σ_noise(Wm)]_ii
    """
    N_loc = Sigma_Y.shape[0]
    Ss    = Sigma_signal.copy();  Syth = Sigma_Y_given_theta.copy()
    SY    = Sigma_Y.copy();       Sn   = Sigma_noise.copy()
    selected: list[int] = []; remaining: list[int] = list(range(N_loc))
    scores_inf: list[float] = []; scores_sup: list[float] = []
    eig_inf = eig_sup = 0.0

    for _ in range(n_sensors):
        best_idx = None; best_dinf = -np.inf; best_dsup = None
        for idx in remaining:
            ni = Ss[idx,idx]; di = Syth[idx,idx]
            nu = SY[idx,idx]; du = Sn[idx,idx]
            if min(ni, di, nu, du) <= 1e-14:
                warnings.warn(f"Variance quasi-nulle capteur {idx} — ignoré.",
                              RuntimeWarning, stacklevel=2)
                continue
            dinf = 0.5*np.log(ni/di); dsup = 0.5*np.log(nu/du)
            if dinf > best_dinf:
                best_dinf, best_dsup, best_idx = dinf, dsup, idx
        if best_idx is None:
            warnings.warn("Aucun capteur valide — arrêt prématuré.", RuntimeWarning, stacklevel=2)
            break
        for S in (Ss, Syth, SY, Sn):
            col = S[:, best_idx].copy()
            S  -= np.outer(col, col) / S[best_idx, best_idx]
        selected.append(best_idx); remaining.remove(best_idx)
        eig_inf += best_dinf; eig_sup += best_dsup
        scores_inf.append(eig_inf); scores_sup.append(eig_sup)

    return {
        "indices":         np.array(selected, dtype=int),
        "scores_inf":      scores_inf,
        "scores_sup":      scores_sup,
        "EIG_lower_bound": eig_inf,
        "EIG_upper_bound": eig_sup,
    }


def incremental_bounds_given_W(Sigma_signal, Sigma_Y_given_theta, Sigma_Y, Sigma_noise, W):
    _, ls = la.slogdet(W.T @ Sigma_signal       @ W)
    _, lt = la.slogdet(W.T @ Sigma_Y_given_theta @ W)
    _, ly = la.slogdet(W.T @ Sigma_Y             @ W)
    _, ln = la.slogdet(W.T @ Sigma_noise          @ W)
    return 0.5*(ls - lt), 0.5*(ly - ln)

# =============================================================================
# Helper bornes pour un jeu de matrices
# =============================================================================

def _bounds_for_W(Ss, Sn, Sigma_Y, SYth, budgets, method):
    """
    Ss  : Sigma_signal_misfit  (numérateur signal, préconditionné par Sigma_prior)
    Sn  : Sigma_noise_misfit   (dénominateur bruit, préconditionné par Sigma_eta|theta)
    SYth: Sigma_Y_given_theta  (dénominateur fixe de eig_UB et eig_UB incrémental)

    Retourne (cons_lb, cons_ub, inc_lb, inc_ub), listes de longueur len(budgets).
      cons_lb / cons_ub : eig_LB / eig_UB sur le W sélectionné (greedy ou incrémental)
      inc_lb  / inc_ub  : bornes incrémentales cumulées
    """
    cons_lb, cons_ub, inc_lb, inc_ub = [], [], [], []
    max_b = max(budgets)

    if method == "cons":
        indices = greedy_maximize_LB(Sigma_Y, Sn, max_b)
        for budget in budgets:
            W, _ = build_selection_matrices(N, indices[:budget])
            cons_lb.append(eig_LB(Sigma_Y, Sn, W))
            cons_ub.append(eig_UB(Ss, SYth, W))
            lb, ub = incremental_bounds_given_W(Ss, SYth, Sigma_Y, Sn, W)
            inc_lb.append(lb); inc_ub.append(ub)

    elif method == "inc":
        for budget in budgets:
            res  = incremental_bounds(Ss, SYth, Sigma_Y, Sn, budget)
            W, _ = build_selection_matrices(N, res["indices"])
            cons_lb.append(eig_LB(Sigma_Y, Sn, W))
            cons_ub.append(eig_UB(Ss, SYth, W))
            inc_lb.append(res["EIG_lower_bound"])
            inc_ub.append(res["EIG_upper_bound"])

    return cons_lb, cons_ub, inc_lb, inc_ub

# =============================================================================
# Une répétition
# =============================================================================

def run_one_repeat(lambda_, seed, eig_offset, Sigma_Y_given_theta,
                   Sigma_signal_free, Sigma_noise_free):
    prior, joint_prior, Sigma_obs, G = build_objects(lambda_)

    d = joint_prior["d"]
    Sigma_theta        = prior.Sigma[:d, :d]
    Sigma_eta          = prior.Sigma[d:, d:]
    Sigma_et           = prior.Sigma[d:, :d]
    Sigma_eta_given_theta = Sigma_eta - Sigma_et @ la.solve(Sigma_theta, Sigma_et.T)
    Sigma_eta_given_theta = 0.5*(Sigma_eta_given_theta + Sigma_eta_given_theta.T)
    Sigma_eta_given_theta += 1e-12 * np.eye(joint_prior["q"])

    rng = np.random.default_rng(seed)
    res_l   = estimate_E_JT(G, joint_prior, N_SAMPLES, rng=rng)
    res_H   = estimate_jacobian_covariances_mc(G, joint_prior, Sigma_obs, N_SAMPLES, rng=rng)
    Sigma_Y = estimate_Sigma_Y(G, joint_prior, Sigma_obs, N_SAMPLES_SIGMA_Y, rng=rng)

    l_full = res_l["EJ_full_T"]; H_full = res_H["Cov_full"]
    l_eta  = res_l["EJ_eta_T"];  H_eta  = res_H["Cov_eta"]

    # Variante misfit : préconditionné par Sigma_prior / Sigma_eta|theta
    # Capte les directions de maximum d'information (VP de H_m = Sigma^{1/2} H Sigma^{1/2})
    Sigma_signal_misfit = compute_Sigma_signal_misfit(l_full, H_full, prior.Sigma, Sigma_obs)
    Sigma_noise_misfit  = compute_Sigma_noise_misfit(l_eta, H_eta, Sigma_eta_given_theta, Sigma_obs)

    result = {}
    for method in ("cons", "inc"):
        # Variante fd : jacobiens par différences finies, matrices misfit
        clb, cub, ilb, iub = _bounds_for_W(
            Sigma_signal_misfit, Sigma_noise_misfit,
            Sigma_Y, Sigma_Y_given_theta, SENSOR_BUDGETS, method,
        )
        result[f"{method}_fd_lb"]     = [v + eig_offset for v in clb]
        result[f"{method}_fd_ub"]     = [v + eig_offset for v in cub]
        result[f"{method}_fd_inc_lb"] = ilb
        result[f"{method}_fd_inc_ub"] = iub

        # Variante free : gradient-free (régression), matrices misfit approchées sans jacobiens
        clb, cub, ilb, iub = _bounds_for_W(
            Sigma_signal_free, Sigma_noise_free,
            Sigma_Y, Sigma_Y_given_theta, SENSOR_BUDGETS, method,
        )
        result[f"{method}_free_lb"]     = [v + eig_offset for v in clb]
        result[f"{method}_free_ub"]     = [v + eig_offset for v in cub]
        result[f"{method}_free_inc_lb"] = ilb
        result[f"{method}_free_inc_ub"] = iub

    return result

# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lam in LAMBDAS:
        print(f"\n{'='*60}\n  lambda = {lam}\n{'='*60}")

        prior, joint_prior, Sigma_obs, G = build_objects(lam)
        d = joint_prior["d"]

        # ------------------------------------------------------------------
        # 1. Sigma_Y_given_theta  (MC imbriqué, une fois)
        # ------------------------------------------------------------------
        print("  [1/4] Sigma_Y_given_theta (MC imbriqué)...")
        Sigma_Y_given_theta = estimate_E_cov_Y_given_theta(
            G, joint_prior, Sigma_obs, n_theta=N_THETA_YTH, n_eta=N_ETA_INNER,
            rng=np.random.default_rng(BASE_SEED), n_jobs=N_JOBS,
        )

        # ------------------------------------------------------------------
        # 2. EIG offset — KL sur theta, bruit = Sigma_Y_given_theta
        # ------------------------------------------------------------------
        print("  [2/4] EIG offset (KL)...")
        theta_prior_mu    = joint_prior["mu"][:d]
        theta_prior_Sigma = joint_prior["Sigma"][:d, :d]
        mu_eta            = joint_prior["mu"][d:]
        Sigma_eta_theta   = joint_prior["Sigma"][d:, :d]  # Cov(eta, theta)
        A_cond = la.solve(theta_prior_Sigma, Sigma_eta_theta.T).T   # E[eta|theta]

        kl_prior_theta = KLPrior(theta_prior_mu, theta_prior_Sigma)

        def G_eff(theta):
            return G(theta, mu_eta + A_cond @ (theta - theta_prior_mu))

        def J_G_eff(theta):
            eta_c = mu_eta + A_cond @ (theta - theta_prior_mu)
            Jt = jacobian_fd_theta(G, theta, eta_c)   # (m, d)
            Je = jacobian_fd_eta(G,  theta, eta_c)    # (m, q)
            return Jt + Je @ A_cond                    # (m, d)

        eig_offset = estimate_eig_kl(
            G_eff, J_G_eff, kl_prior_theta,
            Sigma_Y_given_theta, N_SAMPLES_EIG,
            seed=BASE_SEED, n_jobs=N_JOBS,
        )
        print(f"    eig_offset (KL) = {eig_offset:.4f}")

        # ------------------------------------------------------------------
        # 3. Sigma_noise_free  (régression (theta,Y) → G)
        # ------------------------------------------------------------------
        print("  [3/4] Sigma_noise_free (régression theta,Y → G)...")
        _, _, M_noise, _ = solve_linear_regression_theta_Y(
            G, joint_prior, Sigma_obs,
            n_samples=N_SAMPLES_FREE, random_state=BASE_SEED + 1,
        )
        Sobs_inv = la.inv(np.asarray(Sigma_obs))
        EIytheta = Sobs_inv @ (np.eye(N) - M_noise @ Sobs_inv)
        Sigma_noise_free = la.inv(EIytheta)
        Sigma_noise_free = 0.5*(Sigma_noise_free + Sigma_noise_free.T)

        # ------------------------------------------------------------------
        # 4. Sigma_signal_free  (régression Y → G)
        # ------------------------------------------------------------------
        print("  [4/4] Sigma_signal_free (régression Y → G)...")
        _, _, _, M_formula, _ = solve_linear_matrix_regression_minimization(
            G, joint_prior, Sigma_obs,
            n_samples=N_SAMPLES_FREE, random_state=BASE_SEED + 2,
        )
        Iy = Sobs_inv @ (np.eye(N) - M_formula @ Sobs_inv)
        Sigma_signal_free = la.inv(Iy)
        Sigma_signal_free = 0.5*(Sigma_signal_free + Sigma_signal_free.T)

        # ------------------------------------------------------------------
        # 5. Répétitions
        # ------------------------------------------------------------------
        print(f"  Repeats (n={N_REPEATS})...")
        seeds = [BASE_SEED + 1000*LAMBDAS.index(lam) + r for r in range(N_REPEATS)]
        repeat_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_repeat)(
                lam, s, eig_offset,
                Sigma_Y_given_theta, Sigma_signal_free, Sigma_noise_free,
            )
            for s in seeds
        )

        # ------------------------------------------------------------------
        # 6. Sauvegarde .npz
        # ------------------------------------------------------------------
        keys = [
            "cons_fd_lb",   "cons_fd_ub",   "cons_fd_inc_lb",   "cons_fd_inc_ub",
            "cons_free_lb", "cons_free_ub", "cons_free_inc_lb", "cons_free_inc_ub",
            "inc_fd_lb",    "inc_fd_ub",    "inc_fd_inc_lb",    "inc_fd_inc_ub",
            "inc_free_lb",  "inc_free_ub",  "inc_free_inc_lb",  "inc_free_inc_ub",
        ]
        arrays = {k: np.array([r[k] for r in repeat_results]) for k in keys}

        fname = OUTPUT_DIR / f"results_GO_lambda_{lam:.2f}.npz"
        np.savez(
            fname,
            eig_offset     = np.float64(eig_offset),
            sensor_budgets = np.array(SENSOR_BUDGETS),
            lambda_val     = np.float64(lam),
            n_repeats      = np.int64(N_REPEATS),
            **arrays,
        )
        print(f"  Sauvegardé : {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
