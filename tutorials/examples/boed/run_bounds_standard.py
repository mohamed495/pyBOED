from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyboed_mpl")

import numpy as np
import numpy.linalg as la
from joblib import Parallel, delayed # type: ignore
from scipy.stats import norm, qmc

import torch # pyright: ignore[reportMissingImports]
import torch.nn as nn # type: ignore
import torch.optim as optim # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader, TensorDataset # pyright: ignore[reportMissingImports]
import multiprocessing

# --- ajout du repo au path si nécessaire ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2]
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from boed.core import make_u0
from boed.core.noise import NoiseModel
from boed.pde.burgers import Burgers_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import Matern32
from boed.utils.observation import build_selection_matrices

# =============================================================================
# Hyperparamètres
# =============================================================================

LAMBDAS        = [0.0, 0.25, 0.5, 1.0]
SENSOR_BUDGETS = [5, 10, 15, 20, 25]
N_REPEATS      = 5
BASE_SEED      = 42
N_JOBS         = -1

# Modèle
N           = 100
DT          = 0.001
N_STEPS     = 100
SIGMA       = 0.1
DIFFUSIVITY = 0.02

# Prior GP
KERNEL_LS       = 0.2
KERNEL_SIG      = 1.0
U0_CENTER       = 0.25
U0_WIDTH        = 0.07
U0_AMPLITUDE    = 1.5

# Monte Carlo
N_SAMPLES             = 500    # l_theta, H_theta, Sigma_signal répétés
N_SAMPLES_SIGMA_Y     = 500    # Sigma_Y répétés
N_SAMPLES_SIGMA_Y_REF = 1500  # Sigma_Y_ref (une fois)
N_SAMPLES_EIG         = 150000  # eig_mem NMC

# NN
N_TRAIN  = 100000
N_TEST   = 2000
N_EPOCHS = 100
HIDDEN   = [64, 64]

OUTPUT_DIR = Path("results_sweep")

# =============================================================================
# Utilitaires MC
# =============================================================================

def _halton_normal(d: int, n: int, seed: int | None = None) -> np.ndarray:
    sampler = qmc.Halton(d=d, scramble=True, seed=seed)
    u = sampler.random(n=n)
    eps = np.finfo(float).eps
    return norm.ppf(np.clip(u, eps, 1.0 - eps))


def sample_prior(prior, n: int, seed: int | None = None) -> np.ndarray:
    mu = np.asarray(prior.mu)
    L  = np.linalg.cholesky(np.asarray(prior.Sigma))
    z  = _halton_normal(len(mu), n, seed=seed)
    return mu + (L @ z.T).T


# =============================================================================
# Jacobien et espérances
# =============================================================================

def jacobian_fd(G, theta, h=None):
    """Jacobien différences finies centrées. Retourne J shape (m, d)."""
    d  = theta.shape[0]
    h  = h or np.finfo(float).eps ** (1/3) * (la.norm(theta) + 1e-8)
    G0 = G(theta)
    m  = G0.shape[0]
    J  = np.zeros((m, d))
    for j in range(d):
        ej = np.zeros(d); ej[j] = 1.0
        J[:, j] = (G(theta + h * ej) - G(theta - h * ej)) / (2 * h)
    return J


def estimate_E_JT(G, prior, n_samples, seed):
    """E[J_theta^T] shape (d, m)."""
    thetas = sample_prior(prior, n_samples, seed=seed)
    Js = Parallel(n_jobs=N_JOBS)(delayed(jacobian_fd)(G, th) for th in thetas)
    return np.mean([J.T for J in Js], axis=0)


def estimate_H_theta(G, prior, Sigma_obs, n_samples, seed):
    """Cov(J_theta^T Sigma_obs^{-1/2}) shape (d, d)."""
    evals, evecs = la.eigh(np.asarray(Sigma_obs))
    S_inv_sqrt   = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    thetas = sample_prior(prior, n_samples, seed=seed)

    def _Z(th):
        return jacobian_fd(G, th).T @ S_inv_sqrt

    Zs    = np.asarray(Parallel(n_jobs=N_JOBS)(delayed(_Z)(th) for th in thetas))
    Zmean = Zs.mean(axis=0)
    C = np.zeros((Zmean.shape[0],) * 2)
    for z in Zs:
        d = z - Zmean; C += d @ d.T
    C /= len(Zs)
    return 0.5 * (C + C.T)


def estimate_Sigma_Y(G, prior, Sigma_obs, n_samples, seed):
    """Sigma_Y = Sigma_obs + Cov_{pi0}[G(theta)]."""
    thetas = sample_prior(prior, n_samples, seed=seed)
    U  = np.asarray(Parallel(n_jobs=N_JOBS)(delayed(G)(th) for th in thetas))
    SY = np.asarray(Sigma_obs) + np.cov(U, rowvar=False, bias=False)
    return 0.5 * (SY + SY.T)


def compute_Sigma_signal(l_theta, H_theta, Sigma_theta, Sigma_obs):
    """Sigma_signal = Sigma_obs + l^T (Sigma_theta^{-1} + H)^{-1} l."""
    A = la.inv(Sigma_theta) + H_theta
    X = la.solve(A, l_theta)
    return np.asarray(Sigma_obs) + l_theta.T @ X


# =============================================================================
# Estimation EIG
# =============================================================================

def eig_linearise(G, prior, Sigma_obs):
    """EIG exact pour le cas linéaire (lambda=0)."""
    J = jacobian_fd(G, np.asarray(prior.mu))
    S = np.asarray(Sigma_obs) + J @ np.asarray(prior.Sigma) @ J.T
    _, ld_S   = la.slogdet(S)
    _, ld_obs = la.slogdet(np.asarray(Sigma_obs))
    return 0.5 * (ld_S - ld_obs)


def _logsumexp_rows(A):
    amax = A.max(axis=1, keepdims=True)
    return amax.squeeze() + np.log(np.exp(A - amax).sum(axis=1))


def estimate_eig_nmc(G, prior, Sigma_obs, n_samples, batch_size=500, seed=None):
    """Estimateur NMC mémoire-efficace de l'EIG."""
    thetas = sample_prior(prior, n_samples, seed=seed)
    U = np.asarray(Parallel(n_jobs=N_JOBS)(delayed(G)(th) for th in thetas))
    m     = U.shape[1]
    Sinv  = la.inv(Sigma_obs)
    _, ld = la.slogdet(Sigma_obs)
    log_c = -0.5 * (ld + m * np.log(2 * np.pi))

    rng = np.random.default_rng(seed)
    Y   = U + rng.multivariate_normal(np.zeros(m), Sigma_obs, size=n_samples)
    YM, UM = Y @ Sinv, U @ Sinv
    nY = np.einsum("ij,ij->i", Y, YM)
    nU = np.einsum("ij,ij->i", U, UM)

    eig = 0.0
    for i in range(0, n_samples, batch_size):
        ie    = min(i + batch_size, n_samples)
        bs    = ie - i
        cross = YM[i:ie] @ U.T
        quad  = nY[i:ie, None] + nU[None, :] - 2 * cross
        ll    = log_c - 0.5 * quad
        ln    = ll[np.arange(bs), i + np.arange(bs)]
        ld_   = _logsumexp_rows(ll) - np.log(n_samples)
        eig  += (ln - ld_).sum()
    return float(eig / n_samples)


# =============================================================================
# Régression linéaire → Sigma_signal_free
# =============================================================================

def compute_Sigma_signal_free_linear(G, prior, Sigma_obs, n_samples, seed):
    """
    Sigma_signal_free via régression linéaire matricielle.
    Résout min_{A,b} E[||theta - A*G(theta) - b||^2].
    Retourne Sigma_signal_free = inv(Iy) avec
    Iy = Sigma_obs^{-1} (I - M_formula @ Sigma_obs^{-1}).
    """
    rng = np.random.default_rng(seed)
    Sigma_obs = np.asarray(Sigma_obs, dtype=float)

    thetas = rng.multivariate_normal(prior.mu, prior.Sigma, size=n_samples)
    X = np.asarray(Parallel(n_jobs=N_JOBS)(delayed(G)(th) for th in thetas))
    eps = rng.multivariate_normal(np.zeros(X.shape[1]), Sigma_obs, size=n_samples)
    Y = X + eps

    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    n = n_samples
    Sigma_XY = (Xc.T @ Yc) / n
    Sigma_YY = (Yc.T @ Yc) / n
    Sigma_X  = (Xc.T @ Xc) / n

    # M_formula = Sigma_X - Sigma_X (Sigma_X + Sigma_obs)^{-1} Sigma_X
    M_formula = Sigma_X - Sigma_X @ la.solve(Sigma_X + Sigma_obs, Sigma_X)

    Sobs_inv = la.inv(Sigma_obs)
    Iy = Sobs_inv @ (np.eye(N) - M_formula @ Sobs_inv)
    Sigma_signal_free = la.inv(Iy)
    return 0.5 * (Sigma_signal_free + Sigma_signal_free.T)


# =============================================================================
# NN → Sigma_signal_free_nn
# =============================================================================

class _SimpleNN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        layers, prev = [], dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _generate_chunk(G, n, prior, Sigma_obs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu  = np.asarray(prior.mu)
    L   = np.linalg.cholesky(np.asarray(prior.Sigma))
    z   = rng.standard_normal((n, len(mu)))
    thetas = mu + (L @ z.T).T
    X = np.asarray([G(th) for th in thetas])
    eps = rng.multivariate_normal(np.zeros(X.shape[1]), np.asarray(Sigma_obs), size=n)
    return X + eps, X   # Y, target


def compute_Sigma_signal_free_nn(G, prior, Sigma_obs, n_train, n_test,
                                  n_epochs, hidden, seed):
    """
    Sigma_signal_free_nn via réseau de neurones.
    Entraîne un NN G(theta) -> theta, estime Cov des résidus,
    retourne Sigma_signal_free_nn = inv(Iy_nn).
    """
    n_jobs = multiprocessing.cpu_count()
    Sigma_obs_arr = np.asarray(Sigma_obs, dtype=float)

    def _gen(n, s):
        chunk = max(1, n // n_jobs)
        sizes = [chunk] * n_jobs
        for i in range(n % n_jobs): sizes[i] += 1
        res = Parallel(n_jobs=n_jobs)(
            delayed(_generate_chunk)(G, sizes[i], prior, Sigma_obs_arr, s + i)
            for i in range(n_jobs)
        )
        Y_parts, T_parts = zip(*res)
        return (torch.tensor(np.concatenate(Y_parts), dtype=torch.float32),
                torch.tensor(np.concatenate(T_parts), dtype=torch.float32))

    print("    [NN] Génération train...")
    y_tr, t_tr = _gen(n_train, seed)
    print("    [NN] Génération test...")
    y_te, t_te = _gen(n_test, seed + 10000)

    loader_tr = DataLoader(TensorDataset(y_tr, t_tr), batch_size=64, shuffle=True, num_workers=2)
    loader_te = DataLoader(TensorDataset(y_te, t_te), batch_size=64, num_workers=2)

    model = _SimpleNN(N, hidden)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        for by, bt in loader_tr:
            opt.zero_grad()
            crit(model(by), bt).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                te_loss = sum(crit(model(by), bt).item() * by.size(0)
                              for by, bt in loader_te) / n_test
            print(f"    [NN] Epoch {epoch+1:3d} | Test MSE: {te_loss:.6f}")

    model.eval()
    residuals = []
    with torch.no_grad():
        for by, bt in loader_te:
            residuals.append(bt - model(by))
    residuals = torch.cat(residuals, dim=0).numpy()
    M_nn = (residuals.T @ residuals) / len(residuals)

    Sobs_inv = la.inv(Sigma_obs_arr)
    Iy_nn    = Sobs_inv @ (np.eye(N) - M_nn @ Sobs_inv)
    Ssf_nn   = la.inv(Iy_nn)
    return 0.5 * (Ssf_nn + Ssf_nn.T)


# =============================================================================
# Bornes EIG
# =============================================================================

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


def schur(Sigma, Wm):
    if Wm.size == 0:
        return Sigma
    G_mat = Wm.T @ Sigma @ Wm
    return Sigma - Sigma @ Wm @ la.inv(G_mat) @ Wm.T @ Sigma


def greedy_maximize_LB(Sigma_Y, Sigma_noise, n_sensors):
    """Sélection greedy forward maximisant la borne conservative BI."""
    N_grid    = Sigma_Y.shape[0]
    selected  = []
    remaining = list(range(N_grid))
    for _ in range(n_sensors):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            S  = selected + [idx]
            ix = np.ix_(S, S)
            sy, ly = la.slogdet(Sigma_Y[ix])
            sn, ln = la.slogdet(Sigma_noise[ix])
            if sy <= 0 or sn <= 0:
                continue
            score = 0.5 * (ly - ln)
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return np.asarray(selected, dtype=int)


def incremental_bounds(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, n_sensors):
    """Sélection incrémentale maximisant la LB, retourne indices + courbes LB/UB."""
    N_grid    = Sigma_Y.shape[0]
    cands     = [np.eye(N_grid)[:, i] for i in range(N_grid)]
    selected, remaining = [], list(range(N_grid))
    scores_lb, scores_ub = [], []
    eig_lb = eig_ub = 0.0

    for _ in range(n_sensors):
        Wm = (np.column_stack([cands[i] for i in selected])
              if selected else np.empty((N_grid, 0)))
        Ss  = schur(Sigma_signal,  Wm)
        Syt = schur(Sigma_Y_theta, Wm)
        SY  = schur(Sigma_Y,       Wm)
        Sn  = schur(Sigma_noise,   Wm)

        best_idx, best_lb, best_ub = None, -np.inf, None
        for idx in remaining:
            w = cands[idx]
            n_lb = float(w @ Ss  @ w)
            d_lb = float(w @ Syt @ w)
            n_ub = float(w @ SY  @ w)
            d_ub = float(w @ Sn  @ w)
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

    if len(scores_lb) < n_sensors:
        pad_val_lb = scores_lb[-1] if scores_lb else 0.0
        pad_val_ub = scores_ub[-1] if scores_ub else 0.0
        scores_lb.extend([pad_val_lb] * (n_sensors - len(scores_lb)))
        scores_ub.extend([pad_val_ub] * (n_sensors - len(scores_ub)))

    return np.asarray(selected, dtype=int), scores_lb, scores_ub


def incremental_bounds_given_W(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, W):
    """Bornes LB/UB pour un W donné (sans ré-optimiser la sélection)."""
    _, ls = la.slogdet(W.T @ Sigma_signal  @ W)
    _, lt = la.slogdet(W.T @ Sigma_Y_theta @ W)
    _, ly = la.slogdet(W.T @ Sigma_Y       @ W)
    _, ln = la.slogdet(W.T @ Sigma_noise   @ W)
    return 0.5 * (ls - lt), 0.5 * (ly - ln)


# =============================================================================
# Construction modèle / prior
# =============================================================================

def build_objects(lambda_):
    x_grid = np.linspace(0, 1, N + 2)[1:-1]
    model  = Burgers_CN(N=N, dt=DT, diffusivity=DIFFUSIVITY, lambda_=lambda_)
    make_u0(x_grid, "gaussian", center=U0_CENTER, width=U0_WIDTH, amplitude=U0_AMPLITUDE)

    kernel = Matern32(length_scale=KERNEL_LS, sigma=KERNEL_SIG)
    try:
        prior = GaussianProcessPrior(kernel, mu=np.zeros(N), nx=N)
    except TypeError:
        prior = GaussianProcessPrior(kernel, nx=N)
        prior.mu = np.zeros(N)

    noise     = NoiseModel(sigma_noise=SIGMA)
    Sigma_obs = noise.get_covariance(N)

    def G(theta):
        return model.evolve(u0=theta, n_steps=N_STEPS)[-1, :]

    return prior, Sigma_obs, G


# =============================================================================
# Calcul des bornes pour un triplet (Sigma_signal, Sigma_Y, indices_selection)
# =============================================================================

def _bounds_for_W(Sigma_signal, Sigma_Y, Sigma_obs, indices, budgets):
    """
    Pour un jeu de matrices fixé et des indices de sélection pré-calculés,
    retourne les 4 bornes (cons_lb, cons_ub, inc_lb, inc_ub) par budget.
    """
    cons_lb, cons_ub, inc_lb, inc_ub = [], [], [], []
    for budget in budgets:
        W, _ = build_selection_matrices(N, indices[:budget])
        cons_lb.append(eig_BI(Sigma_Y, Sigma_obs, W))
        cons_ub.append(eig_BS(Sigma_signal, Sigma_obs, W))
        lb, ub = incremental_bounds_given_W(
            Sigma_signal, Sigma_obs, Sigma_Y, Sigma_obs, W
        )
        inc_lb.append(lb)
        inc_ub.append(ub)
    return cons_lb, cons_ub, inc_lb, inc_ub


# =============================================================================
# Une répétition
# =============================================================================

def run_one_repeat(lambda_, seed, Sigma_signal_free, Sigma_signal_free_nn):
    """
    Calcule Sigma_signal (FD) + Sigma_Y pour ce seed,
    puis les bornes pour les 3 variantes et les 2 méthodes de sélection.

    Retourne un dict avec 12 listes de longueur n_budgets.
    """
    prior, Sigma_obs, G = build_objects(lambda_)
    budgets = SENSOR_BUDGETS
    max_b   = max(budgets)

    # --- Matrices propres à ce repeat ---
    l_theta      = estimate_E_JT(G, prior, N_SAMPLES, seed=seed)
    H_theta      = estimate_H_theta(G, prior, Sigma_obs, N_SAMPLES, seed=seed + 1)
    Sigma_Y      = estimate_Sigma_Y(G, prior, Sigma_obs, N_SAMPLES_SIGMA_Y, seed=seed + 2)
    Sigma_signal = compute_Sigma_signal(
        l_theta, H_theta, np.asarray(prior.Sigma), Sigma_obs
    )

    # --- Sélection incrémentale (basée sur Sigma_signal FD) ---
    indices_inc, scores_lb_inc, scores_ub_inc = incremental_bounds(
        Sigma_signal, Sigma_obs, Sigma_Y, Sigma_obs, max_b
    )

    # --- Sélection conservative (greedy LB) ---
    indices_cons = greedy_maximize_LB(Sigma_Y, Sigma_obs, max_b)

    result = {}

    # Méthode : incremental — variante : fd
    result["inc_fd_lb"] = scores_lb_inc[:max_b]
    result["inc_fd_ub"] = scores_ub_inc[:max_b]
    # Pour les bornes conservatives sur la même sélection inc :
    clb, cub, _, _ = _bounds_for_W(Sigma_signal, Sigma_Y, Sigma_obs, indices_inc, budgets)
    # (on les stocke dans cons_fd sur sélection incrémentale — voir note ci-dessous)

    # Méthode : conservative (indices_cons) — variantes fd / free / nn
    for key, Ss in [("fd", Sigma_signal),
                    ("free", Sigma_signal_free),
                    ("nn", Sigma_signal_free_nn)]:
        clb, cub, ilb, iub = _bounds_for_W(Ss, Sigma_Y, Sigma_obs, indices_cons, budgets)
        result[f"cons_{key}_lb"] = clb
        result[f"cons_{key}_ub"] = cub
        result[f"cons_{key}_inc_lb"] = ilb   # bornes incrémentales évaluées sur sélection cons
        result[f"cons_{key}_inc_ub"] = iub

    # Méthode : incremental (indices_inc) — variantes free / nn
    # (fd déjà fait via incremental_bounds directement)
    for key, Ss in [("free", Sigma_signal_free),
                    ("nn", Sigma_signal_free_nn)]:
        # ré-optimise la sélection pour cette variante de Sigma_signal
        idx_inc_var, slb, sub = incremental_bounds(
            Ss, Sigma_obs, Sigma_Y, Sigma_obs, max_b
        )
        result[f"inc_{key}_lb"] = slb[:max_b]
        result[f"inc_{key}_ub"] = sub[:max_b]

    # On extrait aussi les scores inc_fd au bon format (par budget, pas par pas)
    result["inc_fd_lb"] = [scores_lb_inc[b - 1] for b in budgets]
    result["inc_fd_ub"] = [scores_ub_inc[b - 1] for b in budgets]
    for key in ("free", "nn"):
        result[f"inc_{key}_lb"] = [result[f"inc_{key}_lb"][b - 1] for b in budgets]
        result[f"inc_{key}_ub"] = [result[f"inc_{key}_ub"][b - 1] for b in budgets]

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lam in LAMBDAS:
        print(f"\n{'='*60}")
        print(f"  lambda = {lam}")
        print(f"{'='*60}")

        prior, Sigma_obs, G = build_objects(lam)

        # ------------------------------------------------------------------
        # 1. EIG offset (une fois)
        # ------------------------------------------------------------------
        print("  [1/4] EIG offset...")
        if lam == 0.0:
            eig_offset = eig_linearise(G, prior, Sigma_obs)
            print(f"    eig_linearise = {eig_offset:.4f}")
        else:
            eig_offset = estimate_eig_nmc(G, prior, Sigma_obs,
                                           N_SAMPLES_EIG, seed=BASE_SEED)
            print(f"    eig_nmc = {eig_offset:.4f}")

        # ------------------------------------------------------------------
        # 2. Sigma_Y_ref (une fois, beaucoup de samples)
        # ------------------------------------------------------------------
        print("  [2/4] Sigma_Y_ref...")
        Sigma_Y_ref = estimate_Sigma_Y(G, prior, Sigma_obs,
                                        N_SAMPLES_SIGMA_Y_REF, seed=BASE_SEED)

        # ------------------------------------------------------------------
        # 3. Sigma_signal_free via régression linéaire (une fois)
        # ------------------------------------------------------------------
        print("  [3/4] Sigma_signal_free (régression linéaire)...")
        Sigma_signal_free = compute_Sigma_signal_free_linear(
            G, prior, Sigma_obs, n_samples=10000, seed=BASE_SEED
        )

        # ------------------------------------------------------------------
        # 4. Sigma_signal_free_nn via NN (une fois)
        # ------------------------------------------------------------------
        print("  [4/4] Sigma_signal_free_nn (NN)...")
        Sigma_signal_free_nn = compute_Sigma_signal_free_nn(
            G, prior, Sigma_obs,
            n_train=N_TRAIN, n_test=N_TEST,
            n_epochs=N_EPOCHS, hidden=HIDDEN,
            seed=BASE_SEED
        )

        # ------------------------------------------------------------------
        # 5. Répétitions
        # ------------------------------------------------------------------
        print(f"  Repeats (n={N_REPEATS})...")
        seeds = [BASE_SEED + 1000 * LAMBDAS.index(lam) + r
                 for r in range(N_REPEATS)]

        repeat_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_repeat)(lam, s, Sigma_signal_free, Sigma_signal_free_nn)
            for s in seeds
        )

        # ------------------------------------------------------------------
        # 6. Assemblage et sauvegarde
        # ------------------------------------------------------------------
        keys = [
            "inc_fd_lb", "inc_fd_ub",
            "inc_free_lb", "inc_free_ub",
            "inc_nn_lb", "inc_nn_ub",
            "cons_fd_lb", "cons_fd_ub",
            "cons_fd_inc_lb", "cons_fd_inc_ub",
            "cons_free_lb", "cons_free_ub",
            "cons_free_inc_lb", "cons_free_inc_ub",
            "cons_nn_lb", "cons_nn_ub",
            "cons_nn_inc_lb", "cons_nn_inc_ub",
        ]

        arrays = {}
        for k in keys:
            arrays[k] = np.array([r[k] for r in repeat_results])  # (n_repeats, n_budgets)

        fname = OUTPUT_DIR / f"results_lambda_{lam:.2f}.npz"
        np.savez(
            fname,
            # scalaires / matrices fixes
            eig_offset           = np.float64(eig_offset),
            Sigma_signal_free    = Sigma_signal_free,
            Sigma_signal_free_nn = Sigma_signal_free_nn,
            Sigma_Y_ref          = Sigma_Y_ref,
            sensor_budgets       = np.array(SENSOR_BUDGETS),
            lambda_val           = np.float64(lam),
            n_repeats            = np.int64(N_REPEATS),
            # arrays répétés (n_repeats, n_budgets)
            **arrays,
        )
        print(f"  Sauvegardé : {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main() 