from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyboed_mpl")

import numpy as np
import numpy.linalg as la
from scipy import linalg as sla
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
N_REPEATS      = 1
BASE_SEED      = 42
N_JOBS         = -1

# Modèle
N           = 200
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
N_SAMPLES_EIG         = 500  # eig

# NN
N_TRAIN  = 100000
N_TEST   = 2000
N_EPOCHS = 30
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

def compute_Sigma_signal_misfit(l_theta, H_theta, Sigma_theta, Sigma_obs):

    # Symmetric square root of Sigma_theta
    eigvals, eigvects = np.linalg.eigh(Sigma_theta)

    # Numerical safeguard
    eigvals = np.clip(eigvals, 0.0, None)

    Sigma_theta_sqrt = (
        eigvects
        @ np.diag(np.sqrt(eigvals))
        @ eigvects.T
    )

    H_misfit = (
        Sigma_theta_sqrt
        @ H_theta
        @ Sigma_theta_sqrt
    )

    I_plus_H_misfit = np.eye(H_misfit.shape[0]) + H_misfit

    A = (
        Sigma_theta_sqrt
        @ la.solve(I_plus_H_misfit, Sigma_theta_sqrt)
    )

    X = la.solve(A, l_theta)

    return Sigma_obs + l_theta.T @ X



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

# ----------------------------------------------------------------------
# 1.  Prior via KL expansion
# ----------------------------------------------------------------------
class KLPrior:
    """
    Prior = N(mu, Sigma) → représentation KL tronquée.
    theta = mu + V_k @ diag(sqrt(lam_k)) @ xi,   xi ~ N(0, I_k)
    """
    def __init__(self, mu, Sigma, energy=0.9999):
        self.mu_full = np.asarray(mu)
        ev, V = np.linalg.eigh(np.asarray(Sigma))
        ev    = ev[::-1];  V = V[:, ::-1]          # décroissant
        cumvar = np.cumsum(ev) / ev.sum()
        self.k = int(np.searchsorted(cumvar, energy)) + 1
        self.V_k      = V[:, :self.k]              # (d, k)
        self.lam_k    = ev[:self.k]                # (k,)
        self.sqrt_lam = np.sqrt(self.lam_k)        # (k,)
        # prior dans l'espace réduit : N(0, I_k)
        self.mu    = np.zeros(self.k)
        self.Sigma = np.eye(self.k)
        self.L     = np.eye(self.k)
        self.Sinv  = np.eye(self.k)
        captured = cumvar[self.k-1] * 100
        print(f"KL tronquée : {self.k} modes, {captured:.3f}% variance")

    def xi_to_theta(self, xi):
        """xi (k,) → theta (d,)"""
        return self.mu_full + self.V_k @ (self.sqrt_lam * xi)

    def make_G_reduced(self, G):
        """G_r(xi) = G( xi_to_theta(xi) )"""
        return lambda xi: G(self.xi_to_theta(xi))

    def make_J_reduced(self, J_G):
        """J_r(xi) = J_G( theta ) @ V_k @ diag(sqrt_lam)"""
        def J_reduced(xi):
            theta = self.xi_to_theta(xi)
            J = np.asarray(J_G(theta))          # (m, d)
            return J @ self.V_k @ np.diag(self.sqrt_lam)   # (m, k)
        return J_reduced


# ----------------------------------------------------------------------
# 2.  Estimateur principal : KL + marginale analytique + correction
# ----------------------------------------------------------------------
def estimate_eig_kl(G, J_G, kl_prior, Sigma_obs, n_outer, seed=None, n_jobs=-1):
    """
    Estimateur sans biais (à la linéarisation près) pour l'EIG.
    Fonctionne pour G linéaire ou faiblement non‑linéaire.
    """
    G_r   = kl_prior.make_G_reduced(G)
    J_r   = kl_prior.make_J_reduced(J_G)
    k     = kl_prior.k
    m     = len(Sigma_obs)

    # pré‑calculs pour Sigma_obs
    L_obs    = sla.cholesky(Sigma_obs, lower=True)
    Sinv_obs = sla.cho_solve((L_obs, True), np.eye(m))
    _, ld    = la.slogdet(Sigma_obs)
    log_c    = -0.5 * (ld + m * np.log(2 * np.pi))

    # seeds reproductibles
    ss    = np.random.SeedSequence(seed)
    seeds = ss.spawn(n_outer)

    def _step(seed_i):
        rng = np.random.default_rng(seed_i)

        # 1. xi ~ N(0, I_k)
        xi = rng.standard_normal(k)
        Gr = np.asarray(G_r(xi))
        Jr = np.asarray(J_r(xi))

        # 2. Matrice marginale S = Jr Jr^T + Sigma_obs
        S      = Jr @ Jr.T + Sigma_obs
        L_S    = sla.cholesky(S, lower=True)
        Sinv_S = sla.cho_solve((L_S, True), np.eye(m))
        _, ld_S = la.slogdet(S)
        log_c_S = -0.5 * (ld_S + m * np.log(2 * np.pi))

        # 3. Correction du biais quadratique
        #    = ½ tr( (S^{-1} - Σ_obs^{-1}) Σ_obs )
        bias_corr = 0.5 * (np.trace(Sinv_S @ Sigma_obs) - m)

        # 4. Simuler y
        eps = rng.multivariate_normal(np.zeros(m), Sigma_obs)
        y   = Gr + eps

        # 5. log p(y | xi)
        log_lik = log_c - 0.5 * np.dot(eps, Sinv_obs @ eps)

        # 6. log p(y) par marginale analytique
        r = y - Gr
        log_py = log_c_S - 0.5 * r @ Sinv_S @ r

        return float(log_lik - log_py - bias_corr)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_step)(seeds[i]) for i in range(n_outer)
    )
    return float(np.mean(results))


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



def greedy_maximize_LB(Sigma_Y, Sigma_noise, n_sensors):
    """
    Greedy forward sensor selection maximizing
    0.5 * log det(Sigma_Y) - 0.5 * log det(Sigma_noise)
    """

    Sigma_Y = 0.5 * (Sigma_Y + Sigma_Y.T)
    Sigma_noise = 0.5 * (Sigma_noise + Sigma_noise.T)

    N = Sigma_Y.shape[0]

    selected = []
    remaining = set(range(N))

    def logdet_spd(A):
        L = np.linalg.cholesky(A)
        return 2.0 * np.sum(np.log(np.diag(L)))

    for _ in range(n_sensors):

        best_idx = None
        best_score = -np.inf

        for idx in remaining:

            S = selected + [idx]
            ix = np.ix_(S, S)

            logdet_y = logdet_spd(Sigma_Y[ix])
            logdet_n = logdet_spd(Sigma_noise[ix])

            score = 0.5 * (logdet_y - logdet_n)

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(selected, dtype=int)


def schur(Sigma, Wm):
    if Wm.size == 0:
        return Sigma

    G = Wm.T @ Sigma @ Wm
    correction = Sigma @ Wm @ np.linalg.solve(G, Wm.T @ Sigma)

    return Sigma - correction


def incremental_bounds(
    Sigma_signal:  np.ndarray,
    Sigma_Y_theta: np.ndarray,
    Sigma_Y:       np.ndarray,
    Sigma_noise:   np.ndarray,
    n_sensors:     int,
):
    N = Sigma_Y.shape[0]
    W_candidates = [np.eye(N)[:, i] for i in range(N)]

    selected   = []
    remaining  = list(range(N))

    inc_inf = []
    inc_sup = []

    eig_inf = 0.0
    eig_sup = 0.0

    for m in range(n_sensors):

        # W_m
        if selected:
            Wm = np.column_stack([W_candidates[i] for i in selected])
        else:
            Wm = np.empty((N, 0))

        # matrices conditionnelles
        Sigma_s_m   = schur(Sigma_signal,  Wm)
        Sigma_yth_m = schur(Sigma_Y_theta, Wm)
        Sigma_Y_m   = schur(Sigma_Y,       Wm)
        Sigma_n_m   = schur(Sigma_noise,   Wm)

        best_idx = None
        best_inc = -np.inf
        best_sup = None

        for idx in remaining:
            w = W_candidates[idx]

            num_inf = float(w.T @ Sigma_s_m   @ w)
            den_inf = float(w.T @ Sigma_yth_m @ w)

            num_sup = float(w.T @ Sigma_Y_m @ w)
            den_sup = float(w.T @ Sigma_n_m @ w)

            if min(num_inf, den_inf, num_sup, den_sup) <= 0:
                continue

            d_inf = 0.5 * np.log(num_inf / den_inf)
            d_sup = 0.5 * np.log(num_sup / den_sup)

            if d_inf > best_inc:
                best_inc = d_inf
                best_sup = d_sup
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        eig_inf += best_inc
        eig_sup += best_sup

        inc_inf.append(best_inc)
        inc_sup.append(best_sup)

    return {
        "indices": np.array(selected, dtype=int),
        "increments_lower": inc_inf,
        "increments_upper": inc_sup,
        "EIG_lower_bound": eig_inf,
        "EIG_upper_bound": eig_sup,
    }


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

def _bounds_for_W(Sigma_signal, Sigma_Y, Sigma_obs, indices, budgets, method="cons"):
    """
    Pour un jeu de matrices fixé et des indices de sélection pré-calculés,
    retourne les 4 bornes (cons_lb, cons_ub, inc_lb, inc_ub) par budget.
    """
    cons_lb, cons_ub, inc_lb, inc_ub = [], [], [], []
    if method == "cons" : 
        for budget in budgets:
            W, _ = build_selection_matrices(N, indices[:budget])
            cons_lb.append(eig_BI(Sigma_Y, Sigma_obs, W))
            cons_ub.append(eig_BS(Sigma_signal, Sigma_obs, W))
            lb, ub = incremental_bounds_given_W(
                Sigma_signal, Sigma_obs, Sigma_Y, Sigma_obs, W
            )
            inc_lb.append(lb)
            inc_ub.append(ub)
    elif method  == "inc" :
        for budget in budgets:
            result = incremental_bounds(
                Sigma_signal  = Sigma_signal,
                Sigma_Y_theta = Sigma_obs,
                Sigma_Y       = Sigma_Y,
                Sigma_noise   = Sigma_obs,
                n_sensors     = budget,       
            )
            W, _ = build_selection_matrices(N, result['indices'])

            lb = eig_BI(Sigma_Y, Sigma_obs, W)
            ub = eig_BS(Sigma_signal, Sigma_obs, W)

            cons_lb.append(lb)
            cons_ub.append(ub)

            inc_lb.append(result["EIG_lower_bound"])
            inc_ub.append(result["EIG_upper_bound"])
    return cons_lb, cons_ub, inc_lb, inc_ub




# =============================================================================
# Une répétition
# =============================================================================

def run_one_repeat(lambda_, seed, eig_offset, Sigma_signal_free, Sigma_signal_free_nn):
    """
    Calcule Sigma_signal (FD) + Sigma_Y pour ce seed,
    puis les bornes pour les 3 variantes et les 2 méthodes de sélection.
    L'offset EIG est sommé à toutes les bornes.

    Retourne un dict avec 12 listes de longueur n_budgets.
    """
    prior, Sigma_obs, G = build_objects(lambda_)
    budgets = SENSOR_BUDGETS
    max_b   = max(budgets)

    # --- Matrices propres à ce repeat ---
    l_theta      = estimate_E_JT(G, prior, N_SAMPLES, seed=seed)
    H_theta      = estimate_H_theta(G, prior, Sigma_obs, N_SAMPLES, seed=seed + 1)
    Sigma_Y      = estimate_Sigma_Y(G, prior, Sigma_obs, N_SAMPLES_SIGMA_Y, seed=seed + 2)
    if lambda_ == 0 :
        Sigma_signal = compute_Sigma_signal(
            l_theta, H_theta, np.asarray(prior.Sigma), Sigma_obs
        )
    else:
        Sigma_signal = compute_Sigma_signal_misfit(
            l_theta, H_theta, np.asarray(prior.Sigma), Sigma_obs
        )

    # --- Sélection incrémentale (basée sur Sigma_signal FD) ---
    result_inc = incremental_bounds(
        Sigma_signal, Sigma_obs, Sigma_Y, Sigma_obs, max_b
    )

    # --- Sélection conservative (greedy LB) ---
    indices_cons = greedy_maximize_LB(Sigma_Y, Sigma_obs, max_b)

    result = {}

    # Méthode : conservative (indices_cons) — variantes fd / free / nn
    # L'offset est sommé uniquement aux bornes conservatives (BI et BS)
    # method : si indices cons ou inc
    for method in ["cons","inc"]:
        if method == "cons" : 
            indices = indices_cons
        elif method == "inc":
            indices = result_inc["indices"]

        for key, Ss in [("fd", Sigma_signal),
                        ("free", Sigma_signal_free),
                        ("nn", Sigma_signal_free_nn)]:
            clb, cub, ilb, iub = _bounds_for_W(Ss, Sigma_Y, Sigma_obs, 
                                indices, budgets, method)
            
            result[f"{method}_{key}_lb"]     = [v + eig_offset for v in clb]
            result[f"{method}_{key}_ub"]     = [v + eig_offset for v in cub]
            result[f"{method}_{key}_inc_lb"] = ilb
            result[f"{method}_{key}_inc_ub"] = iub

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
            # eig_offset = estimate_eig_nmc(G, prior, Sigma_obs,
            #                                N_SAMPLES_EIG, seed=BASE_SEED)
            kl = KLPrior(prior.mu, prior.Sigma, energy=0.9999)

            from functools import partial
            J_G = partial(jacobian_fd, G) 
            eig_offset = estimate_eig_kl(G, J_G, kl, 
                                         Sigma_obs, N_SAMPLES_EIG, 
                                         seed=BASE_SEED, n_jobs=-1)
            print(f"EIG = {eig_offset:.4f}")

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
            delayed(run_one_repeat)(lam, s, eig_offset, Sigma_signal_free, Sigma_signal_free_nn)
            for s in seeds
        )

        # ------------------------------------------------------------------
        # 6. Assemblage et sauvegarde
        # ------------------------------------------------------------------
        keys = [
            "cons_fd_lb", "cons_fd_ub",
            "cons_fd_inc_lb", "cons_fd_inc_ub",
            "cons_free_lb", "cons_free_ub",
            "cons_free_inc_lb", "cons_free_inc_ub",
            "cons_nn_lb", "cons_nn_ub",
            "cons_nn_inc_lb", "cons_nn_inc_ub",
            "inc_fd_lb", "inc_fd_ub",
            "inc_fd_inc_lb", "inc_fd_inc_ub",
            "inc_free_lb", "inc_free_ub",
            "inc_free_inc_lb", "inc_free_inc_ub",
            "inc_nn_lb", "inc_nn_ub",
            "inc_nn_inc_lb", "inc_nn_inc_ub",
                    ]

        arrays = {}
        for k in keys:
            arrays[k] = np.array([r[k] for r in repeat_results])  # (n_repeats, n_budgets)

        fname = OUTPUT_DIR / f"results_lambda_{lam:.2f}.npz"
        np.savez(
            fname,
            # scalaires / matrices fixes
            eig_offset           = np.float64(eig_offset),
            # Sigma_signal_free    = Sigma_signal_free,
            # Sigma_signal_free_nn = Sigma_signal_free_nn,
            # Sigma_Y_ref          = Sigma_Y_ref,
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
