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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NN] Device : {DEVICE}")

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
N_TRAIN    = 100000
N_TEST     = 2000
N_RESIDUAL = 50000   # lot dédié à l'estimation de M (>> N)
N_EPOCHS   = 30
HIDDEN     = [64, 64]

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

def compute_Sigma_signal_free_linear(
    G,
    prior,
    Sigma_obs,
    n_obs: int,
    n_samples: int,
    seed: int,
    use_empirical_formula: bool = False,
) -> np.ndarray:
    """
    Estime Sigma_signal_free via régression linéaire.

    Résout min_{A,b} E[||theta - A*Y - b||^2], ce qui donne
    la covariance résiduelle M = Cov(theta) - Cov(theta,Y) Cov(Y,Y)^{-1} Cov(Y,theta).

    Sous hypothèse gaussienne / G linéaire, M se simplifie en :
        M = Sigma_X - Sigma_X (Sigma_X + Sigma_obs)^{-1} Sigma_X

    Retourne Sigma_sf = sym( I_y^{-1} ) avec
        I_y = Sigma_obs^{-1} (I - M Sigma_obs^{-1})

    Parameters
    ----------
    use_empirical_formula : si True, utilise M = Sigma_X - Sigma_XY Sigma_YY^{-1} Sigma_XY.T
                            (valable pour G non-linéaire, purement empirique)
    """
    rng = np.random.default_rng(seed)
    Sigma_obs_arr = np.asarray(Sigma_obs, dtype=float)

    thetas = rng.multivariate_normal(prior.mu, prior.Sigma, size=n_samples)
    X = np.array(Parallel(n_jobs=N_JOBS)(delayed(G)(th) for th in thetas))

    eps = rng.multivariate_normal(np.zeros(n_obs), Sigma_obs_arr, size=n_samples)
    Y = X + eps

    # Statistiques empiriques centrées, diviseur n-1 (non-biaisé)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    denom = n_samples - 1

    m_X = X.mean(axis=0)
    m_Y = Y.mean(axis=0)
    Sigma_X  = (Xc.T @ Xc) / denom
    Sigma_XY = (Xc.T @ Yc) / denom
    Sigma_YY = (Yc.T @ Yc) / denom

    # Estimateur MMSE linéaire : f*(Y) = A* Y + b*
    A_star = Sigma_XY @ la.solve(Sigma_YY, np.eye(Sigma_YY.shape[0]))
    b_star = m_X - A_star @ m_Y

    if use_empirical_formula:
        M = Sigma_X - Sigma_XY @ la.solve(Sigma_YY, Sigma_XY.T)
    else:
        M = Sigma_X - Sigma_X @ la.solve(Sigma_X + Sigma_obs_arr, Sigma_X)

    Sobs_inv = la.inv(Sigma_obs_arr)
    Iy  = Sobs_inv @ (np.eye(n_obs) - M @ Sobs_inv)
    Ssf = la.inv(Iy)

    return 0.5 * (Ssf + Ssf.T), A_star, b_star


# =============================================================================
# NN → Sigma_signal_free_nn
# =============================================================================

class LinearPlusNN(nn.Module):
    """Modèle f(Y) = A* Y + b* + NN(Y).

    La partie linéaire (MMSE linéaire) est fixe et non-entraînable.
    Le NN apprend uniquement la correction non-linéaire résiduelle.
    """
    def __init__(self, A_star: np.ndarray, b_star: np.ndarray, hidden: list[int]):
        super().__init__()
        n_obs = A_star.shape[0]
        self.register_buffer('A', torch.tensor(A_star, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b_star, dtype=torch.float32))
        layers, prev = [], n_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_obs))
        self.correction = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.A.T + self.b + self.correction(y)


def _generate_chunk(
    G, n: int, prior, Sigma_obs: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n (Y, G(theta)) pairs for one parallel chunk."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(prior.mu)
    L  = np.linalg.cholesky(np.asarray(prior.Sigma))
    z  = rng.standard_normal((n, len(mu)))
    thetas = mu + (L @ z.T).T                          # (n, d_theta)
    X = np.array([G(th) for th in thetas])             # (n, N)
    eps = rng.multivariate_normal(np.zeros(X.shape[1]), Sigma_obs, size=n)
    return X + eps, X   # Y, target = G(theta)


def _generate_dataset(
    G, n: int, prior, Sigma_obs: np.ndarray, seed: int, n_jobs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallelise chunk generation and concatenate into tensors."""
    chunk = max(1, n // n_jobs)
    sizes = [chunk] * n_jobs
    for i in range(n % n_jobs):
        sizes[i] += 1

    results = Parallel(n_jobs=n_jobs)(
        delayed(_generate_chunk)(G, sizes[i], prior, Sigma_obs, seed + i)
        for i in range(n_jobs)
    )
    Y_parts, T_parts = zip(*results)
    Y = torch.tensor(np.concatenate(Y_parts), dtype=torch.float32)
    T = torch.tensor(np.concatenate(T_parts), dtype=torch.float32)
    return Y, T


def compute_Sigma_signal_free_nn(
    G,
    prior,
    Sigma_obs,
    A_star: np.ndarray,
    b_star: np.ndarray,
    n_train: int,
    n_test: int,
    n_residual: int,
    n_epochs: int,
    hidden: list[int],
    seed: int,
) -> np.ndarray:
    """
    Estimate Sigma_signal_free via a neural network surrogate.

    Trains NN: Y -> G(theta), then uses the empirical residual covariance
    M = Cov(G(theta) - NN(Y)) to form the signal-free Fisher information

        I_y = Sigma_obs^{-1} (I - M Sigma_obs^{-1})

    and returns Sigma_sf = sym( I_y^{-1} ).

    Parameters
    ----------
    G         : callable, theta -> R^n_obs
    prior     : object with attributes .mu and .Sigma
    Sigma_obs : (n_obs, n_obs) observation noise covariance
    n_obs     : observation/parameter space dimension
    n_train   : number of training samples
    n_test    : number of test samples (used for residual estimation)
    n_epochs  : training epochs
    hidden    : list of hidden layer widths
    seed      : base random seed
    """
    n_jobs = multiprocessing.cpu_count()
    Sigma_obs_arr = np.asarray(Sigma_obs, dtype=float)
    n_obs = Sigma_obs_arr.shape[0]

    print("    [NN] Generating training data...")
    y_tr, t_tr = _generate_dataset(G, n_train, prior, Sigma_obs_arr, seed, n_jobs)

    print("    [NN] Generating test data (MSE validation)...")
    y_te, t_te = _generate_dataset(G, n_test, prior, Sigma_obs_arr, seed + 10_000, n_jobs)

    bs_train = 256 if DEVICE.type == "cuda" else 64
    loader_tr = DataLoader(TensorDataset(y_tr, t_tr), batch_size=bs_train, shuffle=True, num_workers=0)
    loader_te = DataLoader(TensorDataset(y_te, t_te), batch_size=512, num_workers=0)

    # Partie linéaire fixe + correction NN → sur GPU si disponible
    model = LinearPlusNN(A_star=A_star, b_star=b_star, hidden=hidden).to(DEVICE)
    opt  = optim.Adam(model.correction.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        for by, bt in loader_tr:
            by, bt = by.to(DEVICE), bt.to(DEVICE)
            opt.zero_grad()
            crit(model(by), bt).backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                te_loss = sum(
                    crit(model(by.to(DEVICE)), bt.to(DEVICE)).item() * by.size(0)
                    for by, bt in loader_te
                ) / n_test
            print(f"    [NN] Epoch {epoch+1:3d} | Test MSE: {te_loss:.6f}")

    # --- Residual covariance sur un lot dédié (n_residual >> n_obs) ---
    print(f"    [NN] Estimating residual covariance ({n_residual} samples)...")
    y_res, t_res = _generate_dataset(G, n_residual, prior, Sigma_obs_arr, seed + 20_000, n_jobs)
    loader_res = DataLoader(TensorDataset(y_res, t_res), batch_size=1024, num_workers=0)

    model.eval()
    residuals_list = []
    with torch.no_grad():
        for by, bt in loader_res:
            by, bt = by.to(DEVICE), bt.to(DEVICE)
            residuals_list.append((bt - model(by)).cpu().numpy())

    residuals = np.concatenate(residuals_list, axis=0)   # (n_residual, n_obs)
    r_mean    = residuals.mean(axis=0, keepdims=True)
    M = ((residuals - r_mean).T @ (residuals - r_mean)) / (len(residuals) - 1)

    # --- Signal-free Fisher information ---
    # I_y = Sigma_obs^{-1} (I - M Sigma_obs^{-1})
    Sobs_inv = la.inv(Sigma_obs_arr)
    Iy  = Sobs_inv @ (np.eye(n_obs) - M @ Sobs_inv)
    Ssf = la.inv(Iy)

    return 0.5 * (Ssf + Ssf.T)   # enforce exact symmetry


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



def schur(Sigma: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    """Complément de Schur conditionnel.

    Calcule Σ(Wm) = Σ - Σ Wm (WmᵀΣWm)⁻¹ Wmᵀ Σ,
    c'est-à-dire la covariance de Σ conditionnée sur les directions Wm.

    Utilise une factorisation de Cholesky pour garantir que le résultat
    reste symétrique défini positif numériquement.

    Parameters
    ----------
    Sigma : (N, N) ndarray, symétrique définie positive.
    Wm    : (N, m) ndarray. Si m=0 (matrice vide), retourne Sigma inchangé.

    Returns
    -------
    (N, N) ndarray : Σ(Wm), symétrique définie positive.
    """
    if Wm.size == 0:
        return Sigma
    G = Wm.T @ Sigma @ Wm          # (m, m)  SDP
    L = np.linalg.cholesky(G)      # G = LLᵀ
    B = np.linalg.solve(L, Wm.T @ Sigma)   # (m, N)  B = L⁻¹ WᵀΣ
    return Sigma - B.T @ B         # Σ - BᵀB  symétrique par construction


def incremental_bounds(
    Sigma_signal:        np.ndarray,
    Sigma_Y_given_theta: np.ndarray,
    Sigma_Y:             np.ndarray,
    Sigma_noise:         np.ndarray,
    n_sensors:           int,
) -> dict:
    """Placement glouton de capteurs par bornes incrémentales sur l'EIG.

    Implémente le théorème suivant : pour W_new = eᵢ,

        EIG(Wm ∪ {i}) ≥ EIG(Wm) + ½ log [Σ_signal(Wm)]ᵢᵢ / [Σ_{Y|θ}(Wm)]ᵢᵢ   (borne inf)
        EIG(Wm ∪ {i}) ≤ EIG(Wm) + ½ log [Σ_Y(Wm)]ᵢᵢ     / [Σ_noise(Wm)]ᵢᵢ    (borne sup)

    où Σ_*(Wm) désigne le complément de Schur de Σ_* conditionné sur Wm
    (cf. équations (11)–(14) du théorème de référence).

    La sélection gloutonne maximise la borne inférieure à chaque étape.
    La borne supérieure accumulée correspond à la même séquence de capteurs
    (elle n'est pas nécessairement la borne sup glouton-optimale).

    Parameters
    ----------
    Sigma_signal        : (N, N)  Covariance a priori du paramètre θ.  Éq. (11).
    Sigma_Y_given_theta : (N, N)  Covariance de Y|θ (bruit d'observation).  Éq. (13).
                          ⚠ Ne pas confondre avec Σ_{θ|Y} (posterior sur θ).
    Sigma_Y             : (N, N)  Covariance marginale de Y.  Éq. (14).
    Sigma_noise         : (N, N)  Covariance du bruit capteur.  Éq. (12).
    n_sensors           : int     Nombre de capteurs à sélectionner.

    Returns
    -------
    dict avec les clés :
        "indices"          : (n_sensors,) int    — indices sélectionnés dans l'ordre.
        "increments_lower" : list[float]         — δ⁻(i*_m, m) à chaque étape.
        "increments_upper" : list[float]         — δ⁺(i*_m, m) à chaque étape.
        "EIG_lower_bound"  : float               — borne inf cumulée sur EIG(S_n).
        "EIG_upper_bound"  : float               — borne sup cumulée sur EIG(S_n).
    """
    N = Sigma_Y.shape[0]
    assert Sigma_signal.shape        == (N, N), "Sigma_signal: dimension incorrecte"
    assert Sigma_Y_given_theta.shape == (N, N), "Sigma_Y_given_theta: dimension incorrecte"
    assert Sigma_noise.shape         == (N, N), "Sigma_noise: dimension incorrecte"
    assert 1 <= n_sensors <= N,                 "n_sensors hors borne"

    # Copies de travail pour les mises à jour de Schur incrémentales.
    # À l'étape m, S_*[i,i] = [Σ_*(W_m)]ᵢᵢ sans recalcul depuis zéro.
    S_s   = Sigma_signal.copy()
    S_yth = Sigma_Y_given_theta.copy()
    S_Y   = Sigma_Y.copy()
    S_n   = Sigma_noise.copy()

    selected:  list[int]   = []
    remaining: list[int]   = list(range(N))
    inc_inf:   list[float] = []
    inc_sup:   list[float] = []
    eig_inf = eig_sup = 0.0

    for _ in range(n_sensors):
        best_idx  = None
        best_dinf = -np.inf
        best_dsup = None

        for idx in remaining:
            num_inf = S_s[idx, idx]
            den_inf = S_yth[idx, idx]
            num_sup = S_Y[idx, idx]
            den_sup = S_n[idx, idx]

            # Les variances conditionnelles doivent être > 0 (SDP).
            # Une valeur nulle ou négative signale une dégénérescence numérique.
            if min(num_inf, den_inf, num_sup, den_sup) <= 1e-14:
                warnings.warn(
                    f"Variance conditionnelle quasi-nulle pour le capteur {idx} "
                    f"(num_inf={num_inf:.2e}, den_inf={den_inf:.2e}, "
                    f"num_sup={num_sup:.2e}, den_sup={den_sup:.2e}). "
                    "Capteur ignoré — vérifier le conditionnement des matrices.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            d_inf = 0.5 * np.log(num_inf / den_inf)
            d_sup = 0.5 * np.log(num_sup / den_sup)

            if d_inf > best_dinf:
                best_dinf = d_inf
                best_dsup = d_sup
                best_idx  = idx

        if best_idx is None:
            warnings.warn(
                f"Aucun capteur valide à l'étape {len(selected)+1}/{n_sensors}. "
                "Arrêt prématuré.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # Mise à jour incrémentale de Schur par rang 1 :
        #   Σ_*(W_{m+1}) = Σ_*(W_m) - [Σ_*(W_m) eᵢ eᵢᵀ Σ_*(W_m)] / [Σ_*(W_m)]ᵢᵢ
        # Complexité O(N²) par étape au lieu de O(N³).
        for S in (S_s, S_yth, S_Y, S_n):
            col = S[:, best_idx].copy()          # Σ eᵢ
            S -= np.outer(col, col) / S[best_idx, best_idx]

        selected.append(best_idx)
        remaining.remove(best_idx)
        eig_inf += best_dinf
        eig_sup += best_dsup
        inc_inf.append(best_dinf)
        inc_sup.append(best_dsup)

    return {
        "indices":          np.array(selected, dtype=int),
        "increments_lower": inc_inf,
        "increments_upper": inc_sup,
        "EIG_lower_bound":  eig_inf,
        "EIG_upper_bound":  eig_sup,
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
                Sigma_Y_given_theta = Sigma_obs,
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

        for key, Ss in [# ("fd", Sigma_signal),      # déjà calculé séparément
                        # ("free", Sigma_signal_free), # idem
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
        Sigma_signal_free, A_star, b_star = compute_Sigma_signal_free_linear(
            G, prior, Sigma_obs, n_obs=N, n_samples=10000, seed=BASE_SEED
        )

        # ------------------------------------------------------------------
        # 4. Sigma_signal_free_nn via NN linéaire + correction (une fois)
        # ------------------------------------------------------------------
        print("  [4/4] Sigma_signal_free_nn (linéaire + NN)...")
        Sigma_signal_free_nn = compute_Sigma_signal_free_nn(
            G, prior, Sigma_obs,
            A_star=A_star, b_star=b_star,
            n_train=N_TRAIN, n_test=N_TEST, n_residual=N_RESIDUAL,
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
            "cons_nn_lb", "cons_nn_ub",
            "cons_nn_inc_lb", "cons_nn_inc_ub",
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
