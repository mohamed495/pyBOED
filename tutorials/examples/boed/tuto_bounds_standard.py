"""
Bayesian Optimal Experimental Design — Burgers Equation (Standard Setting)
===========================================================================
Pipeline:
  1. Setup        — PDE model, noise, discretisation
  2. Prior        — GP prior sur theta
  3. Forward      — G : theta -> u(·, T)
  4. Jacobians FD — J_theta et espérances MC
  5. Covariances  — Sigma_Y, Sigma_signal
  6. Gradient-free — estimation via régression linéaire + réseau de neurones
  7. Estimateurs EIG — bornes BS et BI
  8. Placement de capteurs — glouton + bornes incrémentales
"""

# =============================================================================
# 1. Imports
# =============================================================================
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing

from boed.priors.kernels import Matern32, Matern12
from boed.priors.gp_priors import GaussianProcessPrior
from boed.core.noise import NoiseModel, ColoredNoise
from boed.core import make_u0
from boed.pde.burgers import Burgers_CN
from boed.utils.observation import build_selection_matrices

# =============================================================================
# 2. Setup
# =============================================================================
np.random.seed(42)

N        = 50       # nombre de points intérieurs
dt       = 0.001
n_steps  = 100
sigma    = 0.1      # écart-type du bruit
lambda_  = 0.5
n_samples = 500

x_grid = np.linspace(0, 1, N + 2)[1:-1]

model   = Burgers_CN(N=N, dt=dt, diffusivity=0.02, lambda_=lambda_)
u0_true = make_u0(x_grid, "gaussian", center=0.25, width=0.07, amplitude=1.5)
noise   = NoiseModel(sigma_noise=sigma)


# =============================================================================
# 3. Prior
# =============================================================================
kernel = Matern32(length_scale=0.2, sigma=1.0)
prior  = GaussianProcessPrior(kernel, mu=np.zeros(N), nx=N)


# =============================================================================
# Utilitaires
# =============================================================================
def plot_matrices(matrices, titles, figsize=None, cmap="viridis"):
    """Affiche une rangée de matrices avec colorbar individuelle."""
    n = len(matrices)
    if figsize is None:
        figsize = (5 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()



# =============================================================================
# 4. Modèle direct
# =============================================================================
def make_forward_model(pde_model, n_steps):
    """Retourne l'opérateur forward G(theta) -> u(·, T)."""
    def G(theta: np.ndarray):
        return pde_model.evolve(u0=theta, n_steps=n_steps)[-1, :]
    return G


G = make_forward_model(pde_model=model, n_steps=n_steps)



# =============================================================================
# 5. Jacobien par différences finies
# =============================================================================
def jacobian_fd(G, theta, h=None):
    """Jacobien central FD de G par rapport à theta.

    Returns
    -------
    J_theta : ndarray, shape (m, d)
    """
    d = theta.shape[0]
    if h is None:
        h = np.finfo(float).eps ** (1 / 3) * (la.norm(theta) + 1e-8)

    G_theta = G(theta)
    m = G_theta.shape[0]
    J = np.zeros((m, d))
    for j in range(d):
        e_j = np.zeros(d)
        e_j[j] = 1.0
        J[:, j] = (G(theta + h * e_j) - G(theta - h * e_j)) / (2 * h)
    return J


# =============================================================================
# 6. Estimation MC de l_theta et H_theta
# =============================================================================
def estimate_E_JT(G, prior, n_samples, eps=1e-6):
    """Estimation MC de E[J_theta^T] via Halton quasi-MC.

    Returns
    -------
    ndarray (d, m)
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    sampler = qmc.Halton(d=len(prior.mu))
    samples_unif = sampler.random(n=n_samples)
    u_normal = norm.ppf(samples_unif)
    L = np.linalg.cholesky(prior.Sigma)
    theta_samples = prior.mu + (L @ u_normal.T).T  # (n_samples, d)
    
    Jacobians = Parallel(n_jobs=-1)(
        delayed(jacobian_fd)(G, theta, eps) for theta in theta_samples
    )

    sum_JT = np.sum([J.T for J in Jacobians], axis=0)
    return sum_JT / n_samples


def estimate_jacobian_covariances_mc(
    G, prior, Sigma_obs, n_samples,
    h=None, unbiased=False, rng=None, n_jobs=-1, parallel=True
):
    """Estimation MC de Cov(Z_theta) où Z_theta = J_theta^T Sigma_obs^{-1/2}.

    Returns
    -------
    Cov_theta : ndarray (d, d)
    """
    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    evals, evecs = la.eigh(Sigma_obs)
    if np.any(evals <= 0):
        raise ValueError("Sigma_obs must be symmetric positive definite.")
    Sigma_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    if rng is None:
        rng = np.random.default_rng()

    sampler = qmc.Halton(d=len(prior.mu))
    samples_unif = sampler.random(n=n_samples)
    u_normal = norm.ppf(samples_unif)
    L = np.linalg.cholesky(prior.Sigma)
    theta_samples = prior.mu + (L @ u_normal.T).T

    def compute_Z(theta):
        J = jacobian_fd(G, theta, h=h)
        return J.T @ Sigma_inv_sqrt  # shape (d, m)

    if parallel:
        Z_theta_list = Parallel(n_jobs=n_jobs)(
            delayed(compute_Z)(theta) for theta in theta_samples
        )
    else:
        Z_theta_list = [compute_Z(theta) for theta in theta_samples]

    Z_theta_arr = np.array(Z_theta_list)  # (n, d, m)
    Z_theta_mean = np.mean(Z_theta_arr, axis=0)
    d_theta = Z_theta_mean.shape[0]

    Cov_theta = np.zeros((d_theta, d_theta))
    for k in range(n_samples):
        Dt = Z_theta_arr[k] - Z_theta_mean
        Cov_theta += Dt @ Dt.T

    denom = (n_samples - 1) if (unbiased and n_samples > 1) else n_samples
    Cov_theta /= denom
    Cov_theta = 0.5 * (Cov_theta + Cov_theta.T)
    return Cov_theta


l_theta = estimate_E_JT(G=G, prior=prior, n_samples=n_samples)

Cov_theta = estimate_jacobian_covariances_mc(
    G=G, prior=prior,
    Sigma_obs=noise.get_covariance(N), n_samples=n_samples,
)

# =============================================================================
# 7. Matrices de covariance
# =============================================================================
def estimate_Sigma_Y(G, prior, Sigma_obs, n_samples, rng=None):
    """Sigma_Y = Sigma_obs + Cov_{pi_0}[G(theta)] par Monte Carlo."""
    if rng is None:
        rng = np.random.default_rng()

    Sigma_obs = np.asarray(Sigma_obs, dtype=float)
    sampler = qmc.Halton(d=len(prior.mu))
    samples_unif = sampler.random(n=n_samples)
    u_normal = norm.ppf(samples_unif)
    L = np.linalg.cholesky(prior.Sigma)
    theta_samples = prior.mu + (L @ u_normal.T).T

    U = Parallel(n_jobs=-1)(delayed(G)(theta_samples[k]) for k in range(n_samples))
    U = np.array(U)
    SY = Sigma_obs + np.cov(U, rowvar=False, bias=False)
    return 0.5 * (SY + SY.T)


def compute_Sigma_signal(l_theta, H_theta, Sigma_theta, Sigma_obs):
    """Sigma_signal = Sigma_obs + l_theta^T (Sigma_theta^{-1} + H_theta)^{-1} l_theta."""
    A = la.inv(Sigma_theta) + H_theta
    X = la.solve(A, l_theta)
    return Sigma_obs + l_theta.T @ X


Sigma_Y = estimate_Sigma_Y(
    G=G, prior=prior,
    Sigma_obs=noise.get_covariance(N), n_samples=15000,
)

Sigma_signal = compute_Sigma_signal(
    l_theta=l_theta,
    H_theta=Cov_theta,
    Sigma_theta=prior.Sigma,
    Sigma_obs=noise.get_covariance(N),
)

# =============================================================================
# 8. Approche gradient-free : régression linéaire
# =============================================================================
def solve_linear_matrix_regression_minimization(
    G, prior, Sigma_obs, n_samples=10000, random_state=0,
):
    """Résout min_{A,b} E[(X - AY - b)(X - AY - b)^T].

    Returns
    -------
    A_star, b_star, M_emp, M_formula, stats
    """
    rng = np.random.default_rng(random_state)

    Sigma_obs = np.asarray(Sigma_obs)
    theta_samples = rng.multivariate_normal(
        mean=prior.mu, cov=prior.Sigma, size=n_samples
    )

    X = Parallel(n_jobs=-1)(delayed(G)(th) for th in theta_samples)
    X = np.array(X)
    if X.ndim != 2:
        raise ValueError("u(theta) must return a 1D array with fixed length p.")

    n, p = X.shape
    eps = rng.multivariate_normal(np.zeros(p), Sigma_obs, size=n_samples)
    Y = X + eps

    m_X = X.mean(axis=0)
    m_Y = Y.mean(axis=0)
    Xc = X - m_X
    Yc = Y - m_Y

    Sigma_XY = (Xc.T @ Yc) / n
    Sigma_YY = (Yc.T @ Yc) / n
    Sigma_X  = (Xc.T @ Xc) / n

    A_star = Sigma_XY @ np.linalg.inv(Sigma_YY)
    b_star = m_X - A_star @ m_Y

    R     = X - (Y @ A_star.T + b_star)
    M_emp = (R.T @ R) / n

    M_formula = Sigma_X - Sigma_X @ np.linalg.inv(Sigma_X + Sigma_obs) @ Sigma_X

    stats = {
        "m_X": m_X, "m_Y": m_Y,
        "Sigma_X": Sigma_X, "Sigma_XY": Sigma_XY, "Sigma_YY": Sigma_YY,
        "X": X, "Y": Y, "residuals": R,
    }
    return A_star, b_star, M_emp, M_formula, stats


# A_star, b_star, M_emp, M_formula, stats = solve_linear_matrix_regression_minimization(
#     G=G, prior=prior,
#     Sigma_obs=noise.get_covariance(N),
#     n_samples=20000, random_state=0,
# )

# Sigma_obs_arr = noise.get_covariance(N)
# Sobs_inv = np.linalg.inv(Sigma_obs_arr)
# Iy = Sobs_inv @ (np.eye(N) - M_formula @ Sobs_inv)
# Sigma_signal_free = np.linalg.inv(Iy)



# =============================================================================
# 8b. Approche gradient-free : réseau de neurones
# =============================================================================
def generate_chunk(G, n_samples, prior, Sigma_obs, seed):
    torch.manual_seed(seed)
    mu  = torch.as_tensor(prior.mu,    dtype=torch.float32)
    cov = torch.as_tensor(prior.Sigma, dtype=torch.float32)
    Sigma_obs_t = torch.as_tensor(Sigma_obs, dtype=torch.float32)
    dim_y = Sigma_obs_t.shape[0]

    y_list, target_list = [], []
    for _ in range(n_samples):
        theta_t  = torch.distributions.MultivariateNormal(mu, cov).sample()
        G_val    = G(theta_t.numpy())
        eps      = torch.distributions.MultivariateNormal(torch.zeros(dim_y), Sigma_obs_t).sample()
        y        = torch.tensor(G_val, dtype=torch.float32) + eps
        target   = torch.tensor(G_val, dtype=torch.float32)
        y_list.append(y.numpy())
        target_list.append(target.numpy())
    return np.stack(y_list), np.stack(target_list)


def generate_data_parallel(G, n_samples, prior, Sigma_obs, n_jobs=None):
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    chunk_size = n_samples // n_jobs
    remainder  = n_samples % n_jobs
    sizes = [chunk_size] * n_jobs
    for i in range(remainder):
        sizes[i] += 1
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(generate_chunk)(G, sizes[i], prior, Sigma_obs, i) for i in range(n_jobs)
    )
    y_list, target_list = zip(*results)
    y      = np.concatenate(y_list, axis=0)
    target = np.concatenate(target_list, axis=0)
    return torch.tensor(y, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


n_train, n_test = 10000, 2000
Sigma_obs_t = torch.tensor(noise.get_covariance(N), dtype=torch.float32)

print("Génération des données d'entraînement...")
y_train, target_train = generate_data_parallel(G=G, n_samples=n_train, prior=prior, Sigma_obs=Sigma_obs_t)
print("Génération des données de test...")
y_test, target_test   = generate_data_parallel(G=G, n_samples=n_test,  prior=prior, Sigma_obs=Sigma_obs_t)

batch_size   = 64
train_loader = DataLoader(TensorDataset(y_train, target_train), batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(TensorDataset(y_test,  target_test),  batch_size=batch_size, num_workers=2)

nn_model  = SimpleNN(input_dim=N, hidden_dims=[64, 64], output_dim=N)
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=1e-3)
n_epochs  = 30
train_losses, test_losses = [], []

for epoch in range(n_epochs):
    nn_model.train()
    epoch_loss = 0.0
    for batch_y, batch_target in train_loader:
        optimizer.zero_grad()
        pred = nn_model(batch_y)
        loss = criterion(pred, batch_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_y.size(0)
    train_losses.append(epoch_loss / n_train)

    nn_model.eval()
    with torch.no_grad():
        test_loss = sum(
            criterion(nn_model(by), bt).item() * by.size(0)
            for by, bt in test_loader
        )
    test_losses.append(test_loss / n_test)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | Train MSE: {train_losses[-1]:.6f} | Test MSE: {test_losses[-1]:.6f}")

# Matrice de covariance conditionnelle via résidus
nn_model.eval()
residuals = []
with torch.no_grad():
    for by, bt in test_loader:
        residuals.append(bt - nn_model(by))
residuals = torch.cat(residuals, dim=0)
mean_res_cov = (residuals.T @ residuals) / len(residuals)

# Comparaison avec régression ridge
ridge = Ridge(alpha=0.1)
ridge.fit(y_train.numpy(), target_train.numpy())
ridge_pred = ridge.predict(y_test.numpy())
ridge_mse  = np.mean((target_test.numpy() - ridge_pred) ** 2)
print(f"\nMSE réseau:  {test_losses[-1]:.6f}")
print(f"MSE ridge:   {ridge_mse:.6f}")

# Sigma_signal via NN
mean_res_cov_np = mean_res_cov.numpy()
Sobs_inv = np.linalg.inv(noise.get_covariance(N))
Iy_nn = Sobs_inv @ (np.eye(N) - mean_res_cov_np @ Sobs_inv)
Sigma_signal_free_nn = np.linalg.inv(Iy_nn)

plot_matrices(
    [Sigma_signal_free_nn, Sigma_signal, Sigma_Y],
    [r"$I_Y^{-1} = \Sigma_{\rm signal free NN}$", r"$\Sigma_{\rm signal}$", r"$\Sigma_{\rm Y}$"],
    figsize=(12, 5),
)


# =============================================================================
# 9. Estimateurs EIG (NMC mémoire-efficace)
# =============================================================================
def logsumexp_rows(A):
    amax = A.max(axis=1, keepdims=True)
    return amax.squeeze() + np.log(np.exp(A - amax).sum(axis=1))


def estimate_eig_memory_efficient(G, prior, Sigma_obs, n_samples, batch_size=500, parallel=True):
    """Estimateur NMC de l'EIG, batch pour économiser la mémoire."""
    if n_samples < 2:
        raise ValueError("n_samples doit être >= 2")

    sampler = qmc.Halton(d=len(prior.mu))
    samples_unif = sampler.random(n=n_samples)
    u_normal = norm.ppf(samples_unif)
    L = np.linalg.cholesky(prior.Sigma)
    theta_samples = prior.mu + (L @ u_normal.T).T

    if parallel:
        U = np.array(Parallel(n_jobs=-1)(delayed(G)(theta) for theta in theta_samples))
    else:
        U = np.array([G(theta) for theta in theta_samples])

    m = U.shape[1]
    Sinv = la.inv(Sigma_obs)
    _, ldet = la.slogdet(Sigma_obs)
    log_const = -0.5 * (ldet + m * np.log(2 * np.pi))

    noise_mat = np.random.multivariate_normal(np.zeros(m), Sigma_obs, size=n_samples)
    Y   = U + noise_mat
    Y_M = Y @ Sinv
    U_M = U @ Sinv
    normY = np.einsum('ij,ij->i', Y, Y_M)
    normU = np.einsum('ij,ij->i', U, U_M)

    eig_estimate = 0.0
    for i in range(0, n_samples, batch_size):
        i_end       = min(i + batch_size, n_samples)
        bs_i        = i_end - i
        cross_batch = Y_M[i:i_end, :] @ U.T
        normY_batch = normY[i:i_end]
        quad_batch  = normY_batch[:, None] + normU[None, :] - 2 * cross_batch
        log_liks    = log_const - 0.5 * quad_batch
        log_num     = log_liks[np.arange(bs_i), i + np.arange(bs_i)]
        log_den     = logsumexp_rows(log_liks) - np.log(n_samples)
        eig_estimate += (log_num - log_den).sum()

    return eig_estimate / n_samples


eig_mem = estimate_eig_memory_efficient(
    G=G, prior=prior,
    Sigma_obs=noise.get_covariance(N),
    n_samples=20000, batch_size=500,
)


def eig_linearise(J, prior, Sigma_obs):
    S = Sigma_obs + J @ prior.Sigma @ J.T
    _, ldet_S   = np.linalg.slogdet(S)
    _, ldet_obs = np.linalg.slogdet(Sigma_obs)
    return 0.5 * (ldet_S - ldet_obs)


J_lin = jacobian_fd(G, prior.mu)
print(f"EIG linéarisé : {eig_linearise(J_lin, prior, noise.get_covariance(N)):.4f}")


# =============================================================================
# 10. Estimateurs EIG par sélection de capteurs
# =============================================================================
def eig_BS(Sigma_signal, Sigma_Y_given_theta, W):
    """Borne supérieure EIG (BS) relative au design plein."""
    def log_ratio(A, B, M):
        _, la_ = la.slogdet(M.T @ A @ M)
        _, lb_ = la.slogdet(M.T @ B @ M)
        return la_ - lb_

    eye = np.eye(Sigma_Y_given_theta.shape[0])
    return 0.5 * (log_ratio(Sigma_signal, Sigma_Y_given_theta, W)
                  - log_ratio(Sigma_signal, Sigma_Y_given_theta, eye))


def eig_BI(Sigma_Y, Sigma_noise, W):
    """Estimateur direct EIG (BI) relatif au design plein."""
    def log_ratio(A, B, M):
        _, la_ = la.slogdet(M.T @ A @ M)
        _, lb_ = la.slogdet(M.T @ B @ M)
        return la_ - lb_

    eye = np.eye(Sigma_noise.shape[0])
    return 0.5 * (log_ratio(Sigma_Y, Sigma_noise, W)
                  - log_ratio(Sigma_Y, Sigma_noise, eye))


# =============================================================================
# 11. Placement glouton de capteurs
# =============================================================================
def greedy_maximize_LB(Sigma_Y, Sigma_noise, n_sensors):
    """Sélection greedy forward maximisant le score log-ratio BI."""
    N_grid    = Sigma_Y.shape[0]
    selected  = []
    remaining = list(range(N_grid))

    for _ in range(n_sensors):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            S  = selected + [idx]
            ix = np.ix_(S, S)
            sign_y, logdet_y = np.linalg.slogdet(Sigma_Y[ix])
            sign_n, logdet_n = np.linalg.slogdet(Sigma_noise[ix])
            if sign_y <= 0 or sign_n <= 0:
                raise ValueError(f"Sous-matrice non SPD pour S={S}")
            score = 0.5 * (logdet_y - logdet_n)
            if score > best_score:
                best_score = score
                best_idx   = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return np.array(selected, dtype=int)


def schur(Sigma, Wm):
    if Wm.size == 0:
        return Sigma
    G_mat = Wm.T @ Sigma @ Wm
    return Sigma - Sigma @ Wm @ np.linalg.inv(G_mat) @ Wm.T @ Sigma


def incremental_bounds(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, n_sensors):
    """Placement glouton avec bornes incrémentales inférieures et supérieures."""
    N_loc = Sigma_Y.shape[0]
    W_candidates = [np.eye(N_loc)[:, i] for i in range(N_loc)]
    selected  = []
    remaining = list(range(N_loc))
    inc_inf   = []
    inc_sup   = []
    eig_inf = eig_sup = 0.0

    for m in range(n_sensors):
        Wm = np.column_stack([W_candidates[i] for i in selected]) if selected else np.empty((N_loc, 0))

        Sigma_s_m   = schur(Sigma_signal,  Wm)
        Sigma_yth_m = schur(Sigma_Y_theta, Wm)
        Sigma_Y_m   = schur(Sigma_Y,       Wm)
        Sigma_n_m   = schur(Sigma_noise,   Wm)

        best_idx = best_inc = best_sup = None
        best_inc = -np.inf

        for idx in remaining:
            w = W_candidates[idx]
            num_inf = w.T @ Sigma_s_m   @ w
            den_inf = w.T @ Sigma_yth_m @ w
            num_sup = w.T @ Sigma_Y_m   @ w
            den_sup = w.T @ Sigma_n_m   @ w

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
    """Calcule les bornes (lb, ub) pour un W donné."""
    _, logdet_sig = np.linalg.slogdet(W.T @ Sigma_signal  @ W)
    _, logdet_th  = np.linalg.slogdet(W.T @ Sigma_Y_theta @ W)
    lb = 0.5 * (logdet_sig - logdet_th)

    _, logdet_Y = np.linalg.slogdet(W.T @ Sigma_Y     @ W)
    _, logdet_n = np.linalg.slogdet(W.T @ Sigma_noise @ W)
    ub = 0.5 * (logdet_Y - logdet_n)
    return lb, ub


# =============================================================================
# 12. Comparaison des bornes (greedy BI vs bornes incrémentales)
# =============================================================================
def plot_eig_bounds(sensor_budgets, lb_cons, ub_cons, lb_inc, ub_inc, offset, title=""):
    lb_common = np.maximum(lb_cons, lb_inc)
    ub_common = np.minimum(ub_cons, ub_inc)
    mask = lb_common <= ub_common

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sensor_budgets, lb_cons, '--o', linewidth=2, label='Lower bound (conservative)')
    ax.plot(sensor_budgets, ub_cons, '--s', linewidth=2, label='Upper bound (conservative)')
    ax.fill_between(sensor_budgets, lb_cons, ub_cons, alpha=0.15)

    ax.plot(sensor_budgets, lb_inc, '-o', linewidth=2, label='Lower bound (incremental)')
    ax.plot(sensor_budgets, ub_inc, '-s', linewidth=2, label='Upper bound (incremental)')
    ax.fill_between(sensor_budgets, lb_inc, ub_inc, alpha=0.15)

    ax.fill_between(sensor_budgets, lb_common, ub_common,
                    where=mask, interpolate=True, alpha=0.4, hatch='//',
                    label='Common certified region')

    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Information gain")
    ax.set_title(title or "EIG bounds, gaps, and common certified region")
    ax.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


sensor_budgets = [5, 10, 15, 20, 25]
Sigma_obs_arr  = noise.get_covariance(N)

# --- Greedy BI + bornes incrémentales (Sigma_signal)
lb_vals, ub_vals, lb_vals_inc, ub_vals_inc = [], [], [], []
for budget in sensor_budgets:
    result = incremental_bounds(
        Sigma_signal=Sigma_signal, Sigma_Y_theta=Sigma_obs_arr,
        Sigma_Y=Sigma_Y, Sigma_noise=Sigma_obs_arr, n_sensors=budget,
    )
    W_opt, _ = build_selection_matrices(N, result['indices'])
    lb_vals.append(eig_BI(Sigma_Y, Sigma_obs_arr, W_opt))
    ub_vals.append(eig_BS(Sigma_signal, Sigma_obs_arr, W_opt))
    lb_vals_inc.append(result["EIG_lower_bound"])
    ub_vals_inc.append(result["EIG_upper_bound"])

plot_eig_bounds(
    sensor_budgets,
    np.array(lb_vals) + eig_mem, np.array(ub_vals) + eig_mem,
    lb_vals_inc, ub_vals_inc,
    offset=eig_mem,
    title="EIG bounds — greedy incremental (Sigma_signal)",
)

# --- Greedy BI classique + bornes incrémentales
lb_vals, ub_vals, lb_vals_inc, ub_vals_inc = [], [], [], []
for budget in sensor_budgets:
    indices = greedy_maximize_LB(Sigma_Y=Sigma_Y, Sigma_noise=Sigma_obs_arr, n_sensors=budget)
    W_opt, _ = build_selection_matrices(N, indices)
    lb = eig_BI(Sigma_Y, Sigma_obs_arr, W_opt)
    ub = eig_BS(Sigma_signal, Sigma_obs_arr, W_opt)
    lb_inc, ub_inc = incremental_bounds_given_W(
        Sigma_signal=Sigma_signal, Sigma_Y_theta=Sigma_obs_arr,
        Sigma_Y=Sigma_Y, Sigma_noise=Sigma_obs_arr, W=W_opt,
    )
    lb_vals.append(lb)
    ub_vals.append(ub)
    lb_vals_inc.append(lb_inc)
    ub_vals_inc.append(ub_inc)

plot_eig_bounds(
    sensor_budgets,
    np.array(lb_vals) + eig_mem, np.array(ub_vals) + eig_mem,
    lb_vals_inc, ub_vals_inc,
    offset=eig_mem,
    title="EIG bounds — greedy BI (sanity check)",
)

# --- NN-based Sigma_signal_free
lb_vals, ub_vals, lb_vals_inc, ub_vals_inc = [], [], [], []
for budget in sensor_budgets:
    result = incremental_bounds(
        Sigma_signal=Sigma_signal_free_nn, Sigma_Y_theta=Sigma_obs_arr,
        Sigma_Y=Sigma_Y, Sigma_noise=Sigma_obs_arr, n_sensors=budget,
    )
    W_opt, _ = build_selection_matrices(N, result['indices'])
    lb_vals.append(eig_BI(Sigma_Y, Sigma_obs_arr, W_opt))
    ub_vals.append(eig_BS(Sigma_signal, Sigma_obs_arr, W_opt))
    lb_vals_inc.append(result["EIG_lower_bound"])
    ub_vals_inc.append(result["EIG_upper_bound"])

plot_eig_bounds(
    sensor_budgets,
    np.array(lb_vals) + eig_mem, np.array(ub_vals) + eig_mem,
    lb_vals_inc, ub_vals_inc,
    offset=eig_mem,
    title="EIG bounds — NN-based Sigma_signal_free",
)

print(f"Relative error (nuc) Sigma_signal_free_nn vs Sigma_signal: "
      f"{np.linalg.norm(Sigma_signal_free_nn - Sigma_signal, ord='nuc') * 100 / np.linalg.norm(Sigma_signal, ord='nuc'):.2f} %")
print(f"Relative error (fro) Sigma_signal_free_nn vs Sigma_signal: "
      f"{np.linalg.norm(Sigma_signal_free_nn - Sigma_signal, ord='fro') * 100 / np.linalg.norm(Sigma_signal, ord='fro'):.2f} %")
