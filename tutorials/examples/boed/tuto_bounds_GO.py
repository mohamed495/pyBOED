"""
Bayesian Optimal Experimental Design — Burgers Equation (Generalized Observability)
=====================================================================================
Pipeline avec paramètre de nuisance eta :
  1. Setup        — PDE model, noise, discretisation
  2. Prior joint  — GP prior sur (theta, eta)
  3. Forward      — G : (theta, eta) -> u(·, T)
  4. Jacobiens FD — J_theta, J_eta et espérances MC
  5. Covariances  — Sigma_Y, Sigma_noise, Sigma_signal, Sigma_{Y|theta}
  6. Gradient-free — estimation via régression linéaire (cas GO)
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
from joblib import Parallel, delayed

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

N        = 100      # nombre de points intérieurs
dt       = 0.001
n_steps  = 100
sigma    = 0.01     # écart-type du bruit
lambda_  = 0.5
n_samples = 500

x_grid = np.linspace(0, 1, N + 2)[1:-1]

model   = Burgers_CN(N=N, dt=dt, diffusivity=0.02, lambda_=lambda_)
u0_true = make_u0(x_grid, "gaussian", center=0.25, width=0.07, amplitude=1.5)
noise   = NoiseModel(sigma_noise=sigma)


# =============================================================================
# 3. Prior joint (theta, eta)
# =============================================================================
kernel = Matern32(length_scale=0.2, sigma=1.0)
prior  = GaussianProcessPrior(kernel, mu=np.zeros(N), nx=N)

# Régularisation numérique
eigvals = np.linalg.eigvalsh(prior.Sigma)
jitter  = eigvals[-1] / 1000
prior.Sigma += jitter

d = q = N // 2  # theta : première moitié, eta : seconde

theta_prior = {"mu": prior.mu[:d],  "Sigma": prior.Sigma[:d, :d]}
eta_prior   = {"mu": prior.mu[d:],  "Sigma": prior.Sigma[d:, d:]}

Sigma_theta_eta = prior.Sigma[:d, d:]
Sigma_eta_theta = prior.Sigma[d:, :d]

joint_prior = {
    "mu":    prior.mu,
    "Sigma": prior.Sigma,
    "d":     d,
    "q":     q,
}


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


def sample_joint_prior(joint_prior, n_samples, rng=None):
    """Tire n_samples depuis le prior joint z = (theta, eta) ~ N(m, Sigma)."""
    if rng is None:
        rng = np.random.default_rng()
    d = joint_prior["d"]
    z = rng.multivariate_normal(
        mean=joint_prior["mu"],
        cov=joint_prior["Sigma"],
        size=n_samples,
    )
    return z[:, :d], z[:, d:]


plot_matrices(
    [noise.get_covariance(N), theta_prior["Sigma"], eta_prior["Sigma"]],
    [r"$\Sigma_{\rm obs}$", r"$\Sigma_\theta$", r"$\Sigma_\eta$"],
    figsize=(14, 4),
)


# =============================================================================
# 4. Modèle direct
# =============================================================================
def make_forward_model(pde_model, n_steps):
    """Retourne G(theta, eta) -> u(·, T)."""
    def G(theta: np.ndarray, eta: np.ndarray | None):
        u0 = np.concatenate((theta, eta), axis=None) if eta is not None else theta
        return pde_model.evolve(u0=u0, n_steps=n_steps)[-1, :]
    return G


theta, eta = np.split(u0_true, 2)
print(f"theta size : {theta.size}")
print(f"eta   size : {eta.size}")

G = make_forward_model(pde_model=model, n_steps=n_steps)

u1 = G(theta=theta, eta=eta)
u2 = G(theta=u0_true.copy(), eta=None)
print(f"Consistency check ||G(theta,eta) - G(u0)|| = {np.linalg.norm(u1 - u2):.2e}")


# =============================================================================
# 5. Jacobiens par différences finies
# =============================================================================
def jacobian_fd_theta(G, theta, eta, h=None):
    """Jacobien central FD par rapport à theta. Shape (m, d)."""
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
    """Jacobien central FD par rapport à eta. Shape (m, q)."""
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


def jacobian_fd_full(G, theta, eta, h_theta=None, h_eta=None):
    """Jacobien complet [J_theta | J_eta], shape (m, d+q)."""
    J_theta = jacobian_fd_theta(G, theta, eta, h=h_theta)
    J_eta   = jacobian_fd_eta(G, theta, eta, h=h_eta)
    return np.concatenate((J_theta, J_eta), axis=1)


# Vérification convergence FD
hs = [1e-2, 1e-4, 1e-6, 1e-8]
J_theta_list, J_eta_list, J_full_list = [], [], []
for h in hs:
    J_theta = jacobian_fd_theta(G, theta=theta, eta=eta, h=h)
    J_eta   = jacobian_fd_eta(G,   theta=theta, eta=eta, h=h)
    J_full  = jacobian_fd_full(G,  theta=theta, eta=eta, h_theta=h, h_eta=h)
    J_theta_list.append(J_theta)
    J_eta_list.append(J_eta)
    J_full_list.append(J_full)
    print(f"h = {h}  |  J_theta {J_theta.shape}  J_eta {J_eta.shape}  J_full {J_full.shape}")

print("\nDifférences relatives (theta):")
for i in range(len(hs) - 1):
    rel = np.linalg.norm(J_theta_list[i] - J_theta_list[i+1]) / np.linalg.norm(J_theta_list[i+1])
    print(f"  h={hs[i]} -> {hs[i+1]} : {rel:.2e}")


# =============================================================================
# 6. Estimation MC de l_theta, l_eta, l_{(theta,eta)}
# =============================================================================
def _one_sample(G, theta, eta):
    J_theta_T = jacobian_fd_theta(G, theta, eta).T
    J_eta_T   = jacobian_fd_eta(G, theta, eta).T
    return J_theta_T, J_eta_T


def estimate_E_JT(G, joint_prior, n_samples, rng=None):
    """Estimation MC de E[J_theta^T], E[J_eta^T] et leur concaténation.

    Returns
    -------
    dict avec EJ_theta_T (d, m), EJ_eta_T (q, m), EJ_full_T (d+q, m)
    """
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)

    results = Parallel(n_jobs=-1)(
        delayed(_one_sample)(G, theta_samples[k], eta_samples[k])
        for k in range(n_samples)
    )

    J_theta_T_0, J_eta_T_0 = results[0]
    EJ_theta_T_sum = np.zeros_like(J_theta_T_0, dtype=float)
    EJ_eta_T_sum   = np.zeros_like(J_eta_T_0, dtype=float)

    for Jt, Je in results:
        EJ_theta_T_sum += Jt
        EJ_eta_T_sum   += Je

    EJ_theta_T = EJ_theta_T_sum / n_samples
    EJ_eta_T   = EJ_eta_T_sum   / n_samples
    EJ_full_T  = np.concatenate((EJ_theta_T, EJ_eta_T), axis=0)

    return {"EJ_theta_T": EJ_theta_T, "EJ_eta_T": EJ_eta_T, "EJ_full_T": EJ_full_T}


res_l = estimate_E_JT(G=G, joint_prior=joint_prior, n_samples=n_samples)
print(f"l_theta shape : {res_l['EJ_theta_T'].shape}")
print(f"l_eta   shape : {res_l['EJ_eta_T'].shape}")
print(f"l_full  shape : {res_l['EJ_full_T'].shape}")
plot_matrices(
    [res_l["EJ_theta_T"], res_l["EJ_eta_T"], res_l["EJ_full_T"]],
    [r"$l_\theta$", r"$l_\eta$", r"$l_{(\theta,\eta)}$"],
    figsize=(14, 4),
)


# =============================================================================
# 7. Estimation MC des covariances de Jacobiens
# =============================================================================
def _compute_z(theta, eta, G, h_theta, h_eta, Sigma_inv_sqrt):
    """Calcule Z_theta et Z_eta pour un échantillon (theta, eta)."""
    J_theta = jacobian_fd_theta(G, theta, eta, h=h_theta)
    J_eta   = jacobian_fd_eta(G, theta, eta, h=h_eta)
    return J_theta.T @ Sigma_inv_sqrt, J_eta.T @ Sigma_inv_sqrt


def estimate_jacobian_covariances_mc(
    G, joint_prior, Sigma_obs, n_samples,
    h_theta=None, h_eta=None, unbiased=False, rng=None, n_jobs=-1,
):
    """Estimation MC de Cov(Z_theta), Cov(Z_eta) et bloc complet.

    Returns
    -------
    dict avec Cov_theta, Cov_eta, Cov_theta_eta, Cov_full
    """
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
    Z_theta_arr = np.asarray(Z_theta_list, dtype=float)  # (n, d, m)
    Z_eta_arr   = np.asarray(Z_eta_list,   dtype=float)  # (n, q, m)

    Z_theta_mean = np.mean(Z_theta_arr, axis=0)
    Z_eta_mean   = np.mean(Z_eta_arr,   axis=0)

    d_theta = Z_theta_mean.shape[0]
    d_eta   = Z_eta_mean.shape[0]

    Cov_theta     = np.zeros((d_theta, d_theta))
    Cov_eta       = np.zeros((d_eta,   d_eta))
    Cov_theta_eta = np.zeros((d_theta, d_eta))

    for k in range(n_samples):
        Dt = Z_theta_arr[k] - Z_theta_mean
        De = Z_eta_arr[k]   - Z_eta_mean
        Cov_theta     += Dt @ Dt.T
        Cov_eta       += De @ De.T
        Cov_theta_eta += Dt @ De.T

    denom = (n_samples - 1) if (unbiased and n_samples > 1) else n_samples
    Cov_theta     /= denom
    Cov_eta       /= denom
    Cov_theta_eta /= denom

    Cov_theta = 0.5 * (Cov_theta + Cov_theta.T)
    Cov_eta   = 0.5 * (Cov_eta   + Cov_eta.T)

    Cov_full = np.block([
        [Cov_theta,       Cov_theta_eta],
        [Cov_theta_eta.T, Cov_eta],
    ])

    return {"Cov_theta": Cov_theta, "Cov_eta": Cov_eta,
            "Cov_theta_eta": Cov_theta_eta, "Cov_full": Cov_full}


res_H = estimate_jacobian_covariances_mc(
    G=G, joint_prior=joint_prior,
    Sigma_obs=noise.get_covariance(N), n_samples=n_samples,
)
print(f"H_theta shape     : {res_H['Cov_theta'].shape}")
print(f"H_eta   shape     : {res_H['Cov_eta'].shape}")
print(f"H_theta_eta shape : {res_H['Cov_theta_eta'].shape}")
plot_matrices(
    [res_H["Cov_theta"], res_H["Cov_eta"], res_H["Cov_full"]],
    [r"$\mathcal{H}_\theta$", r"$\mathcal{H}_\eta$", r"$\mathcal{H}_{(\theta,\eta)}$"],
    figsize=(12, 4),
)


# =============================================================================
# 8. Matrices de covariance
# =============================================================================
def estimate_Sigma_Y(G, joint_prior, Sigma_obs, n_samples, rng=None):
    """Sigma_Y = Sigma_obs + Cov_{pi_0}[G(theta, eta)] par Monte Carlo."""
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)
    U = Parallel(n_jobs=-1)(
        delayed(G)(theta_samples[k], eta_samples[k]) for k in range(n_samples)
    )
    U = np.array(U)
    SY = Sigma_obs + np.cov(U, rowvar=False, bias=False)
    return 0.5 * (SY + SY.T)


def compute_Sigma_noise(L_eta, H_eta, Sigma_eta_given_theta, Sigma_obs):
    """Sigma_noise = Sigma_obs + l_eta^T (Sigma_{eta|theta}^{-1} + H_eta)^{-1} l_eta."""
    q  = Sigma_eta_given_theta.shape[0]
    HS = H_eta @ Sigma_eta_given_theta
    M  = Sigma_eta_given_theta - Sigma_eta_given_theta @ la.solve(np.eye(q) + HS, HS)
    return Sigma_obs + L_eta.T @ M @ L_eta


def compute_Sigma_signal(l_theta, H_theta, Sigma_theta, Sigma_obs):
    """Sigma_signal = Sigma_obs + l_theta^T (Sigma_theta^{-1} + H_theta)^{-1} l_theta."""
    N_loc = Sigma_theta.shape[0]
    HS    = H_theta @ Sigma_theta
    M     = Sigma_theta - Sigma_theta @ la.solve(np.eye(N_loc) + HS, HS)
    return Sigma_obs + l_theta.T @ M @ l_theta


# Covariance conditionnelle eta | theta (complément de Schur)
Sigma_eta_given_theta = (
    eta_prior["Sigma"]
    - Sigma_eta_theta @ la.solve(theta_prior["Sigma"], Sigma_theta_eta)
)
print(f"Condition number Sigma_eta|theta : {np.linalg.cond(Sigma_eta_given_theta):.2e}")

Sigma_Y = estimate_Sigma_Y(
    G=G, joint_prior=joint_prior,
    Sigma_obs=noise.get_covariance(N), n_samples=n_samples,
)
plot_matrices([Sigma_Y], [r"$\Sigma_Y$"], figsize=(5, 5))

Sigma_noise = compute_Sigma_noise(
    L_eta=res_l["EJ_eta_T"],
    H_eta=res_H["Cov_eta"],
    Sigma_eta_given_theta=Sigma_eta_given_theta,
    Sigma_obs=noise.get_covariance(N),
)
plot_matrices([Sigma_noise], [r"$\Sigma_{\rm noise}$"], figsize=(5, 5))

Sigma_signal = compute_Sigma_signal(
    l_theta=res_l["EJ_full_T"],
    H_theta=res_H["Cov_full"],
    Sigma_theta=prior.Sigma,
    Sigma_obs=noise.get_covariance(N),
)
plot_matrices([Sigma_signal], [r"$\Sigma_{\rm signal}$"], figsize=(5, 5))


# =============================================================================
# 8d. Covariance conditionnelle Sigma_{Y|theta}
# =============================================================================
def _process_theta(i, theta_i, mu_theta, Sigma_theta, mu_eta, Sigma_et,
                   Sigma_eta_given_theta, G, n_eta, base_seed):
    rng = np.random.default_rng(base_seed + i)
    mu_eta_given_theta_i = mu_eta + Sigma_et @ la.solve(Sigma_theta, theta_i - mu_theta)
    eta_cond = rng.multivariate_normal(
        mean=mu_eta_given_theta_i, cov=Sigma_eta_given_theta, size=n_eta,
    )
    vals = np.array([G(theta_i, eta_cond[j]) for j in range(n_eta)])
    if vals.ndim == 1:
        vals = vals[:, None]
    return np.cov(vals, rowvar=False, ddof=1)


def estimate_E_cov_Y_given_theta(
    G, joint_prior, Sigma_obs, n_theta, n_eta, rng=None, n_jobs=-1
):
    """Estimation de E_theta[Cov(Y | theta)] = Sigma_obs + E_theta[Cov(G(theta,eta)|theta)]."""
    if rng is None:
        rng = np.random.default_rng()
    if n_eta < 2:
        raise ValueError("n_eta must be >= 2")

    d = joint_prior["d"]
    mu    = joint_prior["mu"]
    Sigma = joint_prior["Sigma"]

    mu_theta    = mu[:d]
    mu_eta      = mu[d:]
    Sigma_theta = Sigma[:d, :d]
    Sigma_eta   = Sigma[d:, d:]
    Sigma_et    = Sigma[d:, :d]

    Sigma_theta_inv_Sigma_te = la.solve(Sigma_theta, Sigma_et.T)
    Sigma_eta_given_theta_loc = Sigma_eta - Sigma_et @ Sigma_theta_inv_Sigma_te
    Sigma_eta_given_theta_loc = 0.5 * (Sigma_eta_given_theta_loc + Sigma_eta_given_theta_loc.T)
    Sigma_eta_given_theta_loc += 1e-12 * np.eye(Sigma_eta.shape[0])

    theta_samples = rng.multivariate_normal(mean=mu_theta, cov=Sigma_theta, size=n_theta)
    base_seed = rng.integers(0, 2**30)

    inner_cov_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_theta)(
            i, theta_samples[i], mu_theta, Sigma_theta,
            mu_eta, Sigma_et, Sigma_eta_given_theta_loc, G, n_eta, base_seed,
        )
        for i in range(n_theta)
    )

    result = Sigma_obs + np.mean(np.asarray(inner_cov_list), axis=0)
    return 0.5 * (result + result.T)


Sigma_Y_given_theta = estimate_E_cov_Y_given_theta(
    G=G, joint_prior=joint_prior,
    Sigma_obs=noise.get_covariance(N), n_theta=n_samples, n_eta=200,
)
plot_matrices([Sigma_Y_given_theta], [r"$\Sigma_{Y|\theta}$"], figsize=(5, 5))

plot_matrices(
    [Sigma_Y_given_theta, Sigma_noise],
    [r"$\Sigma_{Y|\theta}$", r"$\Sigma_{\rm noise}$"],
    figsize=(10, 5),
)
print(f"Relative error (nuc) Sigma_Y|theta vs Sigma_noise: "
      f"{np.linalg.norm(Sigma_Y_given_theta - Sigma_noise, ord='nuc') / np.linalg.norm(Sigma_Y_given_theta, ord='nuc'):.4f}")

plot_matrices(
    [Sigma_Y, Sigma_signal],
    [r"$\Sigma_{Y}$", r"$\Sigma_{\rm signal}$"],
    figsize=(10, 5),
)
print(f"Relative error (nuc) Sigma_Y vs Sigma_signal: "
      f"{np.linalg.norm(Sigma_Y - Sigma_signal, ord='nuc') / np.linalg.norm(Sigma_Y, ord='nuc'):.4f}")


# =============================================================================
# 9. Approche gradient-free
# =============================================================================
def solve_linear_regression_theta_Y(
    G, joint_prior, Sigma_obs, n_samples=10000, random_state=0, n_jobs=-1,
):
    """Régression linéaire de X sur Z = (theta, Y) pour estimer E[Cov(u|theta, Y)].

    Returns A, b, M_emp, stats
    """
    rng = np.random.default_rng(random_state)
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)

    X_list = Parallel(n_jobs=n_jobs)(
        delayed(G)(theta_samples[k], eta_samples[k]) for k in range(n_samples)
    )
    X = np.asarray(X_list, dtype=float)
    n, p = X.shape

    Sigma_obs = np.asarray(Sigma_obs)
    eps = rng.multivariate_normal(np.zeros(p), Sigma_obs, size=n_samples)
    Y   = X + eps

    theta_arr = np.asarray(theta_samples)
    Z = np.hstack([theta_arr, Y])

    m_X = X.mean(axis=0)
    m_Z = Z.mean(axis=0)
    Xc  = X - m_X
    Zc  = Z - m_Z

    Sigma_XZ = (Xc.T @ Zc) / n
    Sigma_ZZ = (Zc.T @ Zc) / n
    Sigma_X  = (Xc.T @ Xc) / n

    A = Sigma_XZ @ np.linalg.pinv(Sigma_ZZ)
    b = m_X - A @ m_Z
    R     = X - (Z @ A.T + b)
    M_emp = (R.T @ R) / n

    stats = {
        "m_X": m_X, "m_Z": m_Z,
        "Sigma_XZ": Sigma_XZ, "Sigma_ZZ": Sigma_ZZ, "Sigma_X": Sigma_X,
        "X": X, "Y": Y, "Z": Z, "residuals": R,
        "trace_empirical": np.trace(M_emp),
    }
    return A, b, M_emp, stats


def solve_linear_matrix_regression_minimization(
    G, joint_prior, Sigma_obs, n_samples=10000, random_state=0,
):
    """Résout min_{A,b} E[(X - AY - b)(X - AY - b)^T] pour estimer E[Cov(u|Y)].

    Returns A_star, b_star, M_emp, M_formula, stats
    """
    Sigma_obs = np.asarray(Sigma_obs)
    rng = np.random.default_rng()
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)

    X_list = Parallel(n_jobs=-1)(
        delayed(G)(theta_samples[k], eta_samples[k]) for k in range(n_samples)
    )
    X = np.asarray(X_list, dtype=float)
    n, p = X.shape

    eps = rng.multivariate_normal(np.zeros(p), Sigma_obs, size=n_samples)
    Y   = X + eps

    m_X = X.mean(axis=0)
    m_Y = Y.mean(axis=0)
    Xc  = X - m_X
    Yc  = Y - m_Y

    Sigma_XY = (Xc.T @ Yc) / n
    Sigma_YY = (Yc.T @ Yc) / n
    Sigma_X  = (Xc.T @ Xc) / n

    A_star = Sigma_XY @ np.linalg.inv(Sigma_YY)
    b_star = m_X - A_star @ m_Y
    R      = X - (Y @ A_star.T + b_star)
    M_emp  = (R.T @ R) / n

    M_formula = Sigma_X - Sigma_X @ np.linalg.inv(Sigma_X + Sigma_obs) @ Sigma_X

    stats = {
        "m_X": m_X, "m_Y": m_Y,
        "Sigma_X": Sigma_X, "Sigma_XY": Sigma_XY, "Sigma_YY": Sigma_YY,
        "X": X, "Y": Y, "residuals": R,
    }
    return A_star, b_star, M_emp, M_formula, stats


# Estimation gradient-free de Sigma_noise
A_star, b_star, M_emp_noise, stats_noise = solve_linear_regression_theta_Y(
    G=G, joint_prior=joint_prior,
    Sigma_obs=noise.get_covariance(N),
    n_samples=20000, random_state=0,
)
Sigma_obs_arr = noise.get_covariance(N)
Sobs_inv = np.linalg.inv(Sigma_obs_arr)
EIytheta = Sobs_inv @ (np.eye(N) - M_emp_noise @ Sobs_inv)
Sigma_noise_free = np.linalg.inv(EIytheta)

plot_matrices(
    [Sigma_noise_free],
    [r"$\mathbb{E}[I_{Y|\theta}]^{-1} = \Sigma_{\rm noise free}$"],
    figsize=(10, 5),
)

# Estimation gradient-free de Sigma_signal
A_star, b_star, M_emp_sig, M_formula, stats_sig = solve_linear_matrix_regression_minimization(
    G=G, joint_prior=joint_prior,
    Sigma_obs=noise.get_covariance(N),
    n_samples=20000, random_state=0,
)
print(f"Relative error M_emp vs M_formula (nuc): "
      f"{np.linalg.norm(M_emp_sig - M_formula, ord='nuc') * 100 / np.linalg.norm(M_formula, ord='nuc'):.2f} %")

Iy = Sobs_inv @ (np.eye(N) - M_formula @ Sobs_inv)
Sigma_signal_free = np.linalg.inv(Iy)

plot_matrices(
    [Sigma_signal_free, Sigma_signal],
    [r"$I_Y^{-1} = \Sigma_{\rm signal free}$", r"$\Sigma_{\rm signal}$"],
    figsize=(10, 5),
)
print(f"Relative error (nuc) Sigma_signal vs Sigma_Y: "
      f"{np.linalg.norm(Sigma_signal - Sigma_Y, ord='nuc') * 100 / np.linalg.norm(Sigma_Y, ord='nuc'):.2f} %")


# =============================================================================
# 10. Estimateurs EIG
# =============================================================================
def logsumexp_rows(A):
    amax = A.max(axis=1, keepdims=True)
    return amax.squeeze() + np.log(np.exp(A - amax).sum(axis=1))


def estimate_eig_memory_efficient(G, joint_prior, Sigma_obs, n_samples, batch_size=500):
    """Estimateur NMC de l'EIG, batch pour économiser la mémoire."""
    if n_samples < 2:
        raise ValueError("n_samples doit être >= 2")

    rng = np.random.default_rng()
    theta_samples, eta_samples = sample_joint_prior(joint_prior, n_samples, rng=rng)

    U = Parallel(n_jobs=-1)(
        delayed(G)(theta_samples[k], eta_samples[k]) for k in range(n_samples)
    )
    U = np.array(U)
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
    G=G, joint_prior=joint_prior,
    Sigma_obs=noise.get_covariance(N),
    n_samples=150000, batch_size=500,
)
print(f"EIG (NMC) = {eig_mem:.4f}")


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


# =============================================================================
# 12. Bornes incrémentales
# =============================================================================
def incremental_bounds(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, n_sensors):
    """Placement glouton par bornes incrémentales (inférieures et supérieures)."""
    N_loc = Sigma_Y.shape[0]
    W_candidates = [np.eye(N_loc)[i] for i in range(N_loc)]
    selected  = []
    remaining = list(range(N_loc))
    scores_inf = []
    scores_sup = []

    for step in range(n_sensors):
        best_idx   = None
        best_score = -np.inf

        for idx in remaining:
            S = selected + [idx]
            W = np.column_stack([W_candidates[i] for i in S])

            def log_ratio(A, B):
                sign_a, ld_a = np.linalg.slogdet(W.T @ A @ W)
                sign_b, ld_b = np.linalg.slogdet(W.T @ B @ W)
                if sign_a <= 0 or sign_b <= 0:
                    return -np.inf
                return 0.5 * (ld_a - ld_b)

            score = log_ratio(Sigma_signal, Sigma_Y_theta)
            if score > best_score:
                best_score = score
                best_idx   = idx

        if best_idx is None:
            print(f"Arrêt anticipé à l'étape {step}.")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        W = np.column_stack([W_candidates[i] for i in selected])
        scores_inf.append(0.5 * (np.linalg.slogdet(W.T @ Sigma_signal  @ W)[1]
                                 - np.linalg.slogdet(W.T @ Sigma_Y_theta @ W)[1]))
        scores_sup.append(0.5 * (np.linalg.slogdet(W.T @ Sigma_Y       @ W)[1]
                                 - np.linalg.slogdet(W.T @ Sigma_noise   @ W)[1]))

    return {
        "indices":          np.array(selected, dtype=int),
        "scores_inf":       scores_inf,
        "scores_sup":       scores_sup,
        "EIG_lower_bound":  scores_inf[-1] if scores_inf else 0.0,
        "EIG_upper_bound":  scores_sup[-1] if scores_sup else None,
    }


def incremental_bounds_given_W(Sigma_signal, Sigma_Y_theta, Sigma_Y, Sigma_noise, W):
    """Calcule (lb, ub) pour un W donné."""
    _, logdet_sig = np.linalg.slogdet(W.T @ Sigma_signal  @ W)
    _, logdet_th  = np.linalg.slogdet(W.T @ Sigma_Y_theta @ W)
    lb = 0.5 * (logdet_sig - logdet_th)

    _, logdet_Y = np.linalg.slogdet(W.T @ Sigma_Y    @ W)
    _, logdet_n = np.linalg.slogdet(W.T @ Sigma_noise @ W)
    ub = 0.5 * (logdet_Y - logdet_n)
    return lb, ub


# =============================================================================
# 13. Comparaison des bornes
# =============================================================================
def plot_eig_bounds(sensor_budgets, lb_cons, ub_cons, lb_inc, ub_inc, title=""):
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

# --- Greedy BI classique
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
    title="EIG bounds — greedy BI (GO setting)",
)

# --- Greedy par bornes incrémentales (Sigma_signal)
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
    title="EIG bounds — greedy incremental (GO setting)",
)
