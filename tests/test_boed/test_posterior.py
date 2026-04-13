import numpy as np

from boed.inference import LinearGaussianModel
from boed.utils.observation import compute_W


class _StaticLinearModel:
    def __init__(self, A: np.ndarray):
        self._A = np.asarray(A, dtype=float)

    def get_forward_operator(self, n_steps: int):
        _ = n_steps
        return self._A


def test_linear_gaussian_posterior_identity():
    n = 5
    A = np.eye(n)
    sigma = 0.1
    Sigma_noise = (sigma**2) * np.eye(n)
    mu_prior = np.zeros(n)
    Sigma_prior = np.eye(n)

    lgm = LinearGaussianModel(A, Sigma_obs=Sigma_noise, mu_prior=mu_prior, Sigma_prior=Sigma_prior)
    y = np.arange(1.0, n + 1.0)
    mu_post, Sigma_post = lgm.posterior(y)

    expected_cov = (1.0 / (1.0 + 1.0 / (sigma**2))) * np.eye(n)
    expected_mean = (1.0 / (1.0 + sigma**2)) * y

    assert np.allclose(Sigma_post, expected_cov)
    assert np.allclose(mu_post, expected_mean)
    assert np.allclose(Sigma_post, Sigma_post.T)


def test_linear_gaussian_accepts_model_object():
    n = 4
    A = np.eye(n)
    sigma = 0.2
    Sigma_noise = (sigma**2) * np.eye(n)
    mu_prior = np.zeros(n)
    Sigma_prior = np.eye(n)

    pde_like = _StaticLinearModel(A)
    lgm = LinearGaussianModel(
        model=pde_like,
        Sigma_obs=Sigma_noise,
        mu_prior=mu_prior,
        Sigma_prior=Sigma_prior,
        n_steps=3,
    )
    y = np.arange(1.0, n + 1.0)
    mu_post, Sigma_post = lgm.posterior(y)

    expected_cov = (1.0 / (1.0 + 1.0 / (sigma**2))) * np.eye(n)
    expected_mean = (1.0 / (1.0 + sigma**2)) * y

    assert np.allclose(Sigma_post, expected_cov)
    assert np.allclose(mu_post, expected_mean)


def test_linear_gaussian_posterior_with_selection_matrix():
    n = 6
    sigma = 0.1
    A = np.eye(n)
    Sigma_noise = (sigma**2) * np.eye(n)
    mu_prior = np.zeros(n)
    Sigma_prior = np.eye(n)
    y = np.arange(1.0, n + 1.0)

    lgm = LinearGaussianModel(
        model=A,
        Sigma_obs=Sigma_noise,
        mu_prior=mu_prior,
        Sigma_prior=Sigma_prior,
    )
    idx = np.array([1, 3, 5])
    W = compute_W(N=n, indices=idx, format="selection")
    mu_post, Sigma_post = lgm.posterior(y, W=W)

    gain = 1.0 / (1.0 + sigma**2)
    expected_mean = np.zeros(n)
    expected_mean[idx] = gain * y[idx]
    expected_var_selected = (sigma**2) / (1.0 + sigma**2)

    assert np.allclose(mu_post, expected_mean)
    assert np.allclose(np.diag(Sigma_post)[idx], expected_var_selected)
    assert np.allclose(np.diag(Sigma_post)[[0, 2, 4]], 1.0)
