import numpy as np

from boed.inference import LinearGaussianModel


def test_linear_gaussian_posterior_identity():
    n = 5
    A = np.eye(n)
    sigma = 0.1
    Sigma_noise = (sigma**2) * np.eye(n)
    mu_prior = np.zeros(n)
    Sigma_prior = np.eye(n)

    lgm = LinearGaussianModel(A, Sigma_noise, mu_prior, Sigma_prior)
    y = np.arange(1.0, n + 1.0)
    mu_post, Sigma_post = lgm.posterior(y)

    expected_cov = (1.0 / (1.0 + 1.0 / (sigma**2))) * np.eye(n)
    expected_mean = (1.0 / (1.0 + sigma**2)) * y

    assert np.allclose(Sigma_post, expected_cov)
    assert np.allclose(mu_post, expected_mean)
    assert np.allclose(Sigma_post, Sigma_post.T)
