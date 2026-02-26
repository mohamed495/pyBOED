import numpy as np
import pytest

from boed.core.noise import NoiseModel
from boed.integration import HybridBayesianInference, PODForwardModel, ReducedPriorDesign
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.priors import GaussianProcessPrior, Gaussian


def test_reduced_prior_design_build_and_guard():
    n = 16
    kernel = Gaussian(length_scale=0.2, sigma=1.0)
    prior = GaussianProcessPrior(kernel, nx=n)
    model = AdvectionDiffusion1D_CN(N=n, dt=0.01, diffusivity=0.01, velocity=0.1)
    noise = NoiseModel(sigma_noise=0.01)

    reduced = ReducedPriorDesign(prior=prior, n_components=8, method="kle")
    assert reduced.Sigma_reduced.shape == (8, 8)

    with pytest.raises(ValueError):
        reduced.run_design(
            model=model,
            noise_model=noise,
            candidates_x=np.arange(n),
            candidates_t=np.arange(2),
            n_budget=1,
            criterion_type="A",
            verbose=False,
        )


def test_hybrid_inference_smoke():
    rng = np.random.default_rng(0)
    n, m = 12, 5
    A = rng.normal(size=(m, n))
    R = 0.05 * np.eye(m)
    mu = np.zeros(n)
    Sigma = np.eye(n)
    y = rng.normal(size=m)

    h = HybridBayesianInference(A, R, mu, Sigma, lis_rank=4)
    h.identify_informative_subspace(y, n_samples=40, verbose=False)
    mu_r, Sigma_r = h.posterior_in_subspace(y)

    assert mu_r.shape == (4,)
    assert Sigma_r.shape == (4, 4)
    assert np.allclose(Sigma_r, Sigma_r.T)


def test_pod_forward_model_smoke():
    model = AdvectionDiffusion1D_CN(N=20, dt=0.01, diffusivity=0.01, velocity=0.1)
    x = np.linspace(0.0, 1.0, model.N)
    samples = [np.sin(np.pi * x), np.sin(2 * np.pi * x)]

    pod_model = PODForwardModel(model, energy_threshold=0.99)
    pod_model.build_reduced_model(samples, n_steps=4, verbose=False)
    traj = pod_model.evolve_reduced(samples[0], n_steps=4, method="galerkin")

    assert traj.shape == (5, model.N)
