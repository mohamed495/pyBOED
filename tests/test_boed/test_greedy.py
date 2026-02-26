import numpy as np
import pytest

from boed.core.noise import NoiseModel
from boed.design.greedy import run_greedy_oed
from boed.design.criteria import DesignCriteria


class DummyModel:
    """Minimal forward model for greedy selection tests."""

    def __init__(self, N: int):
        self.N = N

    def get_transition_matrix(self) -> np.ndarray:
        return np.eye(self.N)


def test_run_greedy_oed_basic():
    model = DummyModel(N=4)
    Sigma_prior = np.eye(4)
    noise = NoiseModel(sigma_noise=0.1)
    candidates_x = np.array([0, 1])
    candidates_t = np.array([0, 1])

    design, history, Sigma_post = run_greedy_oed(
        N=4,
        model=model,
        prior_kernel=Sigma_prior,
        noise_model=noise,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=3,
        criterion_type="A",
    )

    assert len(design) == 3
    assert len(history) == 3
    assert len(set(design)) == 3
    assert Sigma_post.shape == (4, 4)
    assert np.allclose(Sigma_post, Sigma_post.T)


def test_run_greedy_oed_requires_qoi_for_c():
    model = DummyModel(N=3)
    Sigma_prior = np.eye(3)
    noise = NoiseModel(sigma_noise=0.1)
    candidates_x = np.array([0, 1])
    candidates_t = np.array([0])

    with pytest.raises(ValueError, match="L_qoi"):
        run_greedy_oed(
            N=3,
            model=model,
            prior_kernel=Sigma_prior,
            noise_model=noise,
            candidates_x=candidates_x,
            candidates_t=candidates_t,
            n_budget=1,
            criterion_type="C",
        )


def test_c_opt_accepts_1d_qoi_vector():
    Sigma = np.eye(3)
    L = np.array([1.0, 0.0, 0.0])
    score = DesignCriteria.C_opt(Sigma, L)
    assert np.isclose(score, 1.0)
