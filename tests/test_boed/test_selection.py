import numpy as np

from boed.core.noise import NoiseModel
from boed.design.selection import (
    select_sensors_qr_pivot,
    select_sensors_maxvol,
    compare_to_greedy,
)


class DummyModel:
    """Minimal forward model for selection tests."""

    def __init__(self, N: int):
        self.N = N

    def get_transition_matrix(self) -> np.ndarray:
        return np.eye(self.N)


def _candidate_set(candidates_x, candidates_t):
    return {(int(x), int(t)) for t in candidates_t for x in candidates_x}


def test_select_sensors_qr_pivot_basic():
    model = DummyModel(N=4)
    candidates_x = np.array([0, 1, 2])
    candidates_t = np.array([0, 1])

    design = select_sensors_qr_pivot(
        model=model,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=2,
    )

    assert len(design) == 2
    assert len(set(design)) == 2
    assert set(design).issubset(_candidate_set(candidates_x, candidates_t))


def test_select_sensors_maxvol_basic():
    model = DummyModel(N=4)
    candidates_x = np.array([0, 1, 2])
    candidates_t = np.array([0, 1])

    design = select_sensors_maxvol(
        model=model,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=2,
        basis="svd",
    )

    assert len(design) == 2
    assert len(set(design)) == 2
    assert set(design).issubset(_candidate_set(candidates_x, candidates_t))


def test_compare_to_greedy_outputs():
    model = DummyModel(N=4)
    Sigma_prior = np.eye(4)
    noise = NoiseModel(sigma_noise=0.1)
    candidates_x = np.array([0, 1, 2])
    candidates_t = np.array([0, 1])

    results = compare_to_greedy(
        model=model,
        Sigma_prior=Sigma_prior,
        noise_model=noise,
        candidates_x=candidates_x,
        candidates_t=candidates_t,
        n_budget=3,
        criterion_type="A",
    )

    for key in ["greedy", "qr", "maxvol"]:
        assert key in results
        assert len(results[key]["design"]) == 3
        assert len(results[key]["history"]) == 3
        Sigma_post = results[key]["Sigma_post"]
        assert Sigma_post.shape == (4, 4)
        assert np.allclose(Sigma_post, Sigma_post.T)
