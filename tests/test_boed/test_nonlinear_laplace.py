import numpy as np
from scipy.optimize import OptimizeResult

from boed.inference import NonLinearInverseLaplace


class _IdentityPDE:
    def __init__(self, n: int):
        self.N = int(n)

    def evolve(self, u0: np.ndarray, n_steps: int) -> np.ndarray:
        u_vec = np.asarray(u0, dtype=float).reshape(-1)
        return np.tile(u_vec[None, :], (int(n_steps) + 1, 1))

    def get_forward_operator(self, n_steps: int, u0_ref: np.ndarray | None = None) -> np.ndarray:
        _ = n_steps, u0_ref
        return np.eye(self.N)


def _make_model(n: int, obs_steps) -> NonLinearInverseLaplace:
    return NonLinearInverseLaplace(
        pde_model=_IdentityPDE(n),
        Sigma_obs=np.eye(n),
        mu_prior=np.zeros(n),
        Sigma_prior=np.eye(n),
        obs_steps=obs_steps,
    )


def test_log_likelihood_accepts_row_vector_observation():
    model = _make_model(n=4, obs_steps=[0])
    theta = np.zeros(4)
    y_vec = np.ones(4)
    y_row = np.ones((1, 4))

    ll_vec = model.log_likelihood(y=y_vec, theta=theta, T=0)
    ll_row = model.log_likelihood(y=y_row, theta=theta, T=0)

    assert np.isclose(ll_row, ll_vec)


def test_log_likelihood_uses_obs_steps_from_full_trajectory_input():
    obs_steps = np.array([0, 2, 4], dtype=int)
    model = _make_model(n=3, obs_steps=obs_steps)
    theta = np.zeros(3)
    y_traj = np.arange(15.0).reshape(5, 3)

    ll = model.log_likelihood(y=y_traj, theta=theta, T=4)
    y_expected = np.concatenate([y_traj[int(t)] for t in obs_steps])
    expected = -0.5 * float(y_expected @ y_expected)

    assert np.isclose(ll, expected)


def test_laplace_approximation_accepts_optimize_result():
    model = _make_model(n=3, obs_steps=[0])
    result_like = OptimizeResult(x=np.array([0.1, -0.2, 0.3], dtype=float))

    laplace = model.laplace_approximation(theta_map=result_like, n_steps=0)

    assert laplace.mean.shape == (3,)
    assert laplace.cov.shape == (3, 3)
    assert laplace.precision.shape == (3, 3)


def test_log_likelihood_accepts_projection_W():
    model = _make_model(n=4, obs_steps=[0])
    theta = np.zeros(4)
    y = np.ones(2)
    W = np.eye(4)[:, :2]

    ll = model.log_likelihood(y=y, theta=theta, T=0, W=W)

    assert np.isclose(ll, -1.0)
