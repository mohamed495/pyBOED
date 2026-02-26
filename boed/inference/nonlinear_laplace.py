"""Nonlinear inverse models with MAP, Laplace approximation, and pCN sampling."""

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize

from .base import InverseModel

class NonLinearInverseLaplace(InverseModel):
    """
    Nonlinear inverse model — MAP + Laplace approximation + pCN sampler.

    Works with any PDE model that exposes:
        - model.N                                    : spatial dimension
        - model.evolve(u0, n_steps)                  : (n_steps+1, N) trajectory
        - model.get_forward_operator(n_steps, u0_ref): (N, N) linearized propagator

    The Jacobian J = H_eff @ G is built from get_forward_operator(),
    so no adjoint implementation is needed in this class.

    Observations at obs_steps (default: final step only):
        y_k = H u^{n_k} + eps_k,   eps_k ~ N(0, Sigma_obs)

    Optional compression (same interface as LinearGaussianModel):
        obs_map (preferred): indices / mask / projection
        U : (p, m)   -> y_eff = U.T y   (legacy)
        W : diag 0/1 -> y_eff = y[idx]  (legacy)

    Usage
    -----
    Works out of the box with BurgersNonLinear_CN, AdvectionDiffusion1D_CN,
    or any model implementing the interface above.

        from boed.inference import NonlinearLaplaceModel

        model = NonlinearLaplaceModel(
            pde_model=BurgersNonLinear_CN(N=200, dt=0.01),
            H=H, Sigma_obs=..., mu_prior=..., Sigma_prior=...,
            obs_steps=[5, 10, 20],
        )
        res          = model.map_estimate(y, theta_init=mu0, n_steps=20)
        laplace      = model.laplace_approximation(res.x, n_steps=20)
        samples      = model.sample(y, n_samples=500, n_steps=20)
    """

    def __init__(
        self,
        pde_model,
        H: np.ndarray,
        Sigma_obs: np.ndarray,
        mu_prior: np.ndarray,
        Sigma_prior: np.ndarray,
        obs_steps=None,
    ):
        """
        Parameters
        ----------
        pde_model   : any model with .N, .evolve(), .get_forward_operator()
        H           : (m, N) spatial observation operator
        Sigma_obs   : (m, m) observation noise covariance (same at each obs step)
        mu_prior    : (N,) prior mean
        Sigma_prior : (N, N) prior covariance
        obs_steps   : list of time indices to observe (None = final step only)
        """
        self.pde = pde_model
        self.N = pde_model.N
        self.H = H
        self.Sigma_obs = Sigma_obs
        self.mu0 = mu_prior
        self.Sigma0 = Sigma_prior
        self.Q = la.inv(Sigma_prior)
        self.obs_steps = None if obs_steps is None else np.array(obs_steps, dtype=int)

        # cache
        self.theta_map_: np.ndarray | None = None
        self.Sigma_post_: np.ndarray | None = None
        self.acceptance_rate_: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_time_horizon(
        T: int | None = None,
        n_steps: int | None = None,
    ) -> int:
        """Resolve legacy ``T`` and canonical ``n_steps`` aliases."""
        if T is None and n_steps is None:
            raise ValueError("T (or n_steps) is required.")
        if T is not None and n_steps is not None and int(T) != int(n_steps):
            raise ValueError(f"Received inconsistent values for T and n_steps ({T} != {n_steps}).")
        return int(T if T is not None else n_steps)

    def _resolve_obs_map(
        self,
        obs_map=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
    ):
        """Normalize new/legacy observation mapping inputs."""
        if obs_map is not None and (W is not None or U is not None):
            raise ValueError("Provide either 'obs_map' or legacy 'W'/'U' arguments, not both.")

        from boed.inference.observation_map import ObservationMap, normalize_observation_map

        n_obs = int(self.H.shape[0])
        if obs_map is not None:
            return normalize_observation_map(obs_map, n_obs=n_obs)
        if U is not None:
            return ObservationMap.from_projection(U, n_obs=n_obs)
        if W is not None:
            return normalize_observation_map(W, n_obs=n_obs)
        return ObservationMap.identity(n_obs)

    def _compressed_observation_operator(self, obs_map):
        """Return compressed observation operator and noise covariance."""
        if obs_map.is_identity:
            return self.H, self.Sigma_obs

        U_eff = obs_map.as_projection()
        H_eff = U_eff.T @ self.H
        S_eff = U_eff.T @ self.Sigma_obs @ U_eff
        return H_eff, S_eff

    @staticmethod
    def _solve_block_noise_system(S_eff: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Solve ``(I_K ⊗ S_eff) Z = X`` for vector/matrix right-hand sides."""
        X_arr = np.asarray(X, dtype=float)
        m_eff = int(S_eff.shape[0])
        if S_eff.ndim != 2 or S_eff.shape[0] != S_eff.shape[1]:
            raise ValueError("S_eff must be a square matrix.")

        if X_arr.ndim == 1:
            if X_arr.size % m_eff != 0:
                raise ValueError(
                    "Vector length must be a multiple of the compressed observation dimension."
                )
            n_blocks = X_arr.size // m_eff
            solved = la.solve(S_eff, X_arr.reshape(n_blocks, m_eff).T).T
            return solved.reshape(-1)

        if X_arr.ndim == 2:
            if X_arr.shape[0] % m_eff != 0:
                raise ValueError(
                    "Matrix row count must be a multiple of the compressed observation dimension."
                )
            n_blocks = X_arr.shape[0] // m_eff
            blocks = [
                la.solve(S_eff, X_arr[k * m_eff : (k + 1) * m_eff, :])
                for k in range(n_blocks)
            ]
            return np.vstack(blocks)

        raise ValueError("X must be a 1D or 2D array.")

    def _get_obs_steps(self, T: int) -> np.ndarray:
        return np.array([T], dtype=int) if self.obs_steps is None else self.obs_steps

    def _get_H_eff_S_eff_J(
        self,
        theta: np.ndarray,
        T: int,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build compressed (H_eff, S_eff) and stacked Jacobian J.

        J is built from get_forward_operator() — no adjoint needed.
        For each observed time t:
            G_t = get_forward_operator(t, u0_ref=theta)   # (N, N)
            J_t = H_eff @ G_t                              # (m_eff, N)
        J = vstack([J_t for t in obs_steps])               # (K*m_eff, N)
        """
        resolved_obs_map = self._resolve_obs_map(obs_map=obs_map, W=W, U=U)
        H_eff, S_eff = self._compressed_observation_operator(resolved_obs_map)
        obs_steps = self._get_obs_steps(T)

        blocks = [
            H_eff @ self.pde.get_forward_operator(n_steps=int(t), u0_ref=theta)
            for t in obs_steps
        ]
        J = np.vstack(blocks)
        return H_eff, S_eff, J

    def _residual(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        T: int,
        H_eff: np.ndarray,
    ) -> np.ndarray:
        """Stacked residual r = y - [H_eff u^{t_k}]_k."""
        obs_steps = self._get_obs_steps(T)
        traj = self.pde.evolve(theta, n_steps=T)
        y_pred = np.concatenate([H_eff @ traj[int(t)] for t in obs_steps])
        return y - y_pred

    # ------------------------------------------------------------------
    # Likelihood / Prior
    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        T: int | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> float:
        T = self._resolve_time_horizon(T=T, n_steps=n_steps)
        H_eff, S_eff, _ = self._get_H_eff_S_eff_J(theta, T, W=W, U=U, obs_map=obs_map)
        r = self._residual(theta, y, T, H_eff)
        Rinv_r = self._solve_block_noise_system(S_eff, r)
        return float(-0.5 * (r @ Rinv_r))

    def log_prior(self, theta: np.ndarray) -> float:
        d = theta - self.mu0
        return -0.5 * d @ (self.Q @ d)

    # ------------------------------------------------------------------
    # MAP
    # ------------------------------------------------------------------

    def cost_and_grad(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        T: int | None = None,
        n_steps: int | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Negative log-posterior and gradient.
        Gradient of log-likelihood via J (from get_forward_operator):
            grad = J.T @ R^{-1} @ r
        No custom adjoint needed.
        """
        T = self._resolve_time_horizon(T=T, n_steps=n_steps)
        H_eff, S_eff, J = self._get_H_eff_S_eff_J(theta, T, W=W, U=U, obs_map=obs_map)
        r = self._residual(theta, y, T, H_eff)
        d = theta - self.mu0

        # log-likelihood gradient: J.T R^{-1} r with R = I_K ⊗ S_eff.
        Rinv_r = self._solve_block_noise_system(S_eff, r)
        grad_loglik = J.T @ Rinv_r
        grad_logprior = -(self.Q @ d)

        loglik = -0.5 * r @ Rinv_r
        logprior = -0.5 * d @ (self.Q @ d)

        J_cost = -(loglik + logprior)
        grad_cost = -(grad_loglik + grad_logprior)
        return J_cost, grad_cost

    def map_estimate(
        self,
        y: np.ndarray,
        theta_init: np.ndarray,
        T: int | None = None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        maxiter: int = 200,
        method: str = "L-BFGS-B",
        n_steps: int | None = None,
    ):
        """Compute MAP by minimizing negative log-posterior."""
        T = self._resolve_time_horizon(T=T, n_steps=n_steps)

        def fun(th):
            return self.cost_and_grad(th, y, W=W, U=U, obs_map=obs_map, T=T)

        res = minimize(fun, theta_init, jac=True, method=method,
                       options={"maxiter": maxiter})
        self.theta_map_ = res.x
        return res

    # ------------------------------------------------------------------
    # Laplace approximation
    # ------------------------------------------------------------------

    def laplace_posterior(
        self,
        theta_star: np.ndarray,
        T: int | None = None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        n_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Laplace approximation at theta_star:
            Sigma_post ≈ (Q + J^T R^{-1} J)^{-1}

        J comes from get_forward_operator — no adjoint needed.

        Returns
        -------
        Sigma_post : (N, N)
        Precision  : (N, N)
        J          : (K*m_eff, N)
        """
        T = self._resolve_time_horizon(T=T, n_steps=n_steps)
        _, S_eff, J = self._get_H_eff_S_eff_J(theta_star, T, W=W, U=U, obs_map=obs_map)
        Rinv_J = self._solve_block_noise_system(S_eff, J)
        Precision = self.Q + J.T @ Rinv_J
        Sigma_post = la.inv(Precision)
        self.Sigma_post_ = Sigma_post
        return Sigma_post, Precision, J

    def laplace_approximation(
        self,
        theta_map: np.ndarray,
        n_steps: int,
        obs_map=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
    ):
        """Preferred name for the Laplace approximation with structured output."""
        from boed.inference.results import LaplaceResult

        Sigma_post, Precision, J = self.laplace_posterior(
            theta_map,
            T=n_steps,
            W=W,
            U=U,
            obs_map=obs_map,
        )
        resolved_obs_map = self._resolve_obs_map(obs_map=obs_map, W=W, U=U)
        return LaplaceResult(
            mean=np.asarray(theta_map, dtype=float).copy(),
            cov=Sigma_post,
            precision=Precision,
            jacobian=J,
            metadata={
                "n_steps": int(n_steps),
                "obs_steps": self._get_obs_steps(int(n_steps)).copy(),
                "obs_dim": resolved_obs_map.reduced_dim,
                "compression_kind": resolved_obs_map.kind,
            },
        )

    # ------------------------------------------------------------------
    # D-opt score
    # ------------------------------------------------------------------

    def d_opt_score(
        self,
        theta_star: np.ndarray,
        T: int | None = None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        n_steps: int | None = None,
    ) -> float:
        """
        D-opt score = -log det(Sigma_post).
        Larger values are more informative (equivalent to maximizing ``-D_opt``).
        """
        T = self._resolve_time_horizon(T=T, n_steps=n_steps)
        Sigma_post, _, _ = self.laplace_posterior(theta_star, T=T, W=W, U=U, obs_map=obs_map)
        _, logdet = la.slogdet(Sigma_post)
        return -logdet

    def d_optimality(
        self,
        theta_map: np.ndarray,
        n_steps: int,
        obs_map=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        sign: str = "maximize",
    ) -> float:
        """Canonical D-optimality score with explicit sign convention.

        Parameters
        ----------
        sign : {"maximize", "minimize"}
            - ``"maximize"`` returns ``-logdet(Sigma_post)`` (larger is better)
            - ``"minimize"`` returns ``+logdet(Sigma_post)`` (smaller is better)
        """
        Sigma_post, _, _ = self.laplace_posterior(
            theta_map,
            T=n_steps,
            W=W,
            U=U,
            obs_map=obs_map,
        )
        _, logdet = la.slogdet(Sigma_post)
        if sign == "maximize":
            return float(-logdet)
        if sign == "minimize":
            return float(logdet)
        raise ValueError("sign must be either 'maximize' or 'minimize'.")

    # ------------------------------------------------------------------
    # pCN sampler
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        y: np.ndarray,
        n_samples: int,
        T: int | None = None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        theta_init: np.ndarray | None = None,
        step_size: float = 0.1,
        n_burnin: int = 500,
        n_steps: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        pCN sampler preconditioned by the Laplace covariance.

        Workflow:
            1. If theta_init is None -> MAP as warm start
            2. If Sigma_post_ not cached -> laplace_posterior
            3. pCN proposals: theta_prop = theta + step_size * L @ z

        Parameters
        ----------
        step_size : float
            Proposal scale (target acceptance ~0.234 in high dim)
        n_burnin  : int
            Burn-in steps (discarded)

        Returns
        -------
        samples : np.ndarray, shape (n_samples, N)
        """
        T = self._resolve_time_horizon(T=T, n_steps=n_steps)

        if theta_init is None:
            res = self.map_estimate(y, self.mu0.copy(), T=T, W=W, U=U, obs_map=obs_map)
            theta_init = res.x

        if self.Sigma_post_ is None:
            self.laplace_posterior(theta_init, T=T, W=W, U=U, obs_map=obs_map)

        L = np.linalg.cholesky(self.Sigma_post_)
        theta = theta_init.copy()
        samples = np.zeros((n_samples, self.N))
        logp = self.log_likelihood(y, theta, W=W, U=U, obs_map=obs_map, T=T) + self.log_prior(theta)
        n_accepted = 0

        for k in range(n_burnin + n_samples):
            theta_prop = theta + step_size * (L @ np.random.randn(self.N))
            logp_prop = (self.log_likelihood(y, theta_prop, W=W, U=U, obs_map=obs_map, T=T)
                         + self.log_prior(theta_prop))

            if np.log(np.random.rand()) < logp_prop - logp:
                theta, logp = theta_prop, logp_prop
                if k >= n_burnin:
                    n_accepted += 1

            if k >= n_burnin:
                samples[k - n_burnin] = theta

        self.acceptance_rate_ = n_accepted / n_samples
        return samples

    def sample(
        self,
        y: np.ndarray,
        n_samples: int,
        n_steps: int,
        obs_map=None,
        method: str = "pcn",
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        theta_init: np.ndarray | None = None,
        step_size: float = 0.1,
        n_burnin: int = 500,
        **kwargs,
    ):
        """Preferred nonlinear posterior sampling entry point with structured output."""
        from boed.inference.results import MCMCResult

        if method.lower() != "pcn":
            raise ValueError("Only method='pcn' is currently supported.")

        samples = self.sample_posterior(
            y=y,
            n_samples=n_samples,
            T=n_steps,
            W=W,
            U=U,
            obs_map=obs_map,
            theta_init=theta_init,
            step_size=step_size,
            n_burnin=n_burnin,
            **kwargs,
        )
        resolved_obs_map = self._resolve_obs_map(obs_map=obs_map, W=W, U=U)
        return MCMCResult(
            samples=samples,
            acceptance_rate=self.acceptance_rate_,
            initial_state=None if theta_init is None else np.asarray(theta_init, dtype=float).copy(),
            metadata={
                "method": "pcn",
                "n_steps": int(n_steps),
                "obs_steps": self._get_obs_steps(int(n_steps)).copy(),
                "obs_dim": resolved_obs_map.reduced_dim,
                "compression_kind": resolved_obs_map.kind,
            },
        )


# Canonical public alias
NonlinearLaplaceModel = NonLinearInverseLaplace
