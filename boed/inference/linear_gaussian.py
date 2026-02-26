"""Analytical linear-Gaussian posterior inference models."""

import numpy as np
import numpy.linalg as la
from scipy.linalg import eigh

from .base import InverseModel

class LinearGaussianModel(InverseModel):
    """
    Linear Gaussian inverse model — analytical posterior.

        Y = A @ theta + epsilon
        theta   ~ N(mu_prior, Sigma_prior)
        epsilon ~ N(0, Sigma_obs)
        theta | Y ~ N(mu_post, Sigma_post)

    Observation compression (optional):
        obs_map (preferred): indices / mask / projection
        U  in R^(p x m)   ->  Y_m = U.T @ Y   (legacy, continuous compression)
        W  diagonal 0/1   ->  Y_m = Y[active] (legacy, binary sensor selection)

    EIG (Expected Information Gain):
        EIG(U) = 1/2 * ln( |U.T Sigma_Y U| / |U.T Sigma_obs U| )
        where Sigma_Y = Sigma_obs + A Sigma_prior A.T

    Optimal U is given by the generalized eigenproblem:
        Sigma_Y v = lambda * Sigma_obs v
    """

    def __init__(
        self,
        A: np.ndarray,
        Sigma_noise: np.ndarray,
        mu_prior: np.ndarray,
        Sigma_prior: np.ndarray,
    ):
        self.A = A
        self.Sigma_eps = Sigma_noise
        self.mu0 = mu_prior
        self.Sigma0 = Sigma_prior
        # aliases for external modules
        self.Sigma_noise = Sigma_noise
        self.mu_prior = mu_prior
        self.Sigma_prior = Sigma_prior

    def _resolve_obs_map(
        self,
        obs_map=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
    ):
        """Normalize new/legacy observation mapping inputs."""
        if obs_map is not None and (W is not None or U is not None):
            raise ValueError("Provide either 'obs_map' or legacy 'W'/'U' arguments, not both.")

        # Lazy import avoids circular imports during package initialization.
        from boed.inference.observation_map import ObservationMap, normalize_observation_map

        n_obs = int(self.A.shape[0])

        if obs_map is not None:
            return normalize_observation_map(obs_map, n_obs=n_obs)
        if U is not None:
            return ObservationMap.from_projection(U, n_obs=n_obs)
        if W is not None:
            return normalize_observation_map(W, n_obs=n_obs)
        return ObservationMap.identity(n_obs)

    def _compressed_linear_system(self, y: np.ndarray, obs_map):
        """Build compressed linear system (A_eff, S_eff, y_eff)."""
        y_eff = obs_map.apply(y)
        if obs_map.is_identity:
            return self.A, self.Sigma_eps, y_eff

        U_eff = obs_map.as_projection()
        A_eff = U_eff.T @ self.A
        S_eff = U_eff.T @ self.Sigma_eps @ U_eff
        return A_eff, S_eff, y_eff

    def _posterior_core(
        self,
        y: np.ndarray,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
    ):
        """Shared posterior computation for tuple and result-returning APIs."""
        resolved_obs_map = self._resolve_obs_map(obs_map=obs_map, W=W, U=U)
        A_eff, S_eff, y_eff = self._compressed_linear_system(y, resolved_obs_map)

        S0_inv = la.inv(self.Sigma0)
        S_eff_inv = la.inv(S_eff)
        Sigma_post = la.inv(A_eff.T @ S_eff_inv @ A_eff + S0_inv)
        mu_post = Sigma_post @ (A_eff.T @ S_eff_inv @ y_eff + S0_inv @ self.mu0)
        return mu_post, Sigma_post, resolved_obs_map

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    def posterior(
        self,
        y: np.ndarray,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytical posterior  theta | y.

        Parameters
        ----------
        y : observations (p,)
        W : diagonal 0/1 sensor selection matrix (p, p)
        U : compression matrix (p, m)  — priority over W
        obs_map : unified observation map (preferred). Mutually exclusive with W/U.

        Returns
        -------
        mu_post, Sigma_post
        """
        mu_post, Sigma_post, _ = self._posterior_core(y, W=W, U=U, obs_map=obs_map)
        return mu_post, Sigma_post

    def posterior_result(
        self,
        y: np.ndarray,
        obs_map=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
    ):
        """Analytical posterior with structured return value."""
        from boed.inference.results import PosteriorResult

        mu_post, Sigma_post, resolved_obs_map = self._posterior_core(y, W=W, U=U, obs_map=obs_map)
        return PosteriorResult(
            mean=mu_post,
            cov=Sigma_post,
            obs_dim=resolved_obs_map.reduced_dim,
            compression_kind=resolved_obs_map.kind,
            metadata={"n_obs_full": int(self.A.shape[0]), "n_params": int(self.A.shape[1])},
        )

    def sample_posterior(
        self,
        y: np.ndarray,
        n_samples: int,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
        **kwargs,
    ) -> np.ndarray:
        """Sample from the Gaussian posterior analytically."""
        mu_post, Sigma_post = self.posterior(y, W=W, U=U, obs_map=obs_map)
        return np.random.multivariate_normal(mu_post, Sigma_post, size=n_samples)

    def sample(
        self,
        y: np.ndarray,
        n_samples: int,
        obs_map=None,
        rng=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        **kwargs,
    ):
        """Sample from the analytical Gaussian posterior and return a structured result."""
        from boed.inference.results import MCMCResult

        post = self.posterior_result(y, obs_map=obs_map, W=W, U=U)
        sampler = np.random if rng is None else rng
        if not hasattr(sampler, "multivariate_normal"):
            raise TypeError("rng must provide a 'multivariate_normal' method.")

        samples = sampler.multivariate_normal(post.mean, post.cov, size=n_samples)
        return MCMCResult(
            samples=samples,
            acceptance_rate=None,
            initial_state=None,
            metadata={
                "method": "analytic_gaussian",
                "obs_dim": post.obs_dim,
                "compression_kind": post.compression_kind,
            },
        )

    # ------------------------------------------------------------------
    # QoI
    # ------------------------------------------------------------------

    def qoi_statistics(
        self,
        G: np.ndarray,
        mu_post: np.ndarray,
        Sigma_post: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Posterior statistics for QoI  z = G @ theta.

        Returns
        -------
        mu_z    = G @ mu_post
        Sigma_z = G @ Sigma_post @ G.T
        """
        return G @ mu_post, G @ Sigma_post @ G.T

    def posterior_qoi(
        self,
        y: np.ndarray,
        G: np.ndarray,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shortcut: posterior + QoI projection."""
        mu_post, Sigma_post = self.posterior(y, W=W, U=U, obs_map=obs_map)
        return self.qoi_statistics(G, mu_post, Sigma_post)

    def project_qoi(self, posterior, G: np.ndarray):
        """Project a posterior result (or ``(mean, cov)`` tuple) to a QoI."""
        from boed.inference.results import PosteriorResult

        if isinstance(posterior, PosteriorResult):
            mu_post = posterior.mean
            Sigma_post = posterior.cov
            obs_dim = posterior.obs_dim
            compression_kind = posterior.compression_kind
            metadata = dict(posterior.metadata)
        else:
            mu_post, Sigma_post = posterior
            obs_dim = None
            compression_kind = None
            metadata = {}

        mu_z, Sigma_z = self.qoi_statistics(G, mu_post, Sigma_post)
        metadata.update({"qoi_dim": int(np.atleast_2d(G).shape[0])})
        return PosteriorResult(
            mean=mu_z,
            cov=Sigma_z,
            obs_dim=obs_dim,
            compression_kind=compression_kind,
            metadata=metadata,
        )

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
        **kwargs,
    ) -> float:
        resolved_obs_map = self._resolve_obs_map(obs_map=obs_map, W=W, U=U)
        A_eff, S_eff, y_eff = self._compressed_linear_system(y, resolved_obs_map)
        res = y_eff - A_eff @ theta
        return -0.5 * res @ la.solve(S_eff, res)

    def log_prior(self, theta: np.ndarray) -> float:
        d = theta - self.mu0
        return -0.5 * d @ la.solve(self.Sigma0, d)

    # ------------------------------------------------------------------
    # EIG
    # ------------------------------------------------------------------

    def Sigma_Y(self) -> np.ndarray:
        """Marginal covariance of Y:  Sigma_Y = Sigma_obs + A Sigma_prior A.T"""
        return self.Sigma_eps + self.A @ self.Sigma0 @ self.A.T

    def eig(
        self,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        obs_map=None,
    ) -> float:
        """
        Analytical EIG for a given observation compression.

            EIG(U) = 1/2 * ln( |U.T Sigma_Y U| / |U.T Sigma_obs U| )

        Parameters
        ----------
        W : legacy selection/mask argument
        U : legacy projection argument (priority over W)
        obs_map : unified observation map (preferred). Mutually exclusive with W/U.

        Returns
        -------
        float
        """
        SY = self.Sigma_Y()
        resolved_obs_map = self._resolve_obs_map(obs_map=obs_map, W=W, U=U)
        U_eff = resolved_obs_map.as_projection()
        S_num = U_eff.T @ SY @ U_eff
        S_den = U_eff.T @ self.Sigma_eps @ U_eff

        sign_num, logdet_num = la.slogdet(S_num)
        sign_den, logdet_den = la.slogdet(S_den)

        if sign_num <= 0 or sign_den <= 0:
            raise ValueError("Non positive-definite compressed covariance. Check the observation map.")

        return 0.5 * (logdet_num - logdet_den)

    def expected_information_gain(
        self,
        obs_map=None,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
    ) -> float:
        """Preferred name for analytical EIG in the linear-Gaussian model."""
        return float(self.eig(W=W, U=U, obs_map=obs_map))

    def optimize_U(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Optimal compression matrix U* in R^(p, m) maximizing EIG.

        Solves the generalized eigenproblem:
            Sigma_Y v = lambda * Sigma_obs v

        The m eigenvectors with the largest eigenvalues form U*.
        Each eigenvalue lambda_i is the signal-to-noise ratio in direction v_i:
            EIG(U*) = 1/2 * sum_i ln(lambda_i)

        Parameters
        ----------
        m : int
            Number of directions to keep.

        Returns
        -------
        U_star      : np.ndarray, shape (p, m)
        eigenvalues : np.ndarray, shape (m,)  — signal-to-noise ratios
        """
        SY = self.Sigma_Y()
        eigenvalues, eigenvectors = eigh(SY, self.Sigma_eps)
        idx = np.argsort(eigenvalues)[::-1][:m]
        return eigenvectors[:, idx], eigenvalues[idx]

    def optimal_compression(self, rank: int):
        """Preferred name for optimal EIG compression directions."""
        from boed.inference.observation_map import ObservationMap

        U_star, eigenvalues = self.optimize_U(rank)
        return ObservationMap.from_projection(U_star, n_obs=self.A.shape[0]), eigenvalues

    def eig_optimal(self, m: int) -> float:
        """
        Maximum EIG achievable with m directions.
            = 1/2 * sum of log of the m largest generalized eigenvalues.
        """
        _, lambdas = self.optimize_U(m)
        return 0.5 * np.sum(np.log(lambdas))

    def optimal_expected_information_gain(self, rank: int) -> float:
        """Preferred name for the best EIG achievable at a given compression rank."""
        return float(self.eig_optimal(rank))
