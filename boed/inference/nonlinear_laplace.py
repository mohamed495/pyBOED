"""Nonlinear inverse model with MAP, Laplace approximation, and pCN sampling."""

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize

from .base import InverseModel


class NonLinearInverseLaplace(InverseModel):
    """
    Nonlinear inverse model.

    Forward model:
        u(theta) = pde.evolve(theta, n_steps)

    Observation model:
        y = u(theta) + eps,  eps ~ N(0, Sigma_obs)

    Design/compression:
        y_m = W.T @ y

    Notes
    -----
    - If ``W`` is omitted, the identity is used.
    """

    def __init__(
        self,
        pde_model,
        Sigma_obs: np.ndarray,
        mu_prior: np.ndarray,
        Sigma_prior: np.ndarray,
        obs_steps=None,
    ):
        self.pde = pde_model
        self.N = int(pde_model.N)

        Sigma_obs_arr = np.asarray(Sigma_obs, dtype=float)
        if Sigma_obs_arr.shape != (self.N, self.N):
            raise ValueError(
                f"Sigma_obs must have shape ({self.N}, {self.N}), got {Sigma_obs_arr.shape}."
            )
        self.Sigma_obs = Sigma_obs_arr

        mu_arr = np.asarray(mu_prior, dtype=float).reshape(-1)
        if mu_arr.size != self.N:
            raise ValueError(f"mu_prior must have length {self.N}, got {mu_arr.size}.")
        self.mu0 = mu_arr

        Sigma_prior_arr = np.asarray(Sigma_prior, dtype=float)
        if Sigma_prior_arr.shape != (self.N, self.N):
            raise ValueError(
                f"Sigma_prior must have shape ({self.N}, {self.N}), got {Sigma_prior_arr.shape}."
            )
        self.Sigma0 = Sigma_prior_arr
        self.Q = la.inv(Sigma_prior_arr)

        self.obs_steps = None if obs_steps is None else np.asarray(obs_steps, dtype=int)

        self.theta_map_: np.ndarray | None = None
        self.Sigma_post_: np.ndarray | None = None
        self.acceptance_rate_: float | None = None

    def theta_vector(self, theta, arg_name: str = "theta") -> np.ndarray:
        theta_candidate = theta.x if hasattr(theta, "x") else theta
        theta_vec = np.asarray(theta_candidate, dtype=float).reshape(-1)
        if theta_vec.size != self.N:
            raise ValueError(
                f"{arg_name} must have length {self.N}, got {theta_vec.size}."
            )
        return theta_vec

    @staticmethod
    def resolve_steps(T: int | None = None, n_steps: int | None = None) -> int:
        if T is None and n_steps is None:
            raise ValueError("T (or n_steps) is required.")
        if T is not None and n_steps is not None and int(T) != int(n_steps):
            raise ValueError(f"Inconsistent T and n_steps ({T} != {n_steps}).")
        return int(T if T is not None else n_steps)

    def resolve_W(self, W: np.ndarray | None = None) -> np.ndarray:
        if W is None:
            return np.eye(self.N)

        W_arr = np.asarray(W, dtype=float)
        if W_arr.ndim != 2 or W_arr.shape[0] != self.N:
            raise ValueError(
                f"W must have shape ({self.N}, m), got {W_arr.shape}."
            )
        if W_arr.shape[1] <= 0:
            raise ValueError("W must have at least one column.")
        return W_arr

    def observation_steps(self, n_steps: int) -> np.ndarray:
        obs_steps = (
            np.array([n_steps], dtype=int)
            if self.obs_steps is None
            else np.asarray(self.obs_steps, dtype=int)
        )
        if obs_steps.ndim != 1 or obs_steps.size == 0:
            raise ValueError("obs_steps must be a non-empty 1D array.")
        if np.any(obs_steps < 0) or np.any(obs_steps > n_steps):
            raise ValueError(f"obs_steps must lie in [0, {n_steps}].")
        return obs_steps

    def stack_observations(
        self,
        y: np.ndarray,
        n_steps: int,
        obs_steps: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        k = int(obs_steps.size)
        m_full = self.N
        m_eff = int(W.shape[1])
        expected_sizes = {k * m_full, k * m_eff}

        def compress(blocks: np.ndarray) -> np.ndarray:
            if blocks.shape[1] == m_eff:
                return blocks
            if blocks.shape[1] == m_full:
                return blocks @ W
            raise ValueError(
                f"Observation block width must be {m_eff} or {m_full}, got {blocks.shape[1]}."
            )

        if y_arr.ndim == 1:
            if y_arr.size not in expected_sizes:
                raise ValueError(
                    f"Observation vector length mismatch: got {y_arr.size}, "
                    f"expected {k * m_eff} or {k * m_full}."
                )
            width = m_eff if y_arr.size == k * m_eff else m_full
            return compress(y_arr.reshape(k, width)).reshape(-1)

        if y_arr.ndim == 2:
            if 1 in y_arr.shape and y_arr.size in expected_sizes:
                width = m_eff if y_arr.size == k * m_eff else m_full
                return compress(y_arr.reshape(k, width)).reshape(-1)

            if y_arr.shape[0] == k and y_arr.shape[1] in (m_eff, m_full):
                return compress(y_arr).reshape(-1)
            if y_arr.shape[1] == k and y_arr.shape[0] in (m_eff, m_full):
                return compress(y_arr.T).reshape(-1)

            if y_arr.shape[0] == n_steps + 1 and y_arr.shape[1] in (m_eff, m_full):
                return compress(y_arr[obs_steps, :]).reshape(-1)
            if y_arr.shape[1] == n_steps + 1 and y_arr.shape[0] in (m_eff, m_full):
                return compress(y_arr[:, obs_steps].T).reshape(-1)

            raise ValueError(
                "Unsupported 2D observation layout. "
                "Use (K,m), (m,K), (n_steps+1,m), or (m,n_steps+1)."
            )

        raise ValueError("y must be a 1D or 2D array.")

    @staticmethod
    def solve_block_system(S: np.ndarray, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        m = int(S.shape[0])
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError("S must be square.")

        if X_arr.ndim == 1:
            if X_arr.size % m != 0:
                raise ValueError("Vector size must be a multiple of m.")
            k = X_arr.size // m
            return la.solve(S, X_arr.reshape(k, m).T).T.reshape(-1)

        if X_arr.ndim == 2:
            if X_arr.shape[0] % m != 0:
                raise ValueError("Matrix row count must be a multiple of m.")
            k = X_arr.shape[0] // m
            out = [la.solve(S, X_arr[i * m : (i + 1) * m, :]) for i in range(k)]
            return np.vstack(out)

        raise ValueError("X must be 1D or 2D.")

    def log_likelihood(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        W: np.ndarray | None = None,
        T: int | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> float:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta)
        W_eff = self.resolve_W(W=W)
        obs_steps = self.observation_steps(n_steps_eff)

        y_vec = self.stack_observations(
            y=y,
            n_steps=n_steps_eff,
            obs_steps=obs_steps,
            W=W_eff,
        )

        traj = self.pde.evolve(theta_vec, n_steps=n_steps_eff)
        y_pred = np.concatenate([W_eff.T @ traj[int(t)] for t in obs_steps])
        r = y_vec - y_pred

        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        S_inv_r = self.solve_block_system(S_eff, r)
        return float(-0.5 * (r @ S_inv_r))

    def log_prior(self, theta: np.ndarray, mu : np.ndarray) -> float:
        theta_vec = self.theta_vector(theta)
        d = theta_vec - mu
        return float(-0.5 * d @ (self.Q @ d))

    def cost_and_grad(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
        T: int | None = None,
        n_steps: int | None = None,
        **kwargs,
    ) -> tuple[float, np.ndarray]:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta)
        W_eff = self.resolve_W(W=W)
        obs_steps = self.observation_steps(n_steps_eff)

        y_vec = self.stack_observations(
            y=y,
            n_steps=n_steps_eff,
            obs_steps=obs_steps,
            W=W_eff,
        )

        traj = self.pde.evolve(theta_vec, n_steps=n_steps_eff)
        y_pred = np.concatenate([W_eff.T @ traj[int(t)] for t in obs_steps])
        r = y_vec - y_pred

        J_blocks = []
        for t in obs_steps:
            G_t = self.pde.get_forward_operator(n_steps=int(t), u0_ref=theta_vec)
            J_blocks.append(W_eff.T @ G_t)
        J = np.vstack(J_blocks)

        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        S_inv_r = self.solve_block_system(S_eff, r)

        d = theta_vec - self.mu0
        loglik = -0.5 * r @ S_inv_r
        logprior = -0.5 * d @ (self.Q @ d)

        grad_loglik = J.T @ S_inv_r
        grad_logprior = -(self.Q @ d)

        cost = -(loglik + logprior)
        grad = -(grad_loglik + grad_logprior)
        return float(cost), grad

    def map_estimate(
        self,
        y: np.ndarray,
        theta_init: np.ndarray,
        T: int | None = None,
        n_steps: int | None = None,
        W: np.ndarray | None = None,
        maxiter: int = 200,
        method: str = "L-BFGS-B",
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta0 = self.theta_vector(theta_init, arg_name="theta_init")
        W_eff = self.resolve_W(W=W)

        def objective(th):
            return self.cost_and_grad(
                theta=th,
                y=y,
                T=n_steps_eff,
                W=W_eff,
            )

        res = minimize(
            objective,
            theta0,
            jac=True,
            method=method,
            options={"maxiter": maxiter},
        )
        self.theta_map_ = np.asarray(res.x, dtype=float).copy()
        return res

    def jacobian(self,
        theta_star: np.ndarray,
        obs_steps: int | None = None,
        W: np.ndarray | None = None,
        **kwargs,
    ) ->  np.ndarray:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")
        W_eff = self.resolve_W(W=W)
        J_blocks = []
        for t in obs_steps:
            G_t = self.pde.get_forward_operator(n_steps=int(t), u0_ref=theta_star)
            J_blocks.append(W_eff.T @ G_t)
        J = np.vstack(J_blocks)

        return J

    
    def laplace_posterior(
        self,
        theta_star: np.ndarray,
        T: int | None = None,
        n_steps: int | None = None,
        W: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta_star, arg_name="theta_star")
        W_eff = self.resolve_W(W=W)
        obs_steps = self.observation_steps(n_steps_eff)
        
        J = self.jacobian(theta_star=theta_vec, obs_steps=obs_steps, W=W_eff)

        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        S_inv_J = self.solve_block_system(S_eff, J)
        Precision = self.Q + J.T @ S_inv_J
        Sigma_post = la.inv(Precision)
        self.Sigma_post_ = Sigma_post
        return Sigma_post, Precision, J

    def laplace_approximation(
        self,
        theta_map: np.ndarray,
        n_steps: int,
        W: np.ndarray | None = None,
        **kwargs,
    ):
        from boed.inference.results import LaplaceResult

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        theta_vec = self.theta_vector(theta_map, arg_name="theta_map")
        W_eff = self.resolve_W(W=W)

        Sigma_post, Precision, J = self.laplace_posterior(
            theta_star=theta_vec,
            T=n_steps,
            W=W_eff,
        )
        obs_steps = self.observation_steps(int(n_steps))
        compression_kind = (
            "identity" if np.allclose(W_eff, np.eye(self.N)) else "projection"
        )
        return LaplaceResult(
            mean=theta_vec.copy(),
            cov=Sigma_post,
            precision=Precision,
            jacobian=J,
            metadata={
                "n_steps": int(n_steps),
                "obs_steps": obs_steps.copy(),
                "obs_dim": int(W_eff.shape[1]),
                "compression_kind": compression_kind,
            },
        )
    
    def Sigma_Y(self, theta_star, T=None, n_steps=None, W=None):
        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta_star, arg_name="theta_star")
        W_eff = self.resolve_W(W=W)
        obs_steps = self.observation_steps(n_steps_eff)
        J = self.jacobian(theta_star=theta_vec, obs_steps=obs_steps, W=W_eff)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        return S_eff + J @ self.Sigma0 @ J.T  # (K*m_eff, K*m_eff)

    def expected_fisher_mc(self, n_steps, W=None, N_samples=50):
        """
        Estime E[J^T S^{-1} J] par Monte Carlo sous le prior.
        """
        W_eff = self.resolve_W(W)
        obs_steps = self.observation_steps(n_steps)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        L = la.cholesky(self.Sigma0)

        JtSiJ_samples = []
        for _ in range(N_samples):
            theta_k = self.mu0 + L @ np.random.randn(self.N)
            J_k = self.jacobian(theta_k, obs_steps=obs_steps, W=W_eff)
            S_inv_J = self.solve_block_system(S_eff, J_k)
            JtSiJ_samples.append(J_k.T @ S_inv_J)

        return np.mean(JtSiJ_samples, axis=0)
    

    def H_theta_sigma_obs(self, n_steps, W=None, N_samples=50):
        """
        Cov(J^T Sigma_obs^{-1/2}) = E[J^T S^{-1} J] - E[J]^T S^{-1} E[J]
        """
        W_eff = self.resolve_W(W)
        obs_steps = self.observation_steps(n_steps)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        L = la.cholesky(self.Sigma0)

        J_samples = []
        for _ in range(N_samples):
            theta_k = self.mu0 + L @ np.random.randn(self.N)
            J_k = self.jacobian(theta_k, obs_steps=obs_steps, W=W_eff)
            # J_mean = J_mean + J_k
            J_samples.append(J_k)

        J_samples = np.array(J_samples)  # (N_samples, K*m, p)

        # E[J]
        J_mean = J_samples.mean(axis=0)

        # E[J^T S^{-1} J]
        EF = np.mean([J_k.T @ la.solve(S_eff, J_k) for J_k in J_samples], axis=0)

        # Cov = E[J^T S^{-1} J] - E[J]^T S^{-1} E[J]
        return EF - J_mean.T @ la.solve(S_eff, J_mean)


    def expected_jacobian_mc(self, n_steps, W=None, N_samples=50):
        """Estime E[J(theta)] par Monte Carlo sous le prior."""
        W_eff = self.resolve_W(W)
        obs_steps = self.observation_steps(n_steps)
        L = la.cholesky(self.Sigma0)

        J_samples = []
        for _ in range(N_samples):
            theta_k = self.mu0 + L @ np.random.randn(self.N)
            J_k = self.jacobian(theta_k, obs_steps=obs_steps, W=W_eff)
            J_samples.append(J_k)

        return np.mean(J_samples, axis=0)  # shape (K*m_eff, N)

    def Sigma_signal(self, theta_star, T=None, n_steps=None, W=None, N_samples=50):
        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta_star, arg_name="theta_star")
        W_eff = self.resolve_W(W=W)
        obs_steps = self.observation_steps(n_steps_eff)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff

        # H = Cov(J^T Sigma_obs^{-1/2})
        Htso = self.H_theta_sigma_obs(
            theta_star=theta_vec, T=n_steps_eff, W=W_eff, N_samples=N_samples
        )

        # L = E[J(theta)] — espérance du Jacobien sous le prior
        L = self.expected_jacobian_mc(
            n_steps=n_steps_eff, W=W_eff, N_samples=N_samples
        )

        return S_eff + L.T @ la.inv(self.Q + Htso) @ L

    def eig(self, theta_star, T=None, n_steps=None, W=None):
        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta_star, arg_name="theta_star")
        W_eff = self.resolve_W(W=W)
        SY = self.Sigma_Y(theta_star=theta_vec, n_steps=n_steps_eff, W=W_eff)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        sign_num, logdet_num = la.slogdet(SY)
        sign_den, logdet_den = la.slogdet(S_eff)
        if sign_num <= 0 or sign_den <= 0:
            raise ValueError("Covariances must be positive definite.")
        return float(0.5 * (logdet_num - logdet_den))

    def eig_BI(self, theta_star, W, T=None, n_steps=None, N_samples=50):
        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta_star, arg_name="theta_star")
        W_eff = self.resolve_W(W=W)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff

        SY = self.Sigma_Y(theta_star=theta_vec, n_steps=n_steps_eff, W=None)  # non compressée

        # ine1 = EIG compressé avec SY
        S_num = W_eff.T @ SY @ W_eff
        _, logdet_num = la.slogdet(S_num)
        _, logdet_den = la.slogdet(S_eff)
        ine1 = float(0.5 * (logdet_num - logdet_den))

        # ine2 = EIG non compressé avec SY
        _, logdet_num = la.slogdet(SY)
        _, logdet_den = la.slogdet(self.Sigma_obs)
        ine2 = float(0.5 * (logdet_num - logdet_den))

        return self.eig(theta_star=theta_vec, n_steps=n_steps_eff, W=None) + ine1 - ine2
    
    def eig_BS(self, theta_star, W, T=None, n_steps=None, N_samples=50) -> float:
        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        theta_vec = self.theta_vector(theta_star, arg_name="theta_star")
        W_eff = self.resolve_W(W=W)
        S_eff = W_eff.T @ self.Sigma_obs @ W_eff
        # dénominateur = S_eff, pas W^T Sigma_Y W

        SS = self.Sigma_signal(theta_star=theta_vec, n_steps=n_steps_eff, W=None, N_samples=N_samples)

        # ine1 = EIG compressé avec Sigma_signal
        S_num = W_eff.T @ SS @ W_eff
        _, logdet_num = la.slogdet(S_num)
        _, logdet_den = la.slogdet(S_eff)
        ine1 = float(0.5 * (logdet_num - logdet_den))

        # ine2 = EIG non compressé avec Sigma_signal
        _, logdet_num = la.slogdet(SS)
        _, logdet_den = la.slogdet(self.Sigma_obs)
        ine2 = float(0.5 * (logdet_num - logdet_den))

        return self.eig(theta_star=theta_vec, n_steps=n_steps_eff, W=None) + ine1 - ine2
    

    def greedy_bounds(
        self,
        theta_star: np.ndarray,
        T: int,
        m: int,
        N_samples: int = 50,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Greedy joint : à chaque étape on choisit le capteur qui maximise
        ine1_BI tout en gardant ine1_BS raisonnable — ou un compromis.
        Retourne W_opt (N x m) et la courbe de scores.
        """
        theta_vec = self.theta_vector(theta_star)
        obs_steps = self.observation_steps(T)
        I = np.eye(self.N)

        # Précalcul une seule fois
        J = self.jacobian(theta_star=theta_vec, obs_steps=obs_steps, W=I)
        SY = self.Sigma_obs + J @ self.Sigma0 @ J.T
        SS = self.Sigma_signal(theta_star=theta_vec, n_steps=T, W=None, N_samples=N_samples)
        SN = self.Sigma_obs

        selected = []
        remaining = list(range(self.N))
        history = []

        for k in range(m):
            best_idx, best_score = None, -np.inf
            for i in remaining:
                cols = selected + [i]
                W_cand = I[:, cols]

                # ine1_BI
                SY_c = W_cand.T @ SY @ W_cand
                SN_c = W_cand.T @ SN @ W_cand
                _, ld_num = la.slogdet(SY_c)
                _, ld_den = la.slogdet(SN_c)
                ine1_BI = 0.5 * (ld_num - ld_den)

                # ine1_BS
                SS_c = W_cand.T @ SS @ W_cand
                _, ld_num = la.slogdet(SS_c)
                _, ld_den = la.slogdet(SN_c)
                ine1_BS = 0.5 * (ld_num - ld_den)

                # critère : maximiser BI (on veut serrer la borne inf)
                score = ine1_BI
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_BI = ine1_BI
                    best_BS = ine1_BS

            selected.append(best_idx)
            remaining.remove(best_idx)
            history.append({"step": k+1, "capteur": best_idx, "ine1_BI": best_BI, "ine1_BS": best_BS})
            print(f"Step {k+1}/{m} — capteur {best_idx} | BI={best_BI:.4f} | BS={best_BS:.4f}")

        W_opt = I[:, selected]
        return W_opt, history


    def d_opt_score(
        self,
        theta_star: np.ndarray,
        T: int | None = None,
        n_steps: int | None = None,
        W: np.ndarray | None = None,
        **kwargs,
    ) -> float:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        Sigma_post, _, _ = self.laplace_posterior(
            theta_star=theta_star,
            T=T,
            n_steps=n_steps,
            W=W,
        )
        _, logdet = la.slogdet(Sigma_post)
        return float(-logdet)

    def d_optimality(
        self,
        theta_map: np.ndarray,
        n_steps: int,
        W: np.ndarray | None = None,
        sign: str = "maximize",
        **kwargs,
    ) -> float:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        Sigma_post, _, _ = self.laplace_posterior(
            theta_star=theta_map,
            T=n_steps,
            W=W,
        )
        _, logdet = la.slogdet(Sigma_post)
        if sign == "maximize":
            return float(-logdet)
        if sign == "minimize":
            return float(logdet)
        raise ValueError("sign must be either 'maximize' or 'minimize'.")

    def sample_posterior(
        self,
        y: np.ndarray,
        n_samples: int,
        T: int | None = None,
        n_steps: int | None = None,
        W: np.ndarray | None = None,
        theta_init: np.ndarray | None = None,
        step_size: float = 0.1,
        n_burnin: int = 500,
        **kwargs,
    ) -> np.ndarray:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")

        n_steps_eff = self.resolve_steps(T=T, n_steps=n_steps)
        W_eff = self.resolve_W(W=W)

        if theta_init is None:
            res = self.map_estimate(
                y=y,
                theta_init=self.mu0.copy(),
                T=n_steps_eff,
                W=W_eff,
            )
            theta0 = np.asarray(res.x, dtype=float)
        else:
            theta0 = self.theta_vector(theta_init, arg_name="theta_init")

        if self.Sigma_post_ is None:
            self.laplace_posterior(theta_star=theta0, T=n_steps_eff, W=W_eff)

        L = la.cholesky(self.Sigma_post_)
        theta = theta0.copy()
        samples = np.zeros((n_samples, self.N), dtype=float)

        logp = self.log_likelihood(y=y, theta=theta, T=n_steps_eff, W=W_eff) + self.log_prior(theta)
        accepted = 0

        for k in range(n_burnin + n_samples):
            theta_prop = theta + step_size * (L @ np.random.randn(self.N))
            logp_prop = self.log_likelihood(y=y, theta=theta_prop, T=n_steps_eff, W=W_eff)
            logp_prop += self.log_prior(theta_prop)

            if np.log(np.random.rand()) < (logp_prop - logp):
                theta = theta_prop
                logp = logp_prop
                if k >= n_burnin:
                    accepted += 1

            if k >= n_burnin:
                samples[k - n_burnin, :] = theta

        self.acceptance_rate_ = accepted / n_samples
        return samples

    def sample(
        self,
        y: np.ndarray,
        n_samples: int,
        n_steps: int,
        W: np.ndarray | None = None,
        method: str = "pcn",
        theta_init: np.ndarray | None = None,
        step_size: float = 0.1,
        n_burnin: int = 500,
        **kwargs,
    ):
        from boed.inference.results import MCMCResult

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}.")
        if method.lower() != "pcn":
            raise ValueError("Only method='pcn' is currently supported.")

        W_eff = self.resolve_W(W=W)
        init_state = None if theta_init is None else self.theta_vector(theta_init, arg_name="theta_init")
        samples = self.sample_posterior(
            y=y,
            n_samples=n_samples,
            T=n_steps,
            W=W_eff,
            theta_init=init_state,
            step_size=step_size,
            n_burnin=n_burnin,
        )
        obs_steps = self.observation_steps(int(n_steps))
        compression_kind = (
            "identity" if np.allclose(W_eff, np.eye(self.N)) else "projection"
        )
        return MCMCResult(
            samples=samples,
            acceptance_rate=self.acceptance_rate_,
            initial_state=None if init_state is None else init_state.copy(),
            metadata={
                "method": "pcn",
                "n_steps": int(n_steps),
                "obs_steps": obs_steps.copy(),
                "obs_dim": int(W_eff.shape[1]),
                "compression_kind": compression_kind,
            },
        )


# Canonical public alias
NonlinearLaplaceModel = NonLinearInverseLaplace
