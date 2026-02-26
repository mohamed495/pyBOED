"""Design optimality criteria for Bayesian Optimal Experimental Design.

Implements various criteria for evaluating and selecting experimental designs
in inverse problems with Gaussian parameters.
"""
import numpy as np
import numpy.linalg as la


class DesignCriteria:
    """Static methods for computing design optimality criteria.
    
    Provides multiple criteria for evaluating experimental designs based on
    posterior parameter uncertainty. Each criterion quantifies design quality
    for different objectives.
    
    All methods are static and work with posterior covariance matrices.
    
    Notes
    -----
    - A-optimality (trace): Minimizes total parameter variance
    - D-optimality (log-det): Minimizes the volume of uncertainty region
    - C-optimality: Targets variance of a linear functional of parameters
    - EIG: Quantifies information gain from observations
    
    Examples
    --------
    Evaluate different criteria after computing posterior covariance:
    
    >>> from boed.design.criteria import DesignCriteria
    >>> import numpy as np
    >>> 
    >>> # After inference: obtain posterior covariance Sigma_post
    >>> # and prior covariance Sigma_prior
    >>> 
    >>> # A-optimality (minimize trace)
    >>> a_score = DesignCriteria.A_opt(Sigma_post)
    >>> 
    >>> # D-optimality (minimize log-determinant)
    >>> d_score = DesignCriteria.D_opt(Sigma_post)
    >>> 
    >>> # C-optimality for a linear functional q = L @ u
    >>> L = np.eye(10)[:5, :]  # Observe first 5 parameters
    >>> c_score = DesignCriteria.C_opt(Sigma_post, L)
    >>> 
    >>> # Expected Information Gain
    >>> eig = DesignCriteria.EIG(Sigma_post, Sigma_prior)
    """

    @staticmethod
    def _is_binary_diagonal_mask(matrix: np.ndarray, tol: float = 1e-12) -> bool:
        """Return True if ``matrix`` is a square binary diagonal mask."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False

        if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=tol, rtol=0.0):
            return False

        diag = np.diag(matrix)
        return bool(
            np.all(np.isclose(diag, 0.0, atol=tol) | np.isclose(diag, 1.0, atol=tol))
        )

    @staticmethod
    def _projection_from_observation_input(
        n_obs: int,
        *,
        obs_map=None,
        W: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize observation selection/compression input to a projection matrix.

        Parameters
        ----------
        n_obs : int
            Full observation dimension.
        obs_map : optional
            New unified observation mapping input. Accepted formats:
            - ObservationMap-like object exposing ``as_projection()``
            - 1D array of indices
            - 2D binary diagonal mask
            - 2D projection matrix
        W : np.ndarray, optional
            Legacy compression matrix argument (kept for backward compatibility).

        Returns
        -------
        np.ndarray
            Projection matrix ``U`` of shape ``(n_obs, m)``.
        """
        if obs_map is not None and W is not None:
            raise ValueError("Provide either 'obs_map' or legacy 'W', not both.")

        if obs_map is None and W is None:
            return np.eye(n_obs)

        if obs_map is None:
            U = np.asarray(W, dtype=float)
            if U.ndim != 2:
                raise ValueError("Legacy 'W' compression argument must be a 2D matrix.")
            if U.shape[0] != n_obs:
                raise ValueError(
                    "Legacy 'W' compression matrix row dimension must match "
                    f"the observation dimension ({U.shape[0]} != {n_obs})."
                )
            if U.shape[1] == 0:
                raise ValueError("Compression matrix must have at least one column.")
            return U

        if hasattr(obs_map, "as_projection"):
            U = np.asarray(obs_map.as_projection(), dtype=float)
        else:
            arr = np.asarray(obs_map)
            if arr.ndim == 1:
                idx = arr.astype(int, copy=False)
                if np.any(idx < 0) or np.any(idx >= n_obs):
                    raise ValueError(f"Sensor indices must be in [0, {n_obs - 1}].")
                if np.unique(idx).size != idx.size:
                    raise ValueError("Sensor indices must be unique.")
                U = np.eye(n_obs)[:, idx]
            elif arr.ndim == 2:
                mat = np.asarray(arr, dtype=float)
                if DesignCriteria._is_binary_diagonal_mask(mat):
                    idx = np.flatnonzero(np.diag(mat) > 0.5)
                    if idx.size == 0:
                        raise ValueError("Mask must activate at least one observation.")
                    U = np.eye(n_obs)[:, idx]
                else:
                    U = mat
            else:
                raise TypeError(
                    "obs_map must be an ObservationMap-like object, a 1D index array, "
                    "a 2D binary diagonal mask, or a 2D projection matrix."
                )

        if U.ndim != 2:
            raise ValueError("Projection matrix must be 2D.")
        if U.shape[0] != n_obs:
            raise ValueError(
                f"Projection matrix row dimension must match observation dimension ({U.shape[0]} != {n_obs})."
            )
        if U.shape[1] == 0:
            raise ValueError("Projection matrix must have at least one column.")
        return U

    @staticmethod
    def _eig_from_observation_cov_core(
        Sigma_y: np.ndarray,
        Sigma_noise: np.ndarray,
        U: np.ndarray,
    ) -> float:
        """Core observation-covariance EIG computation using a projection matrix."""
        S_num = U.T @ Sigma_y @ U
        S_den = U.T @ Sigma_noise @ U

        sign_num, logdet_num = la.slogdet(S_num)
        sign_den, logdet_den = la.slogdet(S_den)

        if sign_num <= 0 or sign_den <= 0:
            raise ValueError("Compressed covariance must be positive definite.")

        return float(0.5 * (logdet_num - logdet_den))
    
    @staticmethod
    def A_opt(Sigma_post: np.ndarray) -> float:
        """A-optimality (trace criterion).
        
        Minimizes the sum of marginal variances of parameters.
        Good for overall uncertainty reduction.
        
        Parameters
        ----------
        Sigma_post : np.ndarray
            Posterior covariance matrix, shape (n, n)
            
        Returns
        -------
        float
            Trace of posterior covariance: sum of parameter variances
            
        Notes
        -----
        Smaller values are better. Used for general parameter reconstruction.
        
        Examples
        --------
        >>> a_opt = DesignCriteria.A_opt(Sigma_post)
        """
        return np.trace(Sigma_post)

    @staticmethod
    def D_opt(Sigma_post: np.ndarray) -> float:
        """D-optimality (log-determinant criterion).
        
        Minimizes the volume of the uncertainty ellipsoid (geometric mean).
        Good when all parameter combinations are important.
        
        Parameters
        ----------
        Sigma_post : np.ndarray
            Posterior covariance matrix, shape (n, n),  must be positive definite
            
        Returns
        -------
        float
            Log-determinant of posterior covariance (in natural log units)
            
        Notes
        -----
        Uses slogdet for numerical stability with large matrices.
        Smaller values are better.
        
        Examples
        --------
        >>> d_opt = DesignCriteria.D_opt(Sigma_post)
        >>> # For information in bits: d_opt / np.log(2)
        """
        sign, logdet = la.slogdet(Sigma_post)
        return logdet

    @staticmethod
    def C_opt(Sigma_post: np.ndarray, L: np.ndarray) -> float:
        """C-optimality (linear functional criterion).
        
        Minimizes the variance of a linear functional q = L @ u
        of the parameters. Useful when interested in specific predictions.
        
        Parameters
        ----------
        Sigma_post : np.ndarray
            Posterior covariance matrix, shape (n, n)
        L : np.ndarray
            Observation matrix for the quantity of interest, shape (m, n)
            where m is the number of QoIs and n is the number of parameters
            
        Returns
        -------
        float
            Variance of the linear functional: trace(L @ Sigma @ L^T)
            
        Notes
        -----
        Smaller values are better. Useful for prediction-focused designs.
        
        Examples
        --------
        >>> # Observe mean of first 5 parameters
        >>> L = np.ones((1, 10)) / 5
        >>> c_opt = DesignCriteria.C_opt(Sigma_post, L)
        """
        L = np.atleast_2d(L)
        LSigL = np.dot(L, np.dot(Sigma_post, L.T))
        return float(np.trace(LSigL))
        
    @staticmethod
    def EIG(Sigma_post: np.ndarray, Sigma_prior: np.ndarray) -> float:
        """Expected Information Gain (EIG).
        
        Measures reduction in parameter uncertainty due to observations.
        Based on Kullback-Leibler divergence from prior to posterior.
        
        Parameters
        ----------
        Sigma_post : np.ndarray
            Posterior covariance matrix, shape (n, n)
        Sigma_prior : np.ndarray
            Prior covariance matrix, shape (n, n)
            
        Returns
        -------
        float
            Expected information gain in natural logarithm (nats).
            Divide by np.log(2) for information in bits.
            
        Notes
        -----
        Formula: EIG = 0.5 * (log|Sigma_prior| - log|Sigma_post|)
        
        Always non-negative (clamped to 0 for numerical safety).
        Larger values indicate more informative design.
        
        Examples
        --------
        >>> eig_nats = DesignCriteria.EIG(Sigma_post, Sigma_prior)
        >>> eig_bits = eig_nats / np.log(2)
        """
        return DesignCriteria.eig_from_parameter_cov(Sigma_post, Sigma_prior)

    @staticmethod
    def eig_from_parameter_cov(
        Sigma_post: np.ndarray,
        Sigma_prior: np.ndarray,
    ) -> float:
        """Expected information gain from parameter covariances.

        Formula
        -------
        ``EIG = 0.5 * (log|Sigma_prior| - log|Sigma_post|)``
        """
        _, logdet_prior = la.slogdet(Sigma_prior)
        _, logdet_post = la.slogdet(Sigma_post)

        eig = 0.5 * (logdet_prior - logdet_post)
        return float(max(0.0, eig))

    @staticmethod
    def eig_from_observation_cov(
        Sigma_y: np.ndarray,
        Sigma_noise: np.ndarray,
        *,
        obs_map=None,
        W: np.ndarray | None = None,
    ) -> float:
        """Analytical EIG in the linear-Gaussian case via observation covariances.

        Parameters
        ----------
        Sigma_y : np.ndarray
            Marginal observation covariance ``Sigma_y = Sigma_noise + A Sigma_prior A.T``.
        Sigma_noise : np.ndarray
            Observation noise covariance.
        obs_map : optional
            Unified observation map (preferred): indices, mask, projection, or
            an object exposing ``as_projection()``.
        W : np.ndarray, optional
            Legacy compression matrix argument (kept for backward compatibility).
        """
        Sigma_y = np.asarray(Sigma_y, dtype=float)
        Sigma_noise = np.asarray(Sigma_noise, dtype=float)
        if Sigma_y.ndim != 2 or Sigma_y.shape[0] != Sigma_y.shape[1]:
            raise ValueError("Sigma_y must be a square matrix.")
        if Sigma_noise.ndim != 2 or Sigma_noise.shape != Sigma_y.shape:
            raise ValueError("Sigma_noise must be a square matrix with the same shape as Sigma_y.")

        U = DesignCriteria._projection_from_observation_input(
            Sigma_y.shape[0],
            obs_map=obs_map,
            W=W,
        )
        return DesignCriteria._eig_from_observation_cov_core(Sigma_y, Sigma_noise, U)
    
    @staticmethod
    def EIG_linear_gaussian_obs(
        Sigma_y: np.ndarray,
        Sigma_noise: np.ndarray,
        W: np.ndarray | None = None,
    ) -> float:
        """Legacy alias for observation-covariance EIG (linear-Gaussian case).

        Notes
        -----
        ``W`` is a legacy name for a compression matrix (projection/selection).
        Prefer ``eig_from_observation_cov(..., obs_map=...)``.
        """
        return DesignCriteria.eig_from_observation_cov(
            Sigma_y,
            Sigma_noise,
            W=W,
        )
