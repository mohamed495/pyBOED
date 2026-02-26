"""
GRADIENT-BASED DIMENSION REDUCTION METHODS
===================================================
Full implementation of:
- AS (Active Subspaces)
- LIS (Likelihood-Informed Subspaces)

These methods use gradients to identify directions
that matter most in parameter space.

Reference: Chapter 2.2 of the thesis
"""

import numpy as np
import matplotlib.pyplot as plt
from boed.reduction.inference.active_subspace import ActiveSubspaces
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]
from plotly.subplots import make_subplots # pyright: ignore[reportMissingImports]
from scipy import linalg
from scipy.stats import multivariate_normal
from typing import Tuple, Optional, Callable
import warnings



# ============================================================================
# PARTIE 2.2 : LIKELIHOOD-INFORMED SUBSPACES (LIS)
# ============================================================================

class LikelihoodInformedSubspaces:
    """
    Likelihood-Informed Subspaces (LIS) Method.
    
    THEORETICAL BACKGROUND (Section 2.2.2 of the thesis):
    --------------------------------------------------
    Context: Bayesian inference problem
    
    Data: Y with likelihood π_{Y|X}(y|x)
    Prior : X ~ π_X
    Posterior : π_{X|Y}(x|y) ∝ π_{Y|X}(y|x) π_X(x)
    
    GOAL:
    ----------
    Approximate the posterior by:
    
    π̃_{X|Y}(x|y) ∝ L(P_r x | y) π_X(x)
    
    where L is a ridge approximation of the likelihood.
    
    MATRICE DE DIAGNOSTIC (eq 2.54) :
    ----------------------------------
    H(y) = ∫ ∇ ln π_{y|X}(x) [∇ ln π_{y|X}(x)]^T dπ_{X|Y}(x|y)
    
    DIFFERENCES VS AS:
    ---------------------
    1. AS  : integration over PRIOR π_X
       LIS : integration over POSTERIOR π_{X|Y}
    
    2. AS  : gradients of G
       LIS : gradients of the log-likelihood
    
    3. AS  : general eigenproblem
       LIS : generalized problem (H(y), Σ_pr^{-1})
    
    GENERALIZED EIGENPROBLEM (eq. 2.56):
    ----------------------------------------------------
    H(y) ũ_i = λ_i Σ_{pr}^{-1} ũ_i
    
    OBLIQUE PROJECTOR (eq. 2.56):
    -------------------------------
    P_r^y = Ũ_r Ũ_r^T Σ_{pr}^{-1}
    
    où Ũ_r = [ũ_1, ..., ũ_r]
    
    INTERPRETATION (Section 2.2.2.4):
    -----------------------------------
    ũ_i : directions where the data are MOST INFORMATIVE
          RELATIVE TO THE PRIOR
    
    Rayleigh quotient (eq 2.61) :
    R(ũ) = <ũ, H ũ> / <ũ, Σ_{pr}^{-1} ũ>
    
    If λ_i > 1: data constrain more than the prior
    If λ_i < 1: prior dominates
    
    BORNE D'ERREUR (Theorem 2.11) :
    --------------------------------
    D_{KL}(π_{X|y} || π̃_{X|y}) ≤ (1/2) Σ_{i=r+1}^d λ_i
    """
    
    def __init__(self, n_components: Optional[int] = None,
                 kl_threshold: float = 0.01):
        """
        PARAMETERS:
        ------------
        n_components : informed subspace dimension r
        kl_threshold : KL-divergence threshold for automatic rank selection
        
        NOTE ON THE THRESHOLD:
        -------------------
        D_{KL} = 0.01 : very good approximation
        D_{KL} = 0.1  : acceptable approximation
        D_{KL} = 1.0  : coarse approximation
        """
        self.n_components = n_components
        self.kl_threshold = kl_threshold
        
        # Results
        self.informed_directions_ = None     # Ũ_r
        self.uninformed_directions_ = None   # Ũ_⊥
        self.eigenvalues_ = None             # λ_i
        self.diagnostic_matrix_ = None       # H(y)
        self.prior_precision_ = None         # Σ_{pr}^{-1}
        
    def fit(self, log_likelihood_gradients: np.ndarray,
            prior_covariance: np.ndarray,
            weights: Optional[np.ndarray] = None,
            data_observed: Optional[np.ndarray] = None):
        """
        Construct the likelihood-informed subspace.
        
        PARAMETERS:
        ------------
        log_likelihood_gradients : (N, d) gradients of ln π_{y|X}
                                   Sampled from the POSTERIOR π_{X|Y}
        prior_covariance : (d, d) prior covariance matrix Σ_{pr}
        weights : (N,) optional importance weights
                  Default: uniform weights
        data_observed : observed data (informational only)
        
        CRITICAL WARNING:
        -----------------------
        Gradients must be sampled from the POSTERIOR,
        not the prior!
        
        In practice:
        1. Use a Laplace approximation of the posterior
        2. Or MCMC with burn-in
        3. Or an iterative approximation (see Section 2.2.2.3)
        
        ALGORITHM:
        ------------
        1. Compute H(y) = Σ_i w_i ∇ ln π(x^i) [∇ ln π(x^i)]^T
        2. Compute Σ_{pr}^{-1}
        3. Solve H ũ = λ Σ_{pr}^{-1} ũ (generalized problem)
        4. Select the r dominant eigenvalues
        
        TRANSFORMATION TO A STANDARD PROBLEM (Remark 2.8):
        ---------------------------------------------------
        H ũ = λ Σ_{pr}^{-1} ũ
        ⟺ Σ_{pr}^{1/2} H Σ_{pr}^{1/2} u = λ u
        
        with ũ = Σ_{pr}^{1/2} u
        """
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: DATA VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        
        log_likelihood_gradients = np.asarray(log_likelihood_gradients)
        prior_covariance = np.asarray(prior_covariance)
        
        if log_likelihood_gradients.ndim != 2:
            raise ValueError(f"log_likelihood_gradients must have shape (N,d), got {log_likelihood_gradients.shape}")
        
        N, d = log_likelihood_gradients.shape
        
        if prior_covariance.shape != (d, d):
            raise ValueError(f"prior_covariance must have shape (d,d), got {prior_covariance.shape}")
        
        print(f"LIS: {N} gradients of dimension {d}")
        
        # Check prior covariance
        if not np.allclose(prior_covariance, prior_covariance.T):
            warnings.warn("Prior covariance is not symmetric; symmetrizing")
            prior_covariance = (prior_covariance + prior_covariance.T) / 2
        
        eigvals_check = np.linalg.eigvalsh(prior_covariance)
        min_eig = float(eigvals_check.min())
        if min_eig < -1e-10:
            raise ValueError(
                f"Prior covariance is not positive semidefinite! Min eigenvalue = {min_eig:.2e}"
            )
        if min_eig <= 0:
            jitter = abs(min_eig) + 1e-10
            prior_covariance = prior_covariance + jitter * np.eye(d)
        
        # Weights
        if weights is None:
            weights = np.ones(N) / N
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: BUILD DIAGNOSTIC MATRIX H(y)
        # ─────────────────────────────────────────────────────────────────────
        
        print("Building diagnostic matrix H(y)...")
        
        H = np.zeros((d, d))
        
        for i in range(N):
            g = log_likelihood_gradients[i]  # ∇ ln π_{y|X}(x^i)
            H += weights[i] * np.outer(g, g)
        
        self.diagnostic_matrix_ = H
        
        # Checks
        if not np.allclose(H, H.T):
            warnings.warn("Matrix H is not symmetric; symmetrizing")
            H = (H + H.T) / 2
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: PRECOMPUTATIONS FOR THE GENERALIZED PROBLEM
        # ─────────────────────────────────────────────────────────────────────
        
        print("Solving the generalized eigenproblem...")
        
        # Compute Σ_{pr}^{-1}
        # Use Cholesky for numerical stability
        try:
            L_pr = np.linalg.cholesky(prior_covariance)  # Σ_{pr} = L L^T
        except np.linalg.LinAlgError:
            raise ValueError("Prior covariance is not positive definite (Cholesky failed)")
        
        self.prior_precision_ = np.linalg.inv(prior_covariance)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: TRANSFORM TO A STANDARD PROBLEM
        # ─────────────────────────────────────────────────────────────────────
        
        # METHOD 1: direct generalized problem
        # scipy.linalg.eigh(H, Σ_{pr}^{-1})
        # 
        # METHOD 2: transformation (more numerically stable)
        # H ũ = λ Σ_{pr}^{-1} ũ
        # ⟺ Σ_{pr}^{1/2} H Σ_{pr}^{1/2} u = λ u
        # with ũ = Σ_{pr}^{1/2} u
        
        # Use Method 2 (Remark 2.8)
        
        # Σ_{pr}^{1/2} via Cholesky
        Sigma_sqrt = L_pr  # Since Σ = L L^T, L acts as Σ^{1/2} (up to sign)
        
        # Preconditioned matrix
        H_precond = Sigma_sqrt @ H @ Sigma_sqrt.T
        
        # Standard problem
        eigenvalues, eigenvectors_u = np.linalg.eigh(H_precond)
        
        # Descending sort
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors_u = eigenvectors_u[:, idx]
        
        # Back to generalized eigenvectors
        # ũ = Σ_{pr}^{1/2} u
        eigenvectors_tilde = Sigma_sqrt.T @ eigenvectors_u
        
        # Check: Ũ^T Σ_{pr}^{-1} Ũ = I
        check_orthogonality = eigenvectors_tilde.T @ self.prior_precision_ @ eigenvectors_tilde
        ortho_error = np.linalg.norm(check_orthogonality - np.eye(d), 'fro')
        if ortho_error > 1e-8:
            warnings.warn(f"Sigma_pr^-1 orthogonality error: {ortho_error:.2e}")
        
        self.eigenvalues_ = eigenvalues
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: RANK SELECTION
        # ─────────────────────────────────────────────────────────────────────
        
        # KL error bound (eq. 2.57)
        cumsum_error = np.cumsum(eigenvalues[::-1])[::-1] / 2  # (1/2) Σ_{i>r} λ_i
        
        if self.n_components is None:
            # Automatic selection
            self.n_components = np.searchsorted(cumsum_error[::-1], self.kl_threshold)
            self.n_components = max(1, min(self.n_components, d - 1))
            
            print(f"Automatic selection: {self.n_components} components for "
                  f"D_KL ≤ {self.kl_threshold:.4f}")
        
        r = self.n_components
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: EXTRACT SUBSPACES
        # ─────────────────────────────────────────────────────────────────────
        
        self.informed_directions_ = eigenvectors_tilde[:, :r]  # Ũ_r
        
        if r < d:
            self.uninformed_directions_ = eigenvectors_tilde[:, r:]  # Ũ_⊥
        else:
            self.uninformed_directions_ = None
        
        # ─────────────────────────────────────────────────────────────────────
        # SUMMARY
        # ─────────────────────────────────────────────────────────────────────
        
        print(f"\n✓ Likelihood-Informed Subspace construit :")
        print(f"   Total dimension         : {d}")
        print(f"   Informed dimension      : {r}")
        print(f"   Uninformed dimension    : {d - r}")
        print(f"   D_KL bound              : {cumsum_error[r-1]:.4e}")
        
        print(f"\n   Premiers eigenvalues (λ_i > 1 = data-dominated) :")
        for i in range(min(10, len(eigenvalues))):
            status = "DATA" if eigenvalues[i] > 1 else "PRIOR"
            print(f"   λ_{i+1} = {eigenvalues[i]:.4f} ({status})")
        
        # Count data-dominated directions
        n_data_dominated = np.sum(eigenvalues > 1)
        print(f"\n   Directions data-dominated (λ > 1) : {n_data_dominated}/{d}")
        
        return self
    
    def get_projection_operator(self) -> np.ndarray:
        """
        Return the oblique projector P_r (eq. 2.56).
        
        FORMULA:
        ---------
        P_r = Ũ_r Ũ_r^T Σ_{pr}^{-1}
        
        PROPERTIES:
        ------------
        - NON orthogonal au sens Euclidien
        - Σ_{pr}^{-1}-orthogonal : <P_r x, (I - P_r)x>_{Σ_{pr}^{-1}} = 0
        
        GEOMETRIC INTERPRETATION:
        -----------------------------
        Projection in the Σ_{pr}^{-1} metric, which weights the space
        by prior uncertainty.
        
        RETURNS:
        --------
        P_r : (d, d) projection matrix
        """
        P_r = self.informed_directions_ @ self.informed_directions_.T @ self.prior_precision_
        
        return P_r
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projection sur le sous-espace informé.
        
        ÉQUATION :
        ----------
        X_informed = Ũ_r^T Σ_{pr}^{-1} X
        
        DIFFÉRENCE AVEC AS :
        --------------------
        AS  : X_active = U_r^T X  (projection orthogonale)
        LIS : X_informed = Ũ_r^T Σ_{pr}^{-1} X  (projection oblique)
        
        La multiplication par Σ_{pr}^{-1} "blanchit" X selon le prior.
        
        PARAMETERS:
        ------------
        X : (N, d) échantillons
        
        RETURNS:
        --------
        X_informed : (N, r) coordonnées dans sous-espace informé
        """
        X = np.asarray(X)
        
        if X.ndim == 1:
            return self.informed_directions_.T @ self.prior_precision_ @ X
        else:
            return X @ self.prior_precision_ @ self.informed_directions_

    @property
    def basis(self):
        """Compatibility alias for informed directions."""
        return self.informed_directions_
    
    def sample_reduced_posterior(self, y_observed: np.ndarray,
                                 likelihood_func: Callable,
                                 prior_mean: np.ndarray,
                                 n_samples: int = 1000,
                                 method: str = 'metropolis') -> np.ndarray:
        """
        Échantillonnage du posterior réduit (Algorithm 2 de la thèse).
        
        POSTERIOR RÉDUIT (eq 2.49) :
        ----------------------------
        π̃_{X_r|Y}(x_r|y) ∝ L(x_r|y) π_{X_r}(x_r)
        
        où x_r = P_r x ∈ Im(Ũ_r)
        
        FACTORISATION (eq 2.48) :
        -------------------------
        π̃_{X|Y} = π̃_{X_r|Y} × π_{X_⊥|X_r}
        
        ALGORITHME (simplifié) :
        ------------------------
        1. Échantillonner x_r ~ π̃_{X_r|Y} via MCMC
        2. Pour chaque x_r, échantillonner x_⊥ ~ π_{X_⊥|X_r} (prior conditionnel)
        3. Reconstruire x = x_r + x_⊥
        
        PARAMETERS:
        ------------
        y_observed : données observées
        likelihood_func : fonction π_{Y|X}(y|x)
        prior_mean : moyenne du prior
        n_samples : nombre d'échantillons
        method : 'metropolis' ou 'hmc'
        
        RETURNS:
        --------
        samples : (n_samples, d) échantillons du posterior réduit
        
        NOTE :
        ------
        Implémentation simplifiée ici. Pour production, utiliser
        des bibliothèques MCMC professionnelles (PyMC, Stan, etc.)
        """
        
        d = len(prior_mean)
        r = self.n_components
        
        print(f"Sampling reduced posterior ({r}D instead of {d}D)...")
        
        # Supposons prior Gaussien X ~ N(μ, Σ)
        # Alors la conditionnelle X_⊥ | X_r est aussi Gaussienne
        
        # Décomposition de Σ_{pr}
        # Σ_{pr} = [Σ_rr  Σ_r⊥]
        #          [Σ_⊥r  Σ_⊥⊥]
        
        # Pour simplification, on suppose Σ_{pr} = I (prior standard)
        # Dans ce cas, X_⊥ | X_r ~ N(0, I_{d-r}) indépendant
        
        samples = np.zeros((n_samples, d))
        
        # État initial
        x_current = prior_mean.copy()
        x_r_current = self.transform(x_current)
        
        # Metropolis-Hastings simplifié
        proposal_std = 0.1
        n_accepted = 0
        
        for i in range(n_samples):
            # Proposition pour x_r
            x_r_proposal = x_r_current + np.random.randn(r) * proposal_std
            
            # Reconstruction dans espace complet
            # x = Ũ_r x_r (approximation, projection inverse)
            # Version correcte nécessite la pseudo-inverse
            x_proposal_r = self.informed_directions_ @ x_r_proposal
            
            # Composante non-informée (échantillonnée du prior)
            if self.uninformed_directions_ is not None:
                x_proposal_perp = self.uninformed_directions_ @ np.random.randn(d - r)
            else:
                x_proposal_perp = 0
            
            x_proposal = x_proposal_r + x_proposal_perp
            
            # Ratio d'acceptation (log-scale pour stabilité)
            log_like_current = np.log(likelihood_func(y_observed, x_current) + 1e-300)
            log_like_proposal = np.log(likelihood_func(y_observed, x_proposal) + 1e-300)
            
            log_ratio = log_like_proposal - log_like_current
            
            # Acceptation
            if np.log(np.random.rand()) < log_ratio:
                x_current = x_proposal
                x_r_current = x_r_proposal
                n_accepted += 1
            
            samples[i] = x_current
        
        acceptance_rate = n_accepted / n_samples
        print(f"   Acceptance rate: {acceptance_rate*100:.1f}%")
        
        return samples
    
    def estimate_kl_error(self) -> float:
        """
        Estimation de la borne d'erreur D_KL (eq 2.57).
        
        FORMULA:
        ---------
        D_{KL}(π_{X|y} || π̃_{X|y}) ≤ (1/2) Σ_{i=r+1}^d λ_i
        
        RETURNS:
        --------
        kl_bound : borne supérieure de la divergence KL
        """
        r = self.n_components
        kl_bound = self.eigenvalues_[r:].sum() / 2
        
        return kl_bound


# ============================================================================
# SECTION 2.2.1 : DATA-AVERAGED LIS (Section 2.2.2.5)
# ============================================================================

class DataAveragedLIS:
    """
    Data-Averaged Likelihood-Informed Subspaces.
    
    MOTIVATION (Section 2.2.2.5) :
    -------------------------------
    LIS standard dépend de y (données observées)
    
    H(y) = ∫ ∇ ln π_{y|X}(x) [∇ ln π_{y|X}(x)]^T dπ_{X|Y}(x|y)
    
    Problème : Si on doit résoudre le problème inverse pour
    différentes réalisations de y (ex: Bayesian OED), il faut
    recalculer H(y) à chaque fois.
    
    SOLUTION :
    ----------
    Moyenner H(y) sur la distribution des données :
    
    H̄ = 𝔼_Y[H(Y)] = ∫ I(x) dπ_X(x)  (eq 2.63)
    
    où I(x) est la matrice d'information de Fisher (eq 2.64) :
    
    I(x) = ∫ ∇ ln π_{y|X}(x) [∇ ln π_{y|X}(x)]^T π_{y|X}(x) dy
    
    CAS GAUSSIEN (très important !) :
    ----------------------------------
    Si π_{Y|X}(y|x) = N(G(x), Σ_{obs}), alors :
    
    I(x) = ∇G(x)^T Σ_{obs}^{-1} ∇G(x)
    
    et donc :
    
    H̄ = ∫ ∇G(x)^T Σ_{obs}^{-1} ∇G(x) dπ_X(x)  (eq 2.67)
    
    AVANTAGE :
    ----------
    H̄ peut être calculé AVANT d'observer y, en intégrant sur le prior !
    """
    
    def __init__(self, n_components: Optional[int] = None,
                 kl_threshold: float = 0.01):
        self.n_components = n_components
        self.kl_threshold = kl_threshold
        
        self.informed_directions_ = None
        self.eigenvalues_ = None
        self.diagnostic_matrix_ = None
        self.prior_precision_ = None
        
    def fit(self, forward_gradients: np.ndarray,
            prior_covariance: np.ndarray,
            observation_covariance: np.ndarray,
            weights: Optional[np.ndarray] = None):
        """
        Construction du sous-espace data-averaged.
        
        PARAMETERS:
        ------------
        forward_gradients : (N, d, m) Jacobiennes ∇G(x^i)
                            Échantillonnées depuis le PRIOR π_X
        prior_covariance : (d, d) Σ_{pr}
        observation_covariance : (m, m) Σ_{obs}
        weights : (N,) poids Monte Carlo (optionnel)
        
        CAS SCALAIRE (m=1) :
        --------------------
        forward_gradients : (N, d) gradients ∇G(x^i)
        observation_covariance : scalaire σ²_{obs}
        
        ALGORITHM:
        ------------
        1. Calculer I(x^i) = ∇G(x^i)^T Σ_{obs}^{-1} ∇G(x^i)
        2. Moyenner : H̄ = Σ_i w_i I(x^i)
        3. Résoudre (H̄, Σ_{pr}^{-1}) eigenproblem
        """
        
        forward_gradients = np.asarray(forward_gradients)
        
        # ─────────────────────────────────────────────────────────────────────
        # GESTION CAS SCALAIRE vs VECTORIEL
        # ─────────────────────────────────────────────────────────────────────
        
        if forward_gradients.ndim == 2:
            # Cas scalaire G : ℝᵈ → ℝ
            N, d = forward_gradients.shape
            m = 1
            print(f"Data-Averaged LIS (scalar): {N} gradients of dimension {d}")
            
            # Observation covariance doit être scalaire
            if np.isscalar(observation_covariance):
                obs_precision = 1.0 / observation_covariance
            elif observation_covariance.shape == (1, 1):
                obs_precision = 1.0 / observation_covariance[0, 0]
            else:
                raise ValueError("observation_covariance must be scalar for m=1")
            
        elif forward_gradients.ndim == 3:
            # Cas vectoriel G : ℝᵈ → ℝᵐ
            N, d, m = forward_gradients.shape
            print(f"Data-Averaged LIS (vector): {N} Jacobians {d}x{m}")
            
            observation_covariance = np.asarray(observation_covariance)
            if observation_covariance.shape != (m, m):
                raise ValueError(f"observation_covariance must be ({m},{m})")
            
            # Calcul de Σ_{obs}^{-1}
            obs_precision = np.linalg.inv(observation_covariance)
            
        else:
            raise ValueError(f"forward_gradients must have shape (N,d) or (N,d,m)")
        
        # Weights
        if weights is None:
            weights = np.ones(N) / N
        else:
            weights = np.asarray(weights) / np.asarray(weights).sum()
        
        # ─────────────────────────────────────────────────────────────────────
        # CONSTRUCTION DE H̄ = 𝔼[I(X)]
        # ─────────────────────────────────────────────────────────────────────
        
        print("Building data-averaged matrix H̄...")
        
        H_bar = np.zeros((d, d))
        
        if m == 1:
            # Cas scalaire : I(x) = σ⁻² ∇G(x) ∇G(x)^T
            for i in range(N):
                grad = forward_gradients[i]  # ∇G(x^i) ∈ ℝᵈ
                I_i = obs_precision * np.outer(grad, grad)
                H_bar += weights[i] * I_i
        else:
            # Cas vectoriel : I(x) = J(x)^T Σ_{obs}^{-1} J(x)
            for i in range(N):
                J = forward_gradients[i]  # Jacobienne ∈ ℝᵈˣᵐ
                I_i = J @ obs_precision @ J.T
                H_bar += weights[i] * I_i
        
        self.diagnostic_matrix_ = H_bar
        
        # ─────────────────────────────────────────────────────────────────────
        # RÉSOLUTION PROBLÈME GÉNÉRALISÉ (identique à LIS standard)
        # ─────────────────────────────────────────────────────────────────────
        
        print("Solving generalized problem...")
        
        prior_covariance = np.asarray(prior_covariance)
        
        # Cholesky
        L_pr = np.linalg.cholesky(prior_covariance)
        self.prior_precision_ = np.linalg.inv(prior_covariance)
        
        # Transformation
        H_precond = L_pr @ H_bar @ L_pr.T
        
        # Eigendecomposition
        eigenvalues, eigenvectors_u = np.linalg.eigh(H_precond)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors_u = eigenvectors_u[:, idx]
        
        eigenvectors_tilde = L_pr.T @ eigenvectors_u
        
        self.eigenvalues_ = eigenvalues
        
        # ─────────────────────────────────────────────────────────────────────
        # SÉLECTION DU RANG
        # ─────────────────────────────────────────────────────────────────────
        
        cumsum_error = np.cumsum(eigenvalues[::-1])[::-1] / 2
        
        if self.n_components is None:
            self.n_components = np.searchsorted(cumsum_error[::-1], self.kl_threshold)
            self.n_components = max(1, min(self.n_components, d - 1))
            
            print(f"Automatic selection: {self.n_components} components")
        
        r = self.n_components
        
        self.informed_directions_ = eigenvectors_tilde[:, :r]
        
        print(f"\n✓ Data-Averaged LIS construit :")
        print(f"   Informed dimension : {r}")
        print(f"   Borne 𝔼[D_KL]      : {cumsum_error[r-1]:.4e}")
        
        return self


# ============================================================================
# SECTION 2.2.2 : COMPARAISON AS vs LIS (Section 2.2.2.6)
# ============================================================================

def compare_as_lis_methods(forward_model: Callable,
                           prior_mean: np.ndarray,
                           prior_cov: np.ndarray,
                           obs_cov: np.ndarray,
                           n_samples: int = 500,
                           y_observed: Optional[np.ndarray] = None):
    """
    Comparaison AS, LIS, et Data-Averaged LIS.
    
    TROIS DIAGNOSTIQUES (Section 2.2.2.6) :
    ----------------------------------------
    1. AS sur forward model G :
       H_AS^G = ∫ ∇G(x) ∇G(x)^T dπ_X(x)
    
    2. AS sur negative log-likelihood :
       H_AS^ℓ = ∫ ∇ ln π_{y|X}(x) [∇ ln π_{y|X}(x)]^T dπ_X(x)
    
    3. Data-Averaged LIS (= AS sur G avec Σ_{obs}) :
       H̄ = ∫ ∇G(x)^T Σ_{obs}^{-1} ∇G(x) dπ_X(x)
    
    GOAL:
    ----------
    Montrer que Data-Averaged LIS pondère AS par l'incertitude
    d'observation.
    
    PARAMETERS:
    ------------
    forward_model : fonction G(x) retournant (G, ∇G)
    prior_mean, prior_cov : paramètres du prior Gaussien
    obs_cov : covariance d'observation
    n_samples : nombre d'échantillons
    y_observed : données (optionnel, pour LIS standard)
    """
    
    d = len(prior_mean)
    
    print("="*70)
    print("COMPARAISON AS vs LIS")
    print("="*70)
    
    # Échantillonnage du prior
    print(f"\nSampling {n_samples} prior points...")
    X_prior = np.random.multivariate_normal(prior_mean, prior_cov, n_samples)
    
    # Évaluation forward model
    print("Evaluating forward model...")
    G_vals = []
    G_grads = []
    
    for i in range(n_samples):
        g_val, g_grad = forward_model(X_prior[i])
        G_vals.append(g_val)
        G_grads.append(g_grad)
    
    G_vals = np.array(G_vals)
    G_grads = np.array(G_grads)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MÉTHODE 1 : AS SUR FORWARD MODEL
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-"*70)
    print("1. Active Subspace sur forward model G")
    print("-"*70)
    
    as_forward = ActiveSubspaces(n_components=2)
    as_forward.fit(G_grads)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MÉTHODE 2 : DATA-AVERAGED LIS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-"*70)
    print("2. Data-Averaged LIS")
    print("-"*70)
    
    da_lis = DataAveragedLIS(n_components=2)
    da_lis.fit(G_grads, prior_cov, obs_cov)
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPARAISON
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*70)
    print("COMPARAISON DES EIGENVALUES")
    print("="*70)
    
    print("\n{:<10} {:<15} {:<15}".format("Index", "AS (forward)", "DA-LIS"))
    print("-"*40)
    
    for i in range(min(5, d)):
        print("{:<10} {:<15.6e} {:<15.6e}".format(
            i+1,
            as_forward.eigenvalues_[i],
            da_lis.eigenvalues_[i]
        ))
    
    print("\nINTERPRETATION:")
    print("- AS: directions of large variation of G")
    print("- DA-LIS: high-variation directions weighted by observation precision")
    print("- Si Σ_obs = σ² I, alors DA-LIS ∝ AS")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VISUALISATION OVERLAP DES SOUS-ESPACES
    # ─────────────────────────────────────────────────────────────────────────
    
    overlap = np.abs(as_forward.active_directions_.T @ da_lis.informed_directions_)
    
    print("\nOVERLAP DES SOUS-ESPACES :")
    print(overlap)
    print("(Values close to 1 indicate aligned directions)")
    
    return as_forward, da_lis


# ============================================================================
# SECTION 2.2.3 : DÉMONSTRATION LIS
# ============================================================================

def demo_likelihood_informed_subspaces():
    """
    Démonstration complète LIS sur problème Bayésien.
    """
    print("="*70)
    print("LIKELIHOOD-INFORMED SUBSPACES DEMO")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. SETUP DU PROBLÈME INVERSE
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n1. Defining the inverse problem...")
    
    d = 10  # Dimension paramètre
    m = 5   # Dimension observation
    
    # Forward model linéaire : G(x) = A x
    # (pour simplicité pédagogique)
    np.random.seed(42)
    A = np.random.randn(m, d)
    A[:, :2] *= 5  # Premières directions plus importantes
    
    def forward_model(x):
        """G(x) = A x"""
        G = A @ x
        grad_G = A.T  # Jacobienne (constante pour modèle linéaire)
        return G, grad_G
    
    # Prior Gaussien
    prior_mean = np.zeros(d)
    prior_cov = np.eye(d)
    
    # Observation noise
    obs_noise_std = 0.5
    obs_cov = (obs_noise_std ** 2) * np.eye(m)
    
    # Vraie valeur du paramètre
    x_true = np.random.randn(d)
    x_true[:2] = [3.0, -2.0]  # Valeurs significatives sur directions importantes
    
    # Données observées
    y_true, _ = forward_model(x_true)
    y_observed = y_true + np.random.randn(m) * obs_noise_std
    
    print(f"   Parameter dimension     : {d}")
    print(f"   Dimension observation : {m}")
    print(f"   Forward model: linear G(x) = A x")
    print(f"   Prior : N(0, I_{d})")
    print(f"   Likelihood : N(G(x), {obs_noise_std**2} I_{m})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. CALCUL DU POSTERIOR (analytique pour cas linéaire Gaussien)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n2. Computing analytical posterior...")
    
    # Pour problème linéaire Gaussien :
    # π_{X|Y} = N(μ_{post}, Σ_{post})
    # 
    # Σ_{post} = (A^T Σ_{obs}^{-1} A + Σ_{pr}^{-1})^{-1}
    # μ_{post} = Σ_{post} A^T Σ_{obs}^{-1} y
    
    obs_precision = np.linalg.inv(obs_cov)
    prior_precision = np.linalg.inv(prior_cov)
    
    post_precision = A.T @ obs_precision @ A + prior_precision
    post_cov = np.linalg.inv(post_precision)
    post_mean = post_cov @ A.T @ obs_precision @ y_observed
    
    print(f"   μ_post = {post_mean[:3]}... (first 3)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. ÉCHANTILLONNAGE DU POSTERIOR
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n3. Sampling posterior...")
    
    N_post = 500
    X_post = np.random.multivariate_normal(post_mean, post_cov, N_post)
    
    # Gradients du log-likelihood
    # Pour π(y|x) = N(Ax, Σ_{obs}) :
    # ∇ ln π(y|x) = A^T Σ_{obs}^{-1} (y - Ax)
    
    log_like_grads = np.zeros((N_post, d))
    for i in range(N_post):
        residual = y_observed - A @ X_post[i]
        log_like_grads[i] = A.T @ obs_precision @ residual
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. CONSTRUCTION LIS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n4. Building Likelihood-Informed Subspace...")
    
    lis = LikelihoodInformedSubspaces(n_components=3)
    lis.fit(log_like_grads, prior_cov, data_observed=y_observed)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. CONSTRUCTION DATA-AVERAGED LIS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n5. Building Data-Averaged LIS...")
    
    # Échantillonnage du prior
    N_prior = 500
    X_prior = np.random.multivariate_normal(prior_mean, prior_cov, N_prior)
    
    # Gradients du forward model
    forward_grads = np.tile(A.T, (N_prior, 1, 1))  # (N, d, m)
    
    da_lis = DataAveragedLIS(n_components=3)
    da_lis.fit(forward_grads, prior_cov, obs_cov)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. COMPARAISON
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n6. Comparing methods...")
    
    print("\n" + "="*70)
    print("EIGENVALUES COMPARISON")
    print("="*70)
    
    print("\n{:<10} {:<20} {:<20}".format("Index", "LIS", "DA-LIS"))
    print("-"*50)
    
    for i in range(min(5, d)):
        print("{:<10} {:<20.6f} {:<20.6f}".format(
            i+1,
            lis.eigenvalues_[i],
            da_lis.eigenvalues_[i]
        ))
    
    # Overlap des sous-espaces
    overlap = np.abs(lis.informed_directions_.T @ da_lis.informed_directions_)
    
    print("\nSUBSPACE OVERLAP:")
    print(overlap)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. VALIDATION : ERREUR KL EMPIRIQUE
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n7. Validation (D_KL approximation)...")
    
    # Pour Gaussiennes, D_KL a une formule fermée
    # D_KL(N(μ₁,Σ₁) || N(μ₂,Σ₂)) = 1/2 [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)^T Σ₂⁻¹(μ₂-μ₁) - d + ln(det(Σ₂)/det(Σ₁))]
    
    # Posterior réduit (approximation) : même moyenne, covariance réduite
    # Ici simplifié pour démo
    
    kl_bound_lis = lis.estimate_kl_error()
    kl_bound_da = da_lis.eigenvalues_[da_lis.n_components:].sum() / 2
    
    print(f"\n   Borne D_KL (LIS)     : {kl_bound_lis:.4e}")
    print(f"   Borne D_KL (DA-LIS)  : {kl_bound_da:.4e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nLIS identifies the directions most constrained by the data")
    print("DA-LIS can be computed BEFORE observing data (useful for OED)")
    print("Both methods produce similar subspaces")
    
    return lis, da_lis, X_post


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    # Décommenter pour exécuter les démos
    
    # Active Subspaces
    # as_model, G_func, X, G_vals, true_dirs = demo_active_subspaces()
    
    # Likelihood-Informed Subspaces
    lis, da_lis, X_post = demo_likelihood_informed_subspaces()
