"""
GRADIENT-BASED DIMENSION REDUCTION METHODS
===================================================
Full implementation of:
- AS (Active Subspaces)
- LIS (Likelihood-Informed Subspaces)

These methods use gradients to identify the directions
that matter most in parameter space.

Reference: Chapter 2.2 of the thesis
"""

import numpy as np
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]
from plotly.subplots import make_subplots # pyright: ignore[reportMissingImports]
from scipy import linalg
from scipy.stats import multivariate_normal
from typing import Tuple, Optional, Callable
import warnings

# ============================================================================
# PARTIE 2.1 : ACTIVE SUBSPACES (AS)
# ============================================================================

class ActiveSubspaces:
    """
    Active Subspaces Method.
    
    THEORETICAL BACKGROUND (Section 2.2.1 of the thesis):
    --------------------------------------------------
    Goal: identify the input subspace X ∈ R^d on which
               a function G : R^d -> R^m varies the most.
    
    DIAGNOSTIC MATRIX (eq. 2.29):
    ----------------------------------
    H = 𝔼[∇G(X) ∇G(X)ᵀ] = ∫ ∇G(x) ∇G(x)ᵀ dπ_X(x)
    
    where ∇G(x) ∈ R^(d×m) is the Jacobian matrix.
    
    Pour une fonction scalaire G : ℝᵈ → ℝ :
    H = 𝔼[∇G(X) ∇G(X)ᵀ] ∈ ℝᵈˣᵈ
    
    SPECTRAL DECOMPOSITION:
    -------------------------
    H = U Λ Uᵀ  où Λ = diag(λ₁, ..., λᵈ) avec λ₁ ≥ ... ≥ λᵈ ≥ 0
    
    INTERPRETATION (Proposition 2.2):
    -----------------------------------
    λᵢ = 𝔼[(∇G(X)ᵀ uᵢ)²]
    
    This is the variance of the directional derivative along u_i.
    
    If λ_i ≈ 0, then G varies little along u_i.
    
    ACTIVE SUBSPACE :
    -----------------
    U_r = [u₁, ..., u_r] : les r vecteurs propres dominants
    
    Defines the projection P_r = U_r U_r^T
    
    APPROXIMATION RIDGE (eq 2.33) :
    --------------------------------
    G(x) ≈ G*(x) = 𝔼[G(X) | P_r X = P_r x]
    
    ERROR BOUND (Theorem 2.5):
    -------------------------------
    ||G - G*||²_{L²} ≤ C(X) Σᵢ₌ᵣ₊₁ᵈ λᵢ
    
    where C(X) is the Poincare constant.
    """
    
    def __init__(self, n_components: Optional[int] = None,
                 activity_threshold: float = 0.99,
                 rank: Optional[int] = None):
        """
        PARAMETERS:
        ------------
        n_components : active subspace dimension r (if None, automatic)
        activity_threshold : fraction of activity to capture (default 99%)
        
        ANALOGY WITH PCA:
        -------------------
        PCA seeks directions of maximal variance in X
        AS seeks directions of maximal variance in ∇G(X)
        """
        # Compatibility alias: some external examples use `rank=...`.
        if rank is not None:
            self.n_components = rank
        else:
            self.n_components = n_components
        self.activity_threshold = activity_threshold
        
        # Results (filled after fit)
        self.active_directions_ = None      # U_r in thesis notation
        self.inactive_directions_ = None    # U_⊥ inactive directions
        self.eigenvalues_ = None             # λᵢ
        self.activity_ratio_ = None          # λᵢ / Σλⱼ
        self.diagnostic_matrix_ = None       # H diagnostic matrix
        
        # For prediction
        self.mean_gradient_ = None
        
    def fit(self, gradients: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Build the active subspace from gradient samples.
        
        PARAMETERS:
        ------------
        gradients : (N, d) or (N, d, m) sampled gradients
                    - Cas scalaire (m=1) : gradients[i] = ∇G(xⁱ) ∈ ℝᵈ
                    - Cas vectoriel : gradients[i] = Jacobienne ∈ ℝᵈˣᵐ
        weights : (N,) optional importance weights
                  Default: uniform weights 1/N
        
        ALGORITHME :
        ------------
        1. Calculer H̃ = (1/N) Σᵢ ∇G(xⁱ) ∇G(xⁱ)ᵀ  (estimateur MC)
        2. Spectral decomposition: H̃ = U Λ Uᵀ
        3. Select the r dominant eigenvectors
        
        COMPLEXITY:
        ------------
        Assemblage H̃ : O(N d²)  (ou O(N d² m) si vectoriel)
        Eigendecomp : O(d³)
        Total : O(N d² + d³)
        
        COMMON PITFALLS:
        ---------------------
        1. Do not normalize gradients before computing H
           (we want absolute activity, not relative activity)
        2. Oublier de moyenner par N (biais dans eigenvalues)
        3. Confondre gradient et Jacobienne pour fonctions vectorielles
        """
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: VALIDATION AND PREPROCESSING
        # ─────────────────────────────────────────────────────────────────────
        
        gradients = np.asarray(gradients)
        
        if gradients.ndim == 2:
            # Cas scalaire G : ℝᵈ → ℝ
            # gradients : (N, d)
            N, d = gradients.shape
            m = 1
            print(f"Scalar mode: {N} gradients of dimension {d}")
            
        elif gradients.ndim == 3:
            # Cas vectoriel G : ℝᵈ → ℝᵐ
            # gradients : (N, d, m) where gradients[i] is the Jacobian
            N, d, m = gradients.shape
            print(f"Vector mode: {N} Jacobians {d}x{m}")
            
        else:
            raise ValueError(f"gradients must have shape (N,d) or (N,d,m); received {gradients.shape}")
        
        # Importance weights (for Monte Carlo integration)
        if weights is None:
            weights = np.ones(N) / N
        else:
            weights = np.asarray(weights)
            if weights.shape != (N,):
                raise ValueError(f"weights must have shape ({N},); received {weights.shape}")
            # Normalisation
            weights = weights / weights.sum()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: BUILD DIAGNOSTIC MATRIX H
        # ─────────────────────────────────────────────────────────────────────
        
        print("Building diagnostic matrix H...")
        
        # Initialisation
        H = np.zeros((d, d))
        
        if m == 1:
            # ═════════════════════════════════════════════════════════════════
            # CAS SCALAIRE : H = Σᵢ wᵢ ∇G(xⁱ) ∇G(xⁱ)ᵀ
            # ═════════════════════════════════════════════════════════════════
            
            for i in range(N):
                g = gradients[i]  # ∇G(xⁱ) ∈ ℝᵈ
                H += weights[i] * np.outer(g, g)  # Produit externe
            
            # Vectorized alternative (faster):
            # H = (gradients.T @ np.diag(weights) @ gradients)
            
        else:
            # ═════════════════════════════════════════════════════════════════
            # CAS VECTORIEL : H = Σᵢ wᵢ J(xⁱ) J(xⁱ)ᵀ
            # ═════════════════════════════════════════════════════════════════
            # 
            # where J(xⁱ) ∈ R^(d×m) is the Jacobian
            # 
            # FORMULE :
            # H = Σᵢ wᵢ Σⱼ ∇Gⱼ(xⁱ) ∇Gⱼ(xⁱ)ᵀ
            #   = Σᵢ wᵢ J(xⁱ) J(xⁱ)ᵀ
            
            for i in range(N):
                J = gradients[i]  # Jacobienne ∈ ℝᵈˣᵐ
                H += weights[i] * (J @ J.T)
        
        self.diagnostic_matrix_ = H
        
        # Checks
        if not np.allclose(H, H.T):
            warnings.warn("Matrix H is not symmetric (numerical errors); symmetrizing")
            H = (H + H.T) / 2
        
        eigenvalues_check = np.linalg.eigvalsh(H)
        if np.any(eigenvalues_check < -1e-10):
            warnings.warn(f"Matrix H is not positive semidefinite! Min eigenvalue = {eigenvalues_check.min():.2e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: SPECTRAL DECOMPOSITION
        # ─────────────────────────────────────────────────────────────────────
        
        print("Spectral decomposition of H...")
        
        # np.linalg.eigh: for Hermitian matrices (real symmetric)
        # Plus rapide et plus stable que np.linalg.eig
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Descending sort (eigh returns ascending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Enforce positivity (numerics may create small negative values)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        
        self.eigenvalues_ = eigenvalues
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: COMPUTE ACTIVITY RATIO
        # ─────────────────────────────────────────────────────────────────────
        
        # Total activity
        total_activity = eigenvalues.sum()
        
        if total_activity < 1e-14:
            warnings.warn("Total activity is nearly zero; G is probably constant.")
            self.activity_ratio_ = np.zeros_like(eigenvalues)
        else:
            self.activity_ratio_ = eigenvalues / total_activity
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: RANK SELECTION (active subspace dimension)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.n_components is None:
            # Automatic selection: capture `activity_threshold` of activity
            cumsum_activity = np.cumsum(self.activity_ratio_)
            self.n_components = np.searchsorted(cumsum_activity, self.activity_threshold) + 1
            
            # Cap at d-1 to keep at least one inactive direction
            self.n_components = min(self.n_components, d - 1)
            
            print(f"Automatic selection: {self.n_components} components for "
                  f"{self.activity_threshold*100:.1f}% activity")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: EXTRACT ACTIVE/INACTIVE SUBSPACES
        # ─────────────────────────────────────────────────────────────────────
        
        r = self.n_components
        
        # ACTIVE subspace: high-variation directions
        self.active_directions_ = eigenvectors[:, :r]  # U_r ∈ ℝᵈˣʳ
        
        # INACTIVE subspace: low-variation directions
        if r < d:
            self.inactive_directions_ = eigenvectors[:, r:]  # U_⊥ ∈ ℝᵈˣ⁽ᵈ⁻ʳ⁾
        else:
            self.inactive_directions_ = None
        
        # ─────────────────────────────────────────────────────────────────────
        # SUMMARY
        # ─────────────────────────────────────────────────────────────────────
        
        print(f"\n✓ Active Subspace construit :")
        print(f"   Total dimension       : {d}")
        print(f"   Active dimension      : {r}")
        print(f"   Inactive dimension    : {d - r}")
        print(f"   Captured activity     : {cumsum_activity[r-1]*100:.2f}%")
        print(f"\n   Premiers eigenvalues :")
        for i in range(min(5, len(eigenvalues))):
            print(f"   λ_{i+1} = {eigenvalues[i]:.6e} ({self.activity_ratio_[i]*100:.2f}%)")
        
        if r < d:
            print(f"\n   Rapport λ_r / λ_{r+1} = {eigenvalues[r-1] / (eigenvalues[r] + 1e-14):.2f}")
            print(f"   (large ratio = strong spectral gap = good AS)")
        
        return self

    def fit_from_gradients(
        self,
        samples: np.ndarray,
        gradients: np.ndarray,
        weights: Optional[np.ndarray] = None
    ):
        """
        Compatibility wrapper used by some hybrid examples.

        Parameters
        ----------
        samples : ndarray
            Unused placeholder for API compatibility.
        gradients : ndarray
            Gradient samples passed to `fit`.
        weights : ndarray, optional
            Optional sample weights.
        """
        # `samples` is intentionally unused: AS fit only needs gradients.
        _ = samples
        return self.fit(gradients=gradients, weights=weights)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projection sur le sous-espace actif.
        
        EQUATION:
        ----------
        X_active = U_r^T X ∈ ℝʳ
        
        INTERPRETATION:
        ----------------
        X_active contains the coordinates of X in the active-direction basis.
        C'est la partie "importante" de X pour G.
        
        PARAMETERS:
        ------------
        X : (N, d) samples to project
        
        RETOUR :
        --------
        X_active : (N, r) coordinates in the active subspace
        
        EXEMPLE :
        ---------
        >>> as_model = ActiveSubspaces(n_components=2)
        >>> as_model.fit(gradients)
        >>> X_reduced = as_model.transform(X)  # (N, d) → (N, 2)
        """
        X = np.asarray(X)
        
        if X.ndim == 1:
            # Single sample : (d,) → (r,)
            return self.active_directions_.T @ X
        else:
            # Multiple samples : (N, d) → (N, r)
            return X @ self.active_directions_

    @property
    def basis(self):
        """Compatibility alias for active basis directions."""
        return self.active_directions_

    @property
    def eigenvalues(self):
        """Compatibility alias for eigenvalues_."""
        return self.eigenvalues_
    
    def forward_map(self, X: np.ndarray) -> np.ndarray:
        """
        Full orthogonal projection onto the active subspace.
        
        EQUATION:
        ----------
        X_proj = P_r X = U_r U_r^T X ∈ ℝᵈ
        
        DIFFERENCE FROM `transform()`:
        -----------------------------
        - transform() : returns coordinates in the reduced basis ∈ R^r
        - forward_map() : retourne projection dans espace original ∈ ℝᵈ
        
        UTILISATION :
        -------------
        Use this to evaluate G(X_proj) with the original model (no surrogate).
        
        PARAMETERS:
        ------------
        X : (N, d) échantillons
        
        RETOUR :
        --------
        X_proj : (N, d) projections sur sous-espace actif
        """
        X = np.asarray(X)
        
        if X.ndim == 1:
            return self.active_directions_ @ (self.active_directions_.T @ X)
        else:
            return X @ self.active_directions_ @ self.active_directions_.T
    
    def compute_sufficient_summary(self, X: np.ndarray, 
                                   conditional: bool = False) -> np.ndarray:
        """
        Calcul du "sufficient summary" : la partie de X qui contient
        toute l'information pertinente pour G.
        
        EQUATION (unconditional):
        --------------------------------
        X_summary = U_r^T X ∈ ℝʳ
        
        EQUATION (conditional, eq. 2.38):
        -------------------------------------
        To evaluate G*(x) = E[G(X) | U_r^T X = U_r^T x], we need
        to sample X_⊥ ~ π_{X_⊥ | X_r}
        
        PARAMETERS:
        ------------
        X : (N, d) échantillons
        conditional : si True, retourne aussi composante inactive
        
        RETOUR :
        --------
        Si conditional=False : X_active (N, r)
        Si conditional=True : (X_active, X_inactive) tuple de (N,r) et (N,d-r)
        """
        X_active = self.transform(X)
        
        if conditional and self.inactive_directions_ is not None:
            X_inactive = X @ self.inactive_directions_
            return X_active, X_inactive
        else:
            return X_active
    
    def estimate_error_bound(self, poincare_constant: float = 1.0) -> float:
        """
        Estimate the L² error bound (Theorem 2.5).
        
        THEORETICAL BOUND:
        -----------------
        ||G - G*||²_{L²} ≤ C(X) Σᵢ₌ᵣ₊₁ᵈ λᵢ
        
        where C(X) is the Poincare constant.
        
        FOR GAUSSIAN X (Example 3.10):
        ----------------------------------------------
        Si X ~ N(0, I_d), alors C(X) = 1
        Si X ~ N(0, Σ), alors C(X) = λ_max(Σ)
        
        PARAMETERS:
        ------------
        poincare_constant : estimate of the Poincare constant C(X)
                            (default = 1 for standardized X)
        
        RETOUR :
        --------
        error_bound : upper bound for ||G - G*||_{L²}
        """
        r = self.n_components
        
        # Sum of residual eigenvalues
        residual_eigenvalues = self.eigenvalues_[r:].sum()
        
        # Error bound
        error_bound = np.sqrt(poincare_constant * residual_eigenvalues)
        
        return error_bound


# ============================================================================
# SECTION 2.1.1 : TEST FUNCTION GENERATION FOR AS
# ============================================================================

def generate_ridge_function(d: int, r: int, noise_level: float = 0.1,
                           random_state: int = 42) -> Tuple[Callable, np.ndarray]:
    """
    Generate a ridge function to test Active Subspaces.
    
    MODÈLE :
    --------
    G(x) = f(U_r^T x) + ε
    
    where:
    - U_r ∈ ℝᵈˣʳ : directions actives (orthonormales)
    - f : R^r → R : nonlinear profile function
    - ε : bruit Gaussien
    
    PEDAGOGICAL GOAL:
    ----------------------
    Active Subspaces should recover U_r from gradients only.
    
    EXEMPLE CONCRET :
    -----------------
    Pour d=10, r=2 :
    x ∈ ℝ¹⁰ → y = sin(x₁ + 2x₃) + cos(x₂ - x₅) + bruit
    
    Les directions actives sont [1,0,2,0,0,...] et [0,1,0,0,-1,0,...]
    
    PARAMETERS:
    ------------
    d : input dimension
    r : true intrinsic dimension
    noise_level : niveau de bruit
    random_state : random seed
    
    RETOUR :
    --------
    G_func : fonction G(x) et son gradient ∇G(x)
    true_directions : U_r les vraies directions actives
    """
    rng = np.random.RandomState(random_state)
    
    # Generate random orthonormal active directions
    U_r = rng.randn(d, r)
    U_r, _ = np.linalg.qr(U_r)  # Orthonormalisation
    
    # Fonction de profil f : ℝʳ → ℝ
    # On utilise une combinaison de sinus pour avoir une fonction lisse
    def profile_function(z):
        """
        Nonlinear function on the active subspace.
        
        f(z) = Σᵢ sin(ωᵢ zᵢ + φᵢ)
        
        where ωᵢ are frequencies and φᵢ are phases
        """
        result = 0.0
        for i in range(r):
            omega = (i + 1) * np.pi
            phi = rng.rand() * 2 * np.pi
            result += np.sin(omega * z[i] + phi)
        return result
    
    def profile_gradient(z):
        """Gradient of f with respect to z ∈ R^r."""
        grad = np.zeros(r)
        for i in range(r):
            omega = (i + 1) * np.pi
            phi = rng.rand() * 2 * np.pi
            grad[i] = omega * np.cos(omega * z[i] + phi)
        return grad
    
    # Full function G(x) with gradient
    def G_with_gradient(x, return_gradient=True):
        """
        Evaluate G(x) and optionally its gradient.
        
        FORMULE :
        ---------
        G(x) = f(U_r^T x) + ε
        
        GRADIENT (chain rule):
        -----------------------------
        ∇G(x) = U_r ∇f(U_r^T x)
        
        PARAMETERS:
        ------------
        x : (d,) or (N, d) evaluation point(s)
        return_gradient : si True, retourne aussi le gradient
        
        RETOUR :
        --------
        Si return_gradient=False : G(x)
        Si return_gradient=True : (G(x), ∇G(x))
        """
        single_point = (x.ndim == 1)
        
        if single_point:
            x = x.reshape(1, -1)
        
        N = x.shape[0]
        
        # Projection sur sous-espace actif
        z = x @ U_r  # (N, r)
        
        # Evaluate profile function
        values = np.zeros(N)
        for i in range(N):
            values[i] = profile_function(z[i])
        
        # Ajout de bruit
        if noise_level > 0:
            values += rng.randn(N) * noise_level
        
        if not return_gradient:
            return values[0] if single_point else values
        
        # Calcul du gradient
        gradients = np.zeros((N, d))
        for i in range(N):
            grad_f = profile_gradient(z[i])  # ∇f ∈ ℝʳ
            gradients[i] = U_r @ grad_f      # ∇G = U_r ∇f ∈ ℝᵈ
        
        if single_point:
            return values[0], gradients[0]
        else:
            return values, gradients
    
    return G_with_gradient, U_r


def generate_pde_example(d: int) -> Tuple[Callable, int]:
    """
    More realistic example: parameterized PDE.
    
    PROBLÈME :
    ----------
    -∇·(κ(x) ∇u) = 1  sur Ω = [0,1]
    u = 0 sur ∂Ω
    
    where κ(x; ξ) = exp(Σᵢ ξᵢ φᵢ(x)) with ξ ∈ R^d
    
    OUTPUT :
    --------
    G(ξ) = ∫_Ω u(x; ξ) dx  (quantity of interest)
    
    DIMENSION INTRINSÈQUE :
    -----------------------
    In practice, G mainly depends on the first modes of κ
    donc r << d
    
    NOTE :
    ------
    Simplified implementation here. For a real PDE,
    utiliser FEniCS ou un solveur EF.
    """
    # Simplified analytical version
    # κ(ξ) = 1 + Σᵢ ξᵢ/i²  (fast decay)
    
    def G_func(xi, return_gradient=True):
        """
        Approximation : G(ξ) ≈ 1/(1 + κ̄)
        where κ̄ is the mean of κ
        """
        kappa_mean = 1.0 + sum(xi[i] / (i+1)**2 for i in range(len(xi)))
        
        G = 1.0 / (1.0 + kappa_mean)
        
        if not return_gradient:
            return G
        
        # Gradient
        grad = np.zeros(d)
        for i in range(d):
            grad[i] = -1.0 / ((1.0 + kappa_mean)**2 * (i+1)**2)
        
        return G, grad
    
    # True intrinsic dimension: approximately log(d)
    r_true = max(1, int(np.log(d)))
    
    return G_func, r_true


# ============================================================================
# SECTION 2.1.2 : VISUALISATION ACTIVE SUBSPACES
# ============================================================================

def plot_active_subspace_analysis(as_model: ActiveSubspaces, 
                                  X_samples: np.ndarray,
                                  G_samples: np.ndarray,
                                  true_directions: Optional[np.ndarray] = None):
    """
    Complete visualization of Active Subspace analysis.
    
    GRAPHIQUES :
    ------------
    1. Eigenvalue decay (scree plot)
    2. Cumulative activity
    3. Projection 1D/2D (sufficient summary plot)
    4. Compare true vs estimated directions (if available)
    
    PARAMETERS:
    ------------
    as_model : fitted AS model
    X_samples : (N, d) original samples
    G_samples : (N,) valeurs de G
    true_directions : (d, r) vraies directions (si connues)
    """
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Eigenvalue Decay (Activity Plot)",
            "Cumulative Activity",
            "Sufficient Summary Plot (1D)",
            "Subspace Distance (if truth known)"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 1: EIGENVALUE DECAY
    # ─────────────────────────────────────────────────────────────────────────
    
    d = len(as_model.eigenvalues_)
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, d + 1),
            y=as_model.eigenvalues_,
            mode='markers+lines',
            name='Eigenvalues',
            marker=dict(size=8, color='blue'),
            hovertemplate='λ_%{x} = %{y:.4e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Vertical line at selected rank
    if as_model.n_components < d:
        fig.add_vline(
            x=as_model.n_components,
            line_dash="dash",
            line_color="red",
            annotation_text=f"r = {as_model.n_components}",
            row=1, col=1
        )
    
    fig.update_yaxes(type="log", title_text="Eigenvalue", row=1, col=1)
    fig.update_xaxes(title_text="Index", row=1, col=1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 2: CUMULATIVE ACTIVITY
    # ─────────────────────────────────────────────────────────────────────────
    
    cumsum_activity = np.cumsum(as_model.activity_ratio_)
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, d + 1),
            y=cumsum_activity * 100,
            mode='lines',
            name='Cumulative activity',
            line=dict(color='green', width=2),
            fill='tozeroy',
            hovertemplate='%{x} directions<br>%{y:.2f}% activity<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Ligne horizontale au seuil
    fig.add_hline(
        y=as_model.activity_threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{as_model.activity_threshold*100:.0f}% threshold",
        row=1, col=2
    )
    
    fig.update_yaxes(title_text="Cumulative Activity (%)", row=1, col=2)
    fig.update_xaxes(title_text="Number of directions", row=1, col=2)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 3 : SUFFICIENT SUMMARY PLOT
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # OBJECTIF :
    # Visualiser la dépendance de G par rapport au sufficient summary
    # 
    # Si AS fonctionne bien, les points doivent former une courbe 1D
    # (or a 2D surface if r=2) even when x ∈ R^d with d >> r
    
    X_active = as_model.transform(X_samples)
    
    if as_model.n_components == 1:
        # 1D plot: G vs first active direction
        fig.add_trace(
            go.Scatter(
                x=X_active[:, 0],
                y=G_samples,
                mode='markers',
                name='Samples',
                marker=dict(
                    size=5,
                    color=G_samples,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="G(x)", x=1.15)
                ),
                hovertemplate='u₁ᵀx = %{x:.3f}<br>G = %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="u₁ᵀ x (1st active variable)", row=2, col=1)
        fig.update_yaxes(title_text="G(x)", row=2, col=1)
        
    elif as_model.n_components >= 2:
        # 2D plot: projection on the first two directions
        fig.add_trace(
            go.Scatter(
                x=X_active[:, 0],
                y=X_active[:, 1],
                mode='markers',
                name='Samples',
                marker=dict(
                    size=5,
                    color=G_samples,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="G(x)", x=1.15)
                ),
                hovertemplate='u₁ᵀx = %{x:.3f}<br>u₂ᵀx = %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="u₁ᵀ x", row=2, col=1)
        fig.update_yaxes(title_text="u₂ᵀ x", row=2, col=1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOT 4: COMPARISON WITH GROUND TRUTH
    # ─────────────────────────────────────────────────────────────────────────
    
    if true_directions is not None:
        # DISTANCE DE SOUS-ESPACE :
        # dist(U_true, U_est) = ||U_true U_true^T - U_est U_est^T||_F
        # 
        # Equivalent to projector distance
        
        r_true = true_directions.shape[1]
        r_est = as_model.n_components
        
        if r_true == r_est:
            # Calcul de la distance
            P_true = true_directions @ true_directions.T
            P_est = as_model.active_directions_ @ as_model.active_directions_.T
            
            subspace_distance = np.linalg.norm(P_true - P_est, 'fro')
            
            # Angle principal (pour visualisation)
            # θ = arccos(σ_min(U_true^T U_est))
            singular_values = np.linalg.svd(true_directions.T @ as_model.active_directions_, 
                                           compute_uv=False)
            principal_angle = np.arccos(np.clip(singular_values.min(), 0, 1))
            
            # Affichage texte
            fig.add_annotation(
                text=f"Subspace distance: {subspace_distance:.4f}<br>" +
                     f"Principal angle: {np.degrees(principal_angle):.2f}°<br>" +
                     f"(0° = perfect alignment)",
                xref="x4", yref="y4",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14),
                row=2, col=2
            )
            
            # Heatmap des projections
            overlap = np.abs(true_directions.T @ as_model.active_directions_)
            
            fig.add_trace(
                go.Heatmap(
                    z=overlap,
                    x=[f"est_{i+1}" for i in range(r_est)],
                    y=[f"true_{i+1}" for i in range(r_true)],
                    colorscale='Blues',
                    colorbar=dict(title="|overlap|"),
                    hovertemplate='True dir %{y}<br>Est dir %{x}<br>Overlap: %{z:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        else:
            fig.add_annotation(
                text=f"Different ranks:<br>True: {r_true}, Estimated: {r_est}",
                xref="x4", yref="y4",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=2
            )
    else:
        fig.add_annotation(
            text="No ground truth available",
            xref="x4", yref="y4",
            x=0.5, y=0.5,
            showarrow=False,
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="Active Subspace Complete Analysis",
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


# ============================================================================
# SECTION 2.1.3 : ACTIVE SUBSPACES DEMO
# ============================================================================

def demo_active_subspaces():
    """
    Complete demonstration of the Active Subspaces method.
    """
    print("="*70)
    print("ACTIVE SUBSPACES DEMO")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. TEST FUNCTION GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n1. Generating a ridge function...")
    
    d = 20  # Dimension ambiante
    r_true = 2  # Intrinsic dimension
    
    G_func, true_directions = generate_ridge_function(
        d=d,
        r=r_true,
        noise_level=0.05,
        random_state=42
    )
    
    print(f"   Dimension input : {d}")
    print(f"   True intrinsic dimension: {r_true}")
    print(f"   G(x) = f(U_{r_true}^T x) + noise")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. SAMPLING
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n2. Sampling X and computing gradients...")
    
    N_samples = 500
    
    # Distribution des inputs (Gaussienne standard)
    np.random.seed(42)
    X_samples = np.random.randn(N_samples, d)
    
    # Evaluate G and its gradients
    G_values = np.zeros(N_samples)
    gradients = np.zeros((N_samples, d))
    
    print(f"   Computing {N_samples} evaluations...")
    for i in range(N_samples):
        G_values[i], gradients[i] = G_func(X_samples[i], return_gradient=True)
    
    print(f"   Gradients computed: shape {gradients.shape}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. CONSTRUCTION DU SOUS-ESPACE ACTIF
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n3. Construction de l'Active Subspace...")
    
    as_model = ActiveSubspaces(activity_threshold=0.99)
    as_model.fit(gradients)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. VALIDATION: COMPARISON WITH GROUND TRUTH
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n4. Validation...")
    
    if as_model.n_components == r_true:
        # Distance de sous-espace
        P_true = true_directions @ true_directions.T
        P_est = as_model.active_directions_ @ as_model.active_directions_.T
        
        dist = np.linalg.norm(P_true - P_est, 'fro')
        
        print(f"\n   Subspace distance: {dist:.4f}")
        print(f"   (0 = perfect, <0.1 = very good, <0.5 = acceptable)")
        
        # Overlap des directions
        overlap = true_directions.T @ as_model.active_directions_
        print(f"\n   Overlap matrix:")
        print(f"   {overlap}")
        print(f"   (diagonal ≈ ±1 = good alignment)")
    else:
        print(f"\n   Warning: different dimensions: true={r_true}, estimated={as_model.n_components}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. BORNE D'ERREUR
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n5. Estimating ridge-approximation error...")
    
    # For X ~ N(0, I), the Poincare constant is 1
    error_bound = as_model.estimate_error_bound(poincare_constant=1.0)
    
    print(f"\n   Theoretical bound (Theorem 2.5):")
    print(f"   ||G - G*||_L2 ≤ {error_bound:.4e}") 
    
    # Empirical check (approximate)
    # On teste sur de nouveaux points
    N_test = 200
    X_test = np.random.randn(N_test, d)
    
    errors_sq = []
    for i in range(N_test):
        G_true = G_func(X_test[i], return_gradient=False)
        
        # Approximation G* : moyenne sur X_⊥ conditionnelle
        # (simplified here: 1 sample, see Proposition 2.6)
        x_active = as_model.active_directions_.T @ X_test[i]
        x_inactive = np.random.randn(d - as_model.n_components)
        x_approx = (as_model.active_directions_ @ x_active + 
                   as_model.inactive_directions_ @ x_inactive)
        
        G_approx = G_func(x_approx, return_gradient=False)
        
        errors_sq.append((G_true - G_approx)**2)
    
    empirical_error = np.sqrt(np.mean(errors_sq))
    
    print(f"   Empirical error: {empirical_error:.4e}")
    print(f"   Bound/empirical ratio: {error_bound / empirical_error:.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. VISUALISATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n6. Generating visualizations...")
    
    fig = plot_active_subspace_analysis(
        as_model=as_model,
        X_samples=X_samples,
        G_samples=G_values,
        true_directions=true_directions
    )
    
    fig.write_html("active_subspace_analysis.html")
    print("   Saved to 'active_subspace_analysis.html'")
    
    return as_model, G_func, X_samples, G_values, true_directions


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    as_model, G_func, X, G_vals, true_dirs = demo_active_subspaces()
