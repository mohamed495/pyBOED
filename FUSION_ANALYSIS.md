# Analyse de Fusion : pyCBOED + DimReduction

## 🎯 Vision de la Fusion

**Nom proposé : `pyBOED` (Bayesian Optimal Experimental Design with Model Reduction)**

### Justification

La réduction de dimension est **naturellement complémentaire** au BOED :

- **Accélération** : Réduire la dimensionalité des priors GP pour des calculs plus rapides
- **Efficacité** : Compression des observations pour un design optimal
- **Scalabilité** : Permettre le BOED sur des problèmes haute dimension (10^4-10^6 paramètres)

---

## 📊 Comparaison Structurelle

### pyCBOED (Focus : Design Expérimental)

```markdown
boed/
├── core/              # Modèles mathématiques, posterior, noise
├── priors/            # GP priors, kernels
├── pde/               # Solveurs PDE (advection-diffusion)
├── design/            # Critères (A/D/C-opt, EIG), greedy
├── observations/      # Capteurs espace-temps
├── algebra/           # RNLA (randomized numerical linear algebra)
└── viz/               # Visualisation BOED
```

### DimReduction (Focus : Réduction de Dimension)

```markdown
dimreduction/
├── methods/           # PCA, KLE, POD, RB, AS, LIS
│   ├── base.py       # Classe abstraite commune
│   ├── pca.py
│   ├── kle.py
│   ├── pod.py
│   ├── rb.py
│   ├── active_subspace.py
│   └── lis.py
└── utils/
    ├── covariance.py      # Fonctions de covariance
    ├── data_generation.py # Génération de données
    └── problems.py        # Définitions PDE
```

---

## 🔗 Points de Synergie Identifiés

### 1. **Priors Gaussiens ↔ KLE/PCA**

- **pyCBOED** : Utilise des GP priors avec kernels (SquaredExponential, Matern)
- **DimReduction** : KLE décompose des processus Gaussiens, PCA sur données
- **Fusion** : Utiliser KLE pour représenter les priors GP en dimension réduite

```python
# Cas d'usage
prior_gp = GaussianProcessPrior(kernel, nx=10000)  # 10k params
kle = KLE(domain=[0,1], n_points=10000)
kle.fit(prior_gp.covariance_function)
# Prior réduit : 10000 → 50 dimensions
```

### 2. **PDE Solvers ↔ POD/Reduced Basis**

- **pyCBOED** : `AdvectionDiffusion1D_CN` pour évolution temporelle
- **DimReduction** : POD pour snapshots PDE, RB pour approximation rapide
- **Fusion** : Accélérer les évaluations du forward model via POD

```python
# Cas d'usage
model = AdvectionDiffusion1D_CN(N=1000, dt=0.001)
# Générer snapshots pour différents paramètres
snapshots = [model.evolve(u0_i, n_steps=100) for u0_i in param_samples]
pod = POD(energy_threshold=0.9999)
pod.fit(snapshots)
# Modèle réduit : 1000 → 20 modes
```

### 3. **Design Optimization ↔ Active Subspaces**

- **pyCBOED** : Optimisation greedy avec critères A/D-opt
- **DimReduction** : Active Subspaces identifie les directions importantes
- **Fusion** : Pré-identifier les paramètres influents avant le design

### 4. **Posterior Inference ↔ LIS**

- **pyCBOED** : `LinearGaussianModel` pour inférence analytique
- **DimReduction** : LIS (Likelihood-Informed Subspaces) pour Bayesian inference
- **Fusion** : Accélérer le calcul du posterior en dimension réduite

### 5. **Covariance Functions**

- **pyCBOED** : Kernels dans `priors/kernels.py`
- **DimReduction** : `utils/covariance.py` avec fonctions exponentielles
- **Fusion** : Unifier dans un module `kernels` commun

---

## 🏗️ Structure Proposée pour la Fusion

```markdown
pyBOED/
├── setup.py                    # Installation unifiée
├── requirements.txt
├── README.md                   # Documentation fusionnée
├── MIGRATION_GUIDE.md          # Guide pour utilisateurs existants
│
├── boed/                       # Package principal
│   ├── __init__.py
│   │
│   ├── core/                   # BOED core (inchangé)
│   │   ├── base.py
│   │   ├── initial_conditions.py
│   │   ├── posterior.py
│   │   ├── noise.py
│   │   └── utils.py
│   │
│   ├── priors/                 # Priors (étendu)
│   │   ├── gp_priors.py
│   │   ├── kernels.py         # 🔄 FUSIONNÉ avec covariance.py
│   │   └── reduced_priors.py  # 🆕 NOUVEAU (KLE-based priors)
│   │
│   ├── pde/                    # PDE solvers (inchangé)
│   │   ├── advection_diffusion.py
│   │   └── problems.py        # 🔄 IMPORTÉ de DimReduction
│   │
│   ├── design/                 # Design optimization (étendu)
│   │   ├── criteria.py
│   │   ├── greedy.py
│   │   ├── selection.py
│   │   └── active_design.py   # 🆕 NOUVEAU (AS-informed design)
│   │
│   ├── observations/           # Observations (inchangé)
│   │   └── sensors.py
│   │
│   ├── algebra/                # RNLA (inchangé)
│   │   ├── rnla_core.py
│   │   └── rnla.py
│   │
│   ├── reduction/              # 🆕 NOUVEAU MODULE
│   │   ├── __init__.py
│   │   ├── base.py            # Base class commune
│   │   ├── linear/            # Méthodes linéaires
│   │   │   ├── pca.py
│   │   │   ├── kle.py
│   │   │   └── pod.py
│   │   ├── parametric/        # Méthodes paramétriques
│   │   │   └── rb.py
│   │   └── inference/         # Méthodes pour inférence
│   │       ├── active_subspace.py
│   │       └── lis.py
│   │
│   ├── integration/            # 🆕 NOUVEAU (Ponts BOED↔DimRed)
│   │   ├── __init__.py
│   │   ├── reduced_design.py  # Design avec priors réduits
│   │   ├── accelerated_forward.py  # Forward model via POD/RB
│   │   └── hybrid_inference.py     # Posterior + LIS
│   │
│   ├── utils/                  # Utilitaires (fusionné)
│   │   ├── __init__.py
│   │   ├── data_generation.py # De DimReduction
│   │   └── validation.py      # Validation commune
│   │
│   └── viz/                    # Visualisation (fusionné)
│       ├── boed_visualizer.py
│       ├── boed_visualizer_pro.py
│       └── reduction_viz.py   # 🆕 Pour dim reduction
│
├── tutorials/examples/                   # Exemples réorganisés
│   ├── README.md
│   ├── boed/                  # Exemples BOED purs
│   │   ├── tutorial.py
│   │   ├── run_greedy.py
│   │   └── run_greedy_A_D.py
│   ├── reduction/             # Exemples DimRed purs
│   │   ├── demo_pca.py
│   │   ├── demo_kle.py
│   │   └── demo_pod.py
│   └── hybrid/                # 🆕 Exemples combinés
│       ├── kle_prior_boed.py
│       ├── pod_accelerated_design.py
│       └── active_subspace_design.py
│
├── tutorials/notebooks/                  # Notebooks interactifs
│   ├── boed/                  # De pyCBOED
│   ├── reduction/             # De DimReduction
│   │   ├── 01_pca_interactive.ipynb
│   │   ├── 02_kle_interactive.ipynb
│   │   └── ...
│   └── hybrid/                # 🆕 Nouveaux notebooks
│       ├── kle_gp_prior.ipynb
│       └── pod_boed.ipynb
│
├── data/                      # Données
│   ├── pca/
│   └── generated/
│
├── results/                   # Résultats
│   ├── boed/
│   └── reduction/
│
└── tests/                     # Tests unifiés
    ├── test_boed/
    │   ├── test_core.py
    │   ├── test_greedy.py
    │   └── ...
    ├── test_reduction/
    │   ├── test_pca.py
    │   ├── test_kle.py
    │   └── ...
    └── test_integration/      # 🆕 Tests de l'intégration
        ├── test_reduced_priors.py
        ├── test_pod_forward.py
        └── test_hybrid_design.py
```

---

## 🔧 Modules d'Intégration Clés à Créer

### 1. `boed/integration/reduced_design.py`

**Objectif** : Design BOED avec priors en dimension réduite

```python
"""
BOED avec priors réduits via KLE/PCA
"""
from typing import Optional, Tuple
import numpy as np
from ..priors.gp_priors import GaussianProcessPrior
from ..reduction.linear.kle import KLE
from ..reduction.linear.pca import PCA
from ..design.greedy import run_greedy_oed

class ReducedPriorDesign:
    """
    Design BOED avec prior GP réduit par KLE
    
    Workflow:
    1. Construire le prior GP complet (dim haute)
    2. Réduire via KLE (dim basse)
    3. Faire le design en dim réduite (rapide)
    4. Reconstruire en dim complète si nécessaire
    """
    
    def __init__(
        self, 
        prior: GaussianProcessPrior,
        n_components: int = 50,
        method: str = 'kle'
    ):
        self.prior_full = prior
        self.n_components = n_components
        self.method = method
        
        # Réducteur
        if method == 'kle':
            self.reducer = KLE(
                domain=[0, 1], 
                n_points=prior.Sigma.shape[0]
            )
            self.reducer.fit(prior.kernel.evaluate)
        elif method == 'pca':
            self.reducer = PCA(n_components=n_components)
            # Échantillonner le prior pour PCA
            samples = np.random.multivariate_normal(
                prior.mu, prior.Sigma, size=1000
            )
            self.reducer.fit(samples)
        
        # Prior réduit
        self.Sigma_reduced = self._compute_reduced_covariance()
        
    def _compute_reduced_covariance(self) -> np.ndarray:
        """Calcule la covariance en espace réduit"""
        if self.method == 'kle':
            # Sigma_r = Λ (matrice diagonale des eigenvalues)
            return np.diag(self.reducer.eigenvalues[:self.n_components])
        else:  # pca
            # Projection de la covariance
            U = self.reducer.components_[:self.n_components]
            return U @ self.prior_full.Sigma @ U.T
    
    def run_design(
        self,
        model,
        noise_model,
        candidates_x,
        candidates_t,
        n_budget: int,
        criterion_type: str = "A"
    ):
        """
        Exécute le design en espace réduit
        
        Returns:
            design: Positions sélectionnées
            history: Historique du critère
            Sigma_post_reduced: Covariance posterior réduite
        """
        design, history, Sigma_post_reduced = run_greedy_oed(
            model=model,
            Sigma_prior=self.Sigma_reduced,
            noise_model=noise_model,
            candidates_x=candidates_x,
            candidates_t=candidates_t,
            n_budget=n_budget,
            criterion_type=criterion_type
        )
        
        return design, history, Sigma_post_reduced
    
    def reconstruct_posterior(
        self, 
        Sigma_post_reduced: np.ndarray
    ) -> np.ndarray:
        """Reconstruit la covariance posterior en dim complète"""
        if self.method == 'kle':
            U = self.reducer.eigenfunctions[:, :self.n_components]
            return U @ Sigma_post_reduced @ U.T
        else:  # pca
            U = self.reducer.components_[:self.n_components]
            return U.T @ Sigma_post_reduced @ U
```

### 2. `boed/integration/accelerated_forward.py`

**Objectif** : Accélérer le forward model via POD

```python
"""
Forward model accéléré via POD/Reduced Basis
"""
import numpy as np
from typing import List, Callable
from ..pde.advection_diffusion import AdvectionDiffusion1D_CN
from ..reduction.linear.pod import POD
from ..reduction.parametric.rb import ReducedBasis

class PODForwardModel:
    """
    Modèle réduit du forward PDE via POD
    
    Usage:
    1. Générer des snapshots offline (coûteux)
    2. Construire la base POD (une fois)
    3. Évaluations online rapides pour le design
    """
    
    def __init__(
        self,
        full_model: AdvectionDiffusion1D_CN,
        energy_threshold: float = 0.9999
    ):
        self.full_model = full_model
        self.pod = POD(energy_threshold=energy_threshold)
        self.is_fitted = False
        
    def build_reduced_model(
        self,
        parameter_samples: List[np.ndarray],
        n_steps: int = 100
    ):
        """
        Phase offline : construire le modèle réduit
        
        Args:
            parameter_samples: Liste de conditions initiales u0
            n_steps: Nombre de pas de temps
        """
        print("Building POD reduced model (offline phase)...")
        
        # Générer snapshots
        snapshots = []
        for u0 in parameter_samples:
            trajectory = self.full_model.evolve(u0, n_steps=n_steps)
            snapshots.append(trajectory.flatten())
        
        snapshots = np.array(snapshots).T  # (N*T, n_samples)
        
        # Construire POD
        self.pod.fit(snapshots, use_snapshot_method=True)
        self.is_fitted = True
        
        print(f"✓ POD basis built: {self.pod.n_components} modes")
        print(f"  Energy captured: {self.pod.cumulative_energy[-1]:.4%}")
    
    def evolve_reduced(
        self, 
        u0: np.ndarray, 
        n_steps: int = 100
    ) -> np.ndarray:
        """
        Phase online : évolution rapide en espace réduit
        
        Args:
            u0: Condition initiale
            n_steps: Nombre de pas de temps
            
        Returns:
            trajectory: Solution approximée (N, n_steps+1)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call build_reduced_model first")
        
        # Projeter u0 sur la base POD
        u0_reduced = self.pod.transform(u0.reshape(1, -1))[0]
        
        # Évolution en espace réduit (simplifié ici)
        # Dans un vrai cas, il faudrait projeter l'opérateur aussi
        # Pour l'instant, approximation par projection/reconstruction
        
        trajectory_reduced = np.zeros((self.pod.n_components, n_steps+1))
        trajectory_reduced[:, 0] = u0_reduced
        
        # Evolution (à implémenter proprement)
        for t in range(n_steps):
            # Reconstruire, évoluer, projeter (inefficace, juste démo)
            u_full = self.pod.inverse_transform(
                trajectory_reduced[:, t].reshape(1, -1)
            )[0]
            u_next = self.full_model.step(u_full)
            trajectory_reduced[:, t+1] = self.pod.transform(
                u_next.reshape(1, -1)
            )[0]
        
        # Reconstruire la trajectoire complète
        trajectory_full = np.zeros((self.full_model.N, n_steps+1))
        for t in range(n_steps+1):
            trajectory_full[:, t] = self.pod.inverse_transform(
                trajectory_reduced[:, t].reshape(1, -1)
            )[0]
        
        return trajectory_full
    
    def get_speedup(self) -> float:
        """Estimation du speedup"""
        if not self.is_fitted:
            return 1.0
        
        reduction_factor = self.full_model.N / self.pod.n_components
        # Speedup approximatif (dépend de l'implémentation)
        return reduction_factor ** 2  # O(N^2) → O(r^2)
```

### 3. `boed/integration/hybrid_inference.py`

**Objectif** : Combiner posterior BOED avec LIS

```python
"""
Inférence Bayésienne hybride : BOED + LIS
"""
import numpy as np
from ..core.posterior import LinearGaussianModel
from ..reduction.inference.lis import LikelihoodInformedSubspaces

class HybridBayesianInference:
    """
    Inférence accélérée combinant:
    - LinearGaussianModel (BOED) pour posterior analytique
    - LIS pour identifier le sous-espace informé par la vraisemblance
    """
    
    def __init__(
        self,
        forward_operator: np.ndarray,
        noise_covariance: np.ndarray,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray,
        lis_rank: int = 10
    ):
        # Modèle complet
        self.lgm = LinearGaussianModel(
            A=forward_operator,
            Sigma_noise=noise_covariance,
            mu_prior=prior_mean,
            Sigma_prior=prior_covariance
        )
        
        # LIS pour identifier directions importantes
        self.lis = LikelihoodInformedSubspaces(rank=lis_rank)
        
    def identify_informative_subspace(
        self,
        observations: np.ndarray,
        n_samples: int = 1000
    ):
        """
        Identifie le sous-espace informé par la vraisemblance
        
        Args:
            observations: Observations y
            n_samples: Nombre d'échantillons pour LIS
        """
        # Échantillonner le prior
        prior_samples = np.random.multivariate_normal(
            self.lgm.mu_prior,
            self.lgm.Sigma_prior,
            size=n_samples
        )
        
        # Évaluer la log-vraisemblance pour chaque échantillon
        log_likelihoods = np.array([
            self._log_likelihood(sample, observations)
            for sample in prior_samples
        ])
        
        # Construire LIS
        self.lis.fit(prior_samples, log_likelihoods)
        
    def _log_likelihood(
        self, 
        parameter: np.ndarray, 
        observations: np.ndarray
    ) -> float:
        """Calcule log p(y|θ)"""
        predicted = self.lgm.A @ parameter
        residual = observations - predicted
        
        # log N(y | Aθ, Σ_noise)
        log_lik = -0.5 * (
            residual.T @ np.linalg.solve(self.lgm.Sigma_noise, residual)
        )
        log_lik -= 0.5 * np.linalg.slogdet(self.lgm.Sigma_noise)[1]
        log_lik -= 0.5 * len(observations) * np.log(2 * np.pi)
        
        return log_lik
    
    def posterior_in_subspace(
        self,
        observations: np.ndarray
    ) -> tuple:
        """
        Calcule le posterior dans le sous-espace LIS
        
        Returns:
            mu_post_reduced: Moyenne posterior réduite
            Sigma_post_reduced: Covariance posterior réduite
        """
        # Posterior complet
        mu_post, Sigma_post = self.lgm.posterior(observations)
        
        # Projection sur LIS
        U = self.lis.basis  # (d, r)
        mu_post_reduced = U.T @ mu_post
        Sigma_post_reduced = U.T @ Sigma_post @ U
        
        return mu_post_reduced, Sigma_post_reduced
    
    def reconstruct_from_subspace(
        self,
        mu_reduced: np.ndarray
    ) -> np.ndarray:
        """Reconstruit depuis le sous-espace LIS"""
        return self.lis.basis @ mu_reduced
```

---

## 📝 Plan de Migration par Étapes

### Phase 1 : Préparation (1-2 jours)

- [ ] Créer la structure de dossiers unifiée
- [ ] Copier pyCBOED comme base
- [ ] Créer `boed/reduction/` avec sous-modules

### Phase 2 : Migration DimReduction (2-3 jours)

- [ ] Copier `methods/` → `boed/reduction/`
- [ ] Adapter les imports (retirer `dimreduction.`)
- [ ] Fusionner `utils/covariance.py` avec `priors/kernels.py`
- [ ] Déplacer `utils/problems.py` → `pde/problems.py`
- [ ] Migrer `utils/data_generation.py` → `utils/`

### Phase 3 : Création des Modules d'Intégration (3-4 jours)

- [ ] Implémenter `integration/reduced_design.py`
- [ ] Implémenter `integration/accelerated_forward.py`
- [ ] Implémenter `integration/hybrid_inference.py`
- [ ] Créer `priors/reduced_priors.py`

### Phase 4 : Exemples et Tests (2-3 jours)

- [ ] Créer `tutorials/examples/hybrid/` avec 3-4 exemples
- [ ] Migrer notebooks dans structure unifiée
- [ ] Écrire tests d'intégration dans `tests/test_integration/`
- [ ] Valider que tous les anciens tests passent

### Phase 5 : Documentation (1-2 jours)

- [ ] Fusionner README.md
- [ ] Créer MIGRATION_GUIDE.md
- [ ] Documenter les nouvelles fonctionnalités
- [ ] Créer tutoriel "Getting Started with Hybrid BOED"

### Phase 6 : Finalisation (1 jour)

- [ ] Mettre à jour `setup.py` et `requirements.txt`
- [ ] Vérifier compatibilité backward
- [ ] Tagger version 2.0.0
- [ ] Créer CHANGELOG.md détaillé

---

## 🎓 Exemples d'Usage Hybrides

### Exemple 1 : Design avec Prior KLE

```python
# example/hybrid/kle_prior_boed.py
import numpy as np
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import SquaredExponential
from boed.integration.reduced_design import ReducedPriorDesign
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.core.noise import NoiseModel

# 1. Prior GP haute dimension (10k paramètres)
N = 10000
kernel = SquaredExponential(length_scale=0.5, sigma=1.0)
prior_full = GaussianProcessPrior(kernel, nx=N)

# 2. Réduction par KLE (10k → 50)
reduced_design = ReducedPriorDesign(
    prior=prior_full,
    n_components=50,
    method='kle'
)

print(f"Dimension originale: {N}")
print(f"Dimension réduite: {reduced_design.n_components}")
print(f"Speedup attendu: ~{(N/50)**2:.0f}x")

# 3. Forward model
model = AdvectionDiffusion1D_CN(N, dt=0.001, diffusivity=0.01)
noise = NoiseModel(sigma_noise=0.001)

# 4. Design en espace réduit (RAPIDE!)
design, history, Sigma_post_reduced = reduced_design.run_design(
    model=model,
    noise_model=noise,
    candidates_x=np.arange(N),
    candidates_t=np.arange(50),
    n_budget=20,
    criterion_type="A"
)

print(f"Design sélectionné: {design}")
print(f"Critère final: {history[-1]:.4e}")

# 5. Reconstruire le posterior complet si besoin
Sigma_post_full = reduced_design.reconstruct_posterior(Sigma_post_reduced)
```

### Exemple 2 : Forward Model POD

```python
# tutorials/examples/hybrid/pod_accelerated_design.py
import numpy as np
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.core import make_u0
from boed.integration.accelerated_forward import PODForwardModel

# 1. Modèle complet
N = 1000
full_model = AdvectionDiffusion1D_CN(N, dt=0.001, diffusivity=0.01)

# 2. Créer modèle POD
pod_model = PODForwardModel(full_model, energy_threshold=0.9999)

# 3. Phase offline: générer snapshots
x = np.linspace(0, 1, N)
parameter_samples = [
    make_u0(x, "gaussian", mu=mu, sigma=0.1)
    for mu in np.linspace(0.2, 0.8, 50)
]

pod_model.build_reduced_model(
    parameter_samples=parameter_samples,
    n_steps=100
)

print(f"Speedup estimé: {pod_model.get_speedup():.1f}x")

# 4. Phase online: évaluations rapides
u0_test = make_u0(x, "gaussian", mu=0.5, sigma=0.1)

import time
t0 = time.time()
traj_full = full_model.evolve(u0_test, n_steps=100)
time_full = time.time() - t0

t0 = time.time()
traj_reduced = pod_model.evolve_reduced(u0_test, n_steps=100)
time_reduced = time.time() - t0

print(f"Temps full model: {time_full:.3f}s")
print(f"Temps POD model: {time_reduced:.3f}s")
print(f"Speedup réel: {time_full/time_reduced:.1f}x")
print(f"Erreur relative: {np.linalg.norm(traj_full - traj_reduced) / np.linalg.norm(traj_full):.2e}")
```

---

## ⚠️ Points d'Attention

### Gestion des Dépendances

```python
# requirements.txt fusionné
numpy>=1.21
scipy>=1.7
matplotlib>=3.5
plotly>=5.0          # Pour DimReduction viz
pyyaml>=5.4          # Pour pyCBOED config
scikit-learn>=1.0    # Pour PCA validation
```

### Conflits Potentiels

1. **Base Classes** : 
   - pyCBOED : `ForwardModelBase` dans `core/base.py`
   - DimReduction : `DimensionalityReductionBase` dans `methods/base.py`
   - ✅ Pas de conflit, domaines différents

2. **utils/** :
   - Fusionner intelligemment, éviter doublons
   - `covariance.py` → intégrer dans `kernels.py`

3. **Visualisation** :
   - Garder séparé : `viz/boed_visualizer.py` et `viz/reduction_viz.py`

### Backward Compatibility

```python
# Pour les utilisateurs de pyCBOED v1.0
# Garder les imports existants fonctionnels
from boed.priors import GaussianProcessPrior  # ✓ OK
from boed.design.criteria import DesignCriteria  # ✓ OK

# Nouveaux imports
from boed.reduction.linear import PCA, KLE, POD  # 🆕
from boed.integration import ReducedPriorDesign  # 🆕
```

---

## 📈 Bénéfices Attendus

### Performance

- **10-1000x** speedup pour problèmes haute dimension (N > 10^4)
- Design sur **10^6 paramètres** devient faisable

### Fonctionnalités

- Design optimal avec priors GP réduits
- Forward models accélérés via POD/RB
- Inférence Bayésienne en sous-espaces

### Scientific Impact

- BOED devient **scalable** pour problèmes réalistes
- Combinaison unique dans l'écosystème Python
- Publications potentielles sur méthodes hybrides

---

## ✅ Checklist de Validation

Avant de finaliser la fusion, vérifier :

- [ ] Tous les tests de pyCBOED passent
- [ ] Tous les tests de DimReduction passent
- [ ] Nouveaux tests d'intégration créés et passent
- [ ] Exemples hybrides exécutables
- [ ] Documentation complète
- [ ] README.md unifié et clair
- [ ] MIGRATION_GUIDE.md pour utilisateurs existants
- [ ] Pas de régression de performance
- [ ] Backward compatibility préservée

---

## 🚀 Prochaines Étapes

Voulez-vous que je :

1. **Génère la structure complète** de dossiers et fichiers vides
2. **Crée les modules d'intégration** détaillés (code complet)
3. **Écris des exemples hybrides** fonctionnels
4. **Prépare un script de migration** automatique
5. **Rédige le README fusionné** et MIGRATION_GUIDE

Quelle option préférez-vous pour commencer ?
