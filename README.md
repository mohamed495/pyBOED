# pyBOED v2.0 - Bayesian Optimal Experimental Design with Model Reduction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library combining **Bayesian Optimal Experimental Design (BOED)** with **Dimensionality Reduction** methods for efficient parameter estimation in high-dimensional inverse problems.

## 🎯 What's New in v2.0

**Major Update**: Integration of dimensionality reduction methods!

- ✨ **KLE/PCA-reduced priors**: Design with 10,000+ parameters
- ⚡ **POD-accelerated forward models**: 10-100x speedup
- 🧮 **Hybrid inference**: Combine BOED with Likelihood-Informed Subspaces
- 📊 **Unified API**: Seamless integration between BOED and reduction methods

**Speedup Example**: A-optimal design with N=10,000 parameters

- v1.0: ~10 hours ⏱️
- v2.0 (KLE, r=50): ~2 minutes ⚡ (~400x faster!)

## 🌟 Key Features

### BOED Core

- **PDE solvers**: Advection-diffusion, customizable forward models
- **GP priors**: Squared Exponential, Matérn kernels
- **Design criteria**: A-optimal, D-optimal, C-optimal, Expected Information Gain
- **Greedy algorithms**: Efficient sensor selection
- **Linear Gaussian inference**: Analytical posterior computation

### Dimensionality Reduction

- **PCA**: Principal Component Analysis
- **KLE**: Karhunen-Loève Expansion (optimal for GPs)
- **POD**: Proper Orthogonal Decomposition (for PDE solutions)
- **RB**: Reduced Basis Method (parametric PDEs)
- **AS**: Active Subspaces (parameter importance)
- **LIS**: Likelihood-Informed Subspaces (Bayesian inference)

### Integration (NEW in v2.0!)

- **ReducedPriorDesign**: BOED with KLE/PCA-reduced priors
- **PODForwardModel**: Accelerated PDE evaluations
- **HybridBayesianInference**: Posterior in informative subspaces

## 📦 Installation

### From source (recommended)

```bash
git clone https://github.com/yourusername/pyBOED.git
cd pyBOED
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

Core: `numpy`, `scipy`, `matplotlib`, `plotly`, `pyyaml`, `scikit-learn`

## 🚀 Quick Start

### Example 1: Standard BOED (v1.0 style)

```python
import numpy as np
from boed.priors import GaussianProcessPrior
from boed.priors.kernels import SquaredExponential
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.core.noise import NoiseModel
from boed.design.greedy import run_greedy_oed

# Create GP prior
kernel = SquaredExponential(length_scale=0.5, sigma=1.0)
prior = GaussianProcessPrior(kernel, nx=200)

# Forward model
model = AdvectionDiffusion1D_CN(N=200, dt=0.005)
noise = NoiseModel(sigma_noise=0.001)

# Run A-optimal design
design, history, Sigma_post = run_greedy_oed(
    model=model,
    Sigma_prior=prior.Sigma,
    noise_model=noise,
    candidates_x=np.arange(200),
    candidates_t=np.arange(20),
    n_budget=10,
    criterion_type="A"
)

print(f"Selected sensors: {design}")
```

### Example 2: BOED with Reduced Prior (NEW!)

```python
from boed.priors import GaussianProcessPrior
from boed.priors.kernels import SquaredExponential
from boed.integration import ReducedPriorDesign

# High-dimensional prior (10,000 parameters!)
kernel = SquaredExponential(length_scale=0.1, sigma=1.0)
prior_full = GaussianProcessPrior(kernel, nx=10000)

# Reduce to 50 dimensions using KLE
reduced_design = ReducedPriorDesign(
    prior=prior_full,
    n_components=50,
    method='kle'
)

print(reduced_design.summary())
# Output:
# Dimension: 10000 → 50
# Speedup estimate: ~400x
# Energy captured: 99.99%

# Run design in reduced space (FAST!)
design, history, Sigma_post_reduced = reduced_design.run_design(
    model=model,
    noise_model=noise,
    candidates_x=candidates_x,
    candidates_t=candidates_t,
    n_budget=20,
    criterion_type="A"
)

# Reconstruct full posterior if needed
Sigma_post_full = reduced_design.reconstruct_posterior(Sigma_post_reduced)
```

### Example 3: POD-Accelerated Forward Model

```python
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.integration import PODForwardModel
from boed.core import make_u0

# Full model
full_model = AdvectionDiffusion1D_CN(N=1000, dt=0.001)

# Create POD model
pod_model = PODForwardModel(full_model, energy_threshold=0.9999)

# Build reduced model (offline, expensive)
x = np.linspace(0, 1, 1000)
parameter_samples = [
    make_u0(x, "gaussian", mu=mu, sigma=0.1)
    for mu in np.linspace(0.2, 0.8, 50)
]
pod_model.build_reduced_model(parameter_samples, n_steps=100)

# Fast online evaluations
u0_new = make_u0(x, "gaussian", mu=0.5, sigma=0.1)
trajectory = pod_model.evolve_reduced(u0_new, n_steps=100)

print(f"Speedup: {pod_model.get_speedup():.0f}x")
# Output: Speedup: ~25x
```

### Example 4: Hybrid Bayesian Inference

```python
from boed.integration import HybridBayesianInference

# Set up hybrid inference
inference = HybridBayesianInference(
    forward_operator=H,
    noise_covariance=R,
    prior_mean=mu_prior,
    prior_covariance=Sigma_prior,
    lis_rank=10  # Reduce to 10D subspace
)

# Identify informative subspace
inference.identify_informative_subspace(observations, n_samples=1000)

# Compute posterior in reduced space
mu_post_reduced, Sigma_post_reduced = inference.posterior_in_subspace(observations)

# Reconstruct full parameter
mu_post_full = inference.reconstruct_from_subspace(mu_post_reduced)

print(inference.summary())
```

## 📚 Documentation

### Package Structure

```markdown
pyBOED/
├── boed/                      # Main package
│   ├── core/                  # BOED fundamentals
│   │   ├── base.py           # Abstract classes
│   │   ├── posterior.py      # Bayesian inference
│   │   ├── noise.py          # Noise models
│   │   └── initial_conditions.py  # u0 library
│   │
│   ├── priors/               # Prior specifications
│   │   ├── gp_priors.py      # Gaussian Process priors
│   │   └── kernels.py        # Covariance kernels
│   │
│   ├── pde/                  # Forward models
│   │   ├── advection_diffusion.py
│   │   └── problems.py       # PDE problem definitions
│   │
│   ├── design/               # Design optimization
│   │   ├── criteria.py       # A/D/C-opt, EIG
│   │   ├── greedy.py         # Greedy algorithms
│   │   └── selection.py      # Sensor selection
│   │
│   ├── observations/         # Observation operators
│   │   └── sensors.py        # Space-time sensors
│   │
│   ├── reduction/            # 🆕 Dimensionality reduction
│   │   ├── linear/           # PCA, KLE, POD
│   │   ├── parametric/       # Reduced Basis
│   │   └── inference/        # Active Subspaces, LIS
│   │
│   ├── integration/          # 🆕 Hybrid methods
│   │   ├── reduced_design.py      # KLE/PCA + BOED
│   │   ├── accelerated_forward.py # POD + forward model
│   │   └── hybrid_inference.py    # LIS + posterior
│   │
│   └── viz/                  # Visualization
│       ├── boed_visualizer.py
│       └── reduction_viz.py
│
├── tutorials/               # User-facing learning material
│   ├── examples/            # Runnable scripts
│   │   ├── boed/            # Classic BOED examples
│   │   ├── reduction/       # Reduction examples
│   │   └── hybrid/          # 🆕 Combined examples
│   └── notebooks/           # Interactive tutorials
│       ├── boed/
│       └── reduction/
│
└── tests/                   # Test suite
    ├── test_boed/
    ├── test_reduction/
    └── test_integration/    # 🆕 Integration tests
```

### Available Reduction Methods

| Method | Module | Best For | Complexity |
|--------|--------|----------|------------|
| **PCA** | `reduction.linear.pca` | Tabular data, exploratory | O(n²m) |
| **KLE** | `reduction.linear.kle` | GP priors, continuous fields | O(n³) |
| **POD** | `reduction.linear.pod` | PDE snapshots, dynamics | O(nm²) |
| **RB** | `reduction.parametric.rb` | Parametric PDEs | O(n³) offline |
| **AS** | `reduction.inference.active_subspace` | Parameter screening | O(nm) |
| **LIS** | `reduction.inference.lis` | Bayesian inference | O(nm) |

*n = ambient dimension, m = number of samples/snapshots*

## 🎓 Tutorials & Examples

### Basic Tutorials

- `tutorials/examples/boed/tutorial.py` - Complete BOED workflow
- `tutorials/examples/reduction/demo_kle.py` - KLE basics
- `tutorials/examples/reduction/demo_pod.py` - POD basics

### Advanced Examples

- `tutorials/examples/hybrid/kle_prior_boed.py` - 10,000D prior reduction
- `tutorials/examples/hybrid/pod_accelerated_design.py` - Fast forward models
- `tutorials/examples/hybrid/active_subspace_design.py` - AS-informed design

### Jupyter Notebooks

- `tutorials/notebooks/boed/` - Interactive BOED tutorials
- `tutorials/notebooks/reduction/` - Dimensionality reduction demos
- Hybrid notebooks are not yet grouped as a dedicated folder

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v --cov=boed
```

Run specific test modules:

```bash
pytest tests/test_boed/ -v          # BOED tests
pytest tests/test_reduction/ -v     # Reduction tests
pytest tests/test_integration/ -v   # Integration tests
```

## 📊 Performance Benchmarks

### Design Optimization Speedup

| N (params) | Method | Time | Speedup |
|------------|--------|------|---------|
| 100 | Full | 2s | 1x |
| 1,000 | Full | 180s | 1x |
| 1,000 | KLE (r=50) | 5s | 36x |
| 10,000 | Full | ~10h | 1x |
| 10,000 | KLE (r=50) | 120s | 400x |

### Forward Model Speedup (POD)

| N (spatial) | Full Time | POD Time (r=20) | Speedup |
|-------------|-----------|-----------------|---------|
| 100 | 0.1s | 0.05s | 2x |
| 500 | 2.5s | 0.15s | 17x |
| 1,000 | 10s | 0.4s | 25x |
| 5,000 | 250s | 5s | 50x |

*Benchmarks on Intel i7, single core*

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/yourusername/pyBOED.git
cd pyBOED
pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (NumPy style)
- Add tests for new features

## 📖 Citation

If you use pyBOED in your research, please cite:

```bibtex
@software{pyBOED2024,
  author = {Doumbou, M.},
  title = {pyBOED: Bayesian Optimal Experimental Design with Model Reduction},
  year = {2024},
  version = {2.0.0},
  url = {https://github.com/yourusername/pyBOED}
}
```

For the BOED component:

```bibtex
@article{boed_paper,
  title = {Bayesian Optimal Experimental Design for Inverse Problems},
  author = {...},
  journal = {...},
  year = {2024}
}
```

For dimensionality reduction methods, please cite the original papers:

- **KLE**: Loève, M. (1977). Probability Theory
- **POD**: Lumley, J. L. (1967). Atmospheric Turbulence
- **Active Subspaces**: Constantine, P. G. (2015). Active Subspaces
- **LIS**: Cui et al. (2014). Likelihood-Informed Dimension Reduction

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Original BOED implementation: pyCBOED project
- Dimensionality reduction methods: DimReduction project
- Integration and unification: v2.0 development

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/pyBOED/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pyBOED/discussions)
- **Email**: <your.email@example.com>

## 🗺️ Roadmap

### v2.1 (Q2 2024)

- [ ] Nonlinear observation operators
- [ ] Multi-fidelity models
- [ ] Adaptive refinement

### v2.2 (Q3 2024)

- [ ] GPU acceleration (CuPy backend)
- [ ] Distributed computing (MPI)
- [ ] Interactive web dashboard

### v3.0 (Q4 2024)

- [ ] Neural network surrogates
- [ ] Deep learning-based reduction
- [ ] Reinforcement learning for design

---

**Version**: 2.0.0 | **Status**: Active Development | **Last Updated**: February 2024
