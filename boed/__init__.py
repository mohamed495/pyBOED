"""
pyBOED - Bayesian Optimal Experimental Design with Model Reduction
===================================================================

A comprehensive library combining:
- Bayesian Optimal Experimental Design (BOED)
- Dimensionality Reduction Methods (PCA, KLE, POD, RB, AS, LIS)

Main modules:
-------------
- boed.core: Core mathematical models, posterior inference, noise
- boed.inference: Public inference API (posterior and inverse models)
- boed.priors: Gaussian Process priors and kernels
- boed.pde: PDE solvers (advection-diffusion, etc.)
- boed.design: Design criteria (A/D/C-opt, EIG) and greedy algorithms
- boed.observations: Sensor placement
- boed.reduction: Dimensionality reduction methods
  - boed.reduction.linear: PCA, KLE, POD
  - boed.reduction.parametric: Reduced Basis
  - boed.reduction.inference: Active Subspaces, LIS
- boed.integration: Hybrid BOED + DimRed methods
- boed.viz: Visualization tools

Quick Start:
-----------
>>> from boed.priors import GaussianProcessPrior
>>> from boed.priors.kernels import Gaussian
>>> from boed.reduction.linear import KLE
>>> from boed.integration import ReducedPriorDesign
>>> 
>>> # Create GP prior
>>> kernel = Gaussian(length_scale=0.5, sigma=1.0)
>>> prior = GaussianProcessPrior(kernel, nx=1000)
>>> 
>>> # Reduce dimension with KLE
>>> reduced_design = ReducedPriorDesign(prior, n_components=50, method='kle')
>>> print(f"Dimension: 1000 → {reduced_design.n_components}")
"""

__version__ = "2.0.0"
__author__ = "Mohamed Doumbouya"

# Core imports
from . import core
from . import inference
from . import priors
from . import pde
from . import design
from . import observations
from . import reduction
from . import integration
from . import utils
from . import viz

# Convenience imports
from .core.initial_conditions import make_u0, available_u0, get_u0_spec
from .core.noise import NoiseModel
from .inference import LinearGaussianModel

__all__ = [
    # Modules
    'core',
    'inference',
    'priors',
    'pde',
    'design',
    'observations',
    'reduction',
    'integration',
    'utils',
    'viz',
    # Convenience functions
    'make_u0',
    'available_u0',
    'get_u0_spec',
    'NoiseModel',
    'LinearGaussianModel',
]
