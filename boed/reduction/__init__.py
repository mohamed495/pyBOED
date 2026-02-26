"""
Dimensionality Reduction Module
================================

Methods for reducing the dimensionality of parameter spaces, observations,
and forward models in the context of BOED.

Submodules:
-----------
- linear: Linear reduction methods (PCA, KLE, POD)
- parametric: Parametric reduction (Reduced Basis)
- inference: Methods for Bayesian inference (Active Subspaces, LIS)

Examples:
---------
>>> from boed.reduction.linear import KLE
>>> from boed.reduction.inference import ActiveSubspaces
>>> 
>>> # Use KLE for GP prior
>>> kle = KLE(domain=[0, 1], n_points=1000)
>>> kle.fit(covariance_function, n_modes=50)
>>> 
>>> # Use Active Subspaces for parameter screening
>>> as_model = ActiveSubspaces(rank=5)
>>> as_model.fit(X, gradients)
"""

# Linear methods
from . import linear
from . import parametric
from . import inference
from . import utils
from . import methods

__all__ = [
    'linear',
    'parametric',
    'inference',
    'utils',
    'methods',
]
