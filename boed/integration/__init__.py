"""
Integration Module
==================

Hybrid methods combining BOED with dimensionality reduction.

Classes:
--------
- ReducedPriorDesign: BOED with KLE/PCA-reduced priors
- PODForwardModel: Accelerated forward model via POD
- HybridBayesianInference: Posterior inference with LIS

Usage:
------
>>> from boed.integration import ReducedPriorDesign
>>> from boed.priors import GaussianProcessPrior
>>> 
>>> prior = GaussianProcessPrior(kernel, nx=10000)
>>> reduced = ReducedPriorDesign(prior, n_components=50, method='kle')
>>> design, history, Sigma_post = reduced.run_design(...)
"""

from .reduced_design import ReducedPriorDesign
from .accelerated_forward import PODForwardModel
from .hybrid_inference import HybridBayesianInference

__all__ = [
    'ReducedPriorDesign',
    'PODForwardModel',
    'HybridBayesianInference',
]
