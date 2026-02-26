"""
Linear Dimensionality Reduction Methods
========================================

Methods based on linear projections and eigendecomposition.

Classes:
--------
- PCA: Principal Component Analysis
- KLE: Karhunen-Loève Expansion (optimal for GPs)
- POD: Proper Orthogonal Decomposition (for PDE snapshots)

Usage:
------
>>> from boed.reduction.linear import PCA, KLE, POD
>>> 
>>> # PCA for tabular data
>>> pca = PCA(n_components=10)
>>> pca.fit(data)
>>> data_reduced = pca.transform(data)
>>> 
>>> # KLE for continuous random fields
>>> kle = KLE(domain=[0, 1], n_points=200)
>>> kle.fit(covariance_function, n_modes=20)
>>> samples = kle.sample(n_samples=100)
>>> 
>>> # POD for PDE snapshots
>>> pod = POD(energy_threshold=0.9999)
>>> pod.fit(snapshots, use_snapshot_method=True)
>>> basis = pod.basis_
"""

from .base import DimensionalityReductionBase
from .pca import PCA
from .kle import KLE
from .pod import POD

__all__ = [
    'DimensionalityReductionBase',
    'PCA',
    'KLE',
    'POD',
]
