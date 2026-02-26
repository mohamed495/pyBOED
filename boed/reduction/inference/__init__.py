"""Inference-oriented dimensionality reduction methods."""

from .active_subspace import ActiveSubspaces
from .lis import LikelihoodInformedSubspaces, DataAveragedLIS

__all__ = [
    "ActiveSubspaces",
    "LikelihoodInformedSubspaces",
    "DataAveragedLIS",
]
