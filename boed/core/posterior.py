"""Compatibility shim for legacy imports from ``boed.core.posterior``.

The inference implementations now live under ``boed.inference`` modules.
This module re-exports the legacy names to preserve backward compatibility.
"""
from boed.inference.base import InverseModel, _compress_operator
from boed.inference.linear_gaussian import LinearGaussianModel
from boed.inference.nonlinear_laplace import NonLinearInverseLaplace

__all__ = [
    "_compress_operator",
    "InverseModel",
    "LinearGaussianModel",
    "NonLinearInverseLaplace",
]
