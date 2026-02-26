"""Core module: base classes, priors, noise models, and validation utilities."""

from boed.core.base import (
    ForwardModelBase,
    KernelBase,
    NoiseModelBase,
    validate_design_indices,
    validate_positive_definite,
)
from boed.core.noise import ColoredNoise, HeteroscedasticNoise, NoiseModel
from boed.core.initial_conditions import (
    U0Spec,
    available_u0,
    get_u0_spec,
    make_u0,
    make_u0_bank,
    u0_registry,
)
from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.priors.gp_priors import GaussianProcessPrior
from boed.priors.kernels import (
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    RationalQuadratic,
    Gaussian,
)


ForwardModel = ForwardModelBase
Kernel = KernelBase

__all__ = [
    "ForwardModelBase",
    "KernelBase",
    "NoiseModelBase",
    "ForwardModel",
    "Kernel",
    "AdvectionDiffusion1D_CN",
    "GaussianProcessPrior",
    "Gaussian",
    "Matern12",
    "Matern32",
    "Matern52",
    "RationalQuadratic",
    "Periodic",
    "NoiseModel",
    "ColoredNoise",
    "HeteroscedasticNoise",
    "U0Spec",
    "available_u0",
    "get_u0_spec",
    "make_u0",
    "make_u0_bank",
    "u0_registry",
    "validate_positive_definite",
    "validate_design_indices",
]
