"""PDE solvers for forward models."""

from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.pde.shallow_water import ShallowWater1D_CN
from boed.pde.burgers import BurgersNonLinear_CN

__all__ = ["AdvectionDiffusion1D_CN","ShallowWater1D_CN", "BurgersNonLinear_CN", "BurgersLinear_CN"]
