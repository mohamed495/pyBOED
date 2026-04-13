"""PDE solvers for forward models."""

from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
from boed.pde.shallow_water import ShallowWater1D_CN
from boed.pde.burgers import Burgers_CN

__all__ = ["AdvectionDiffusion1D_CN","ShallowWater1D_CN", "Burgers_CN"]
