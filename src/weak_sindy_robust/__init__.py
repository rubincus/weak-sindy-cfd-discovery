"""Modular weak/strong SINDy tools for noisy Burgers experiments."""

from weak_sindy_robust.grids import Dataset1D, Grid1D
from weak_sindy_robust.problems import Burgers1DSpec
from weak_sindy_robust.solvers import BurgersRK4Solver

__all__ = [
    "Burgers1DSpec",
    "BurgersRK4Solver",
    "Dataset1D",
    "Grid1D",
]

