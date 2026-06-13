"""Problem specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weak_sindy_robust.grids import Grid1D, Grid2D


@dataclass(frozen=True)
class Burgers1DSpec:
    """Reference Burgers 1D benchmark specification."""

    nu: float = 0.075
    L: float = 2.0 * np.pi
    Nx: int = 128
    Nt: int = 241
    T: float = 1.20

    @property
    def grid(self) -> Grid1D:
        """Return the corresponding periodic grid."""
        return Grid1D(L=self.L, Nx=self.Nx, T=self.T, Nt=self.Nt, periodic=True)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Return the trigonometric reference initial condition."""
        return (
            0.90 * np.sin(x)
            + 0.35 * np.sin(2.0 * x + 0.30)
            - 0.18 * np.cos(3.0 * x - 0.10)
        )

    @property
    def true_coefficients(self) -> np.ndarray:
        """Return coefficients for [1, u, u2, ux, uux, uxx]."""
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0, self.nu], dtype=float)

    @property
    def true_support(self) -> set[int]:
        """Return the true Python-indexed support."""
        return {4, 5}


def burgers_spec_from_config(config: dict) -> Burgers1DSpec:
    """Build a Burgers specification from a resolved configuration dictionary."""
    problem = config.get("problem", {})
    grid = problem.get("grid", {})
    domain = problem.get("domain", {})
    return Burgers1DSpec(
        nu=float(problem.get("nu", 0.075)),
        L=float(domain.get("L", 2.0 * np.pi)),
        Nx=int(grid.get("Nx", 128)),
        Nt=int(grid.get("Nt", 241)),
        T=float(grid.get("T", 1.20)),
    )


def large_library_true_coefficients(nu: float = 0.075) -> np.ndarray:
    """Return true coefficients for the configured large Burgers library."""
    xi = np.zeros(12, dtype=float)
    xi[5] = -1.0
    xi[7] = nu
    return xi


@dataclass(frozen=True)
class NS2DVorticitySpec:
    """Reference 2D incompressible Navier-Stokes vorticity benchmark."""

    nu: float = 0.001
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi
    Nx: int = 64
    Ny: int = 64
    Nt: int = 101
    T: float = 1.0
    dealiasing: bool = True

    @property
    def grid(self) -> Grid2D:
        """Return the corresponding periodic grid."""
        return Grid2D(
            Lx=self.Lx,
            Ly=self.Ly,
            Nx=self.Nx,
            Ny=self.Ny,
            T=self.T,
            Nt=self.Nt,
            periodic=True,
        )

    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return a multimode vorticity field with nonzero nonlinear advection."""
        X, Y = np.meshgrid(x, y)
        omega = (
            1.20 * np.cos(X)
            + 0.80 * np.sin(Y)
            + 0.55 * np.cos(2.0 * X + Y)
            - 0.45 * np.sin(X - 2.0 * Y)
            + 0.25 * np.cos(3.0 * X - 2.0 * Y + 0.20)
        )
        return omega - np.mean(omega)

    @property
    def true_coefficients(self) -> np.ndarray:
        """Return coefficients for the configured vorticity library."""
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, self.nu, 0.0, 0.0, 0.0], dtype=float)

    @property
    def true_support(self) -> set[int]:
        """Return the true Python-indexed support."""
        return {4, 5, 6}


def ns2d_spec_from_config(config: dict) -> NS2DVorticitySpec:
    """Build a 2D vorticity specification from a resolved configuration dictionary."""
    problem = config.get("problem", {})
    grid = problem.get("grid", {})
    domain = problem.get("domain", {})
    solver = config.get("solver", {})
    return NS2DVorticitySpec(
        nu=float(problem.get("nu", 0.001)),
        Lx=float(domain.get("Lx", 2.0 * np.pi)),
        Ly=float(domain.get("Ly", 2.0 * np.pi)),
        Nx=int(grid.get("Nx", 64)),
        Ny=int(grid.get("Ny", 64)),
        Nt=int(grid.get("Nt", 101)),
        T=float(grid.get("T", 1.0)),
        dealiasing=bool(solver.get("dealiasing", True)),
    )
