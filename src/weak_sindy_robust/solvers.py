"""Reference solvers."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.differentiators import FourierSpectral1D, FourierSpectral2D
from weak_sindy_robust.grids import Dataset1D, Dataset2D
from weak_sindy_robust.problems import Burgers1DSpec, NS2DVorticitySpec


class BurgersRK4Solver:
    """Deterministic RK4 solver for the Burgers 1D benchmark."""

    def __init__(self, spec: Burgers1DSpec):
        """Create a solver for a Burgers specification."""
        self.spec = spec
        self.grid = spec.grid
        self.diff = FourierSpectral1D(self.grid)

    def rhs(self, u: np.ndarray) -> np.ndarray:
        """Evaluate u_t = -u*u_x + nu*u_xx."""
        ux = self.diff.dx(u)
        uxx = self.diff.dxx(u)
        return -u * ux + self.spec.nu * uxx

    def solve(self) -> Dataset1D:
        """Integrate the reference trajectory."""
        x = self.grid.x
        t = self.grid.t
        dt = self.grid.dt

        U = np.zeros((self.grid.Nt, self.grid.Nx), dtype=float)
        U[0] = self.spec.initial_condition(x)

        for n in range(self.grid.Nt - 1):
            u = U[n]
            k1 = self.rhs(u)
            k2 = self.rhs(u + 0.5 * dt * k1)
            k3 = self.rhs(u + 0.5 * dt * k2)
            k4 = self.rhs(u + dt * k3)
            U[n + 1] = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        data = Dataset1D(
            U_clean=U,
            U_noisy=None,
            x=x,
            t=t,
            metadata={
                "problem": "burgers_1d",
                "nu": self.spec.nu,
                "Nx": self.grid.Nx,
                "Nt": self.grid.Nt,
                "T": self.grid.T,
                "L": self.grid.L,
                "solver": "rk4_fourier_spectral",
            },
        )
        data.validate()
        return data


class NS2DVorticityRK4Solver:
    """Pseudo-spectral RK4 solver for 2D Navier-Stokes in vorticity form."""

    def __init__(self, spec: NS2DVorticitySpec):
        """Create a solver for a 2D vorticity specification."""
        self.spec = spec
        self.grid = spec.grid
        self.diff = FourierSpectral2D(self.grid)

    def rhs(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate omega_t = -u omega_x - v omega_y + nu Laplacian(omega)."""
        u, v = self.diff.velocity_from_vorticity(omega)
        omega_x = self.diff.dx_field(omega)
        omega_y = self.diff.dy_field(omega)
        lap_omega = self.diff.laplacian_field(omega)
        return -(u * omega_x + v * omega_y) + self.spec.nu * lap_omega

    def _post_step(self, omega: np.ndarray) -> np.ndarray:
        """Remove the mean and optionally dealias after a time step."""
        out = omega - np.mean(omega)
        if self.spec.dealiasing:
            out = self.diff.dealias(out)
        return out

    def solve(self) -> Dataset2D:
        """Integrate the reference vorticity trajectory."""
        x = self.grid.x
        y = self.grid.y
        t = self.grid.t
        dt = self.grid.dt

        W = np.zeros((self.grid.Nt, self.grid.Ny, self.grid.Nx), dtype=float)
        W[0] = self._post_step(self.spec.initial_condition(x, y))

        for n in range(self.grid.Nt - 1):
            omega = W[n]
            k1 = self.rhs(omega)
            k2 = self.rhs(omega + 0.5 * dt * k1)
            k3 = self.rhs(omega + 0.5 * dt * k2)
            k4 = self.rhs(omega + dt * k3)
            W[n + 1] = self._post_step(omega + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        data = Dataset2D(
            U_clean=W,
            U_noisy=None,
            x=x,
            y=y,
            t=t,
            metadata={
                "problem": "ns2d_vorticity_minimal",
                "field": "vorticity",
                "nu": self.spec.nu,
                "Nx": self.grid.Nx,
                "Ny": self.grid.Ny,
                "Nt": self.grid.Nt,
                "T": self.grid.T,
                "Lx": self.grid.Lx,
                "Ly": self.grid.Ly,
                "solver": "rk4_fourier_pseudospectral",
                "dealiasing": self.spec.dealiasing,
            },
        )
        data.validate()
        return data
