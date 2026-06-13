"""Candidate libraries for Burgers identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weak_sindy_robust.differentiators import FourierSpectral1D, FourierSpectral2D


@dataclass(frozen=True)
class LibraryTerm:
    """Metadata for a candidate term."""

    name: str
    index: int
    derivative_order: int
    dimension_tag: str | None = None
    active_in_truth: bool = False


class BurgersBaseLibrary:
    """Base library [1, u, u2, ux, uux, uxx]."""

    terms = [
        LibraryTerm("1", 0, derivative_order=0, active_in_truth=False),
        LibraryTerm("u", 1, derivative_order=0, active_in_truth=False),
        LibraryTerm("u2", 2, derivative_order=0, active_in_truth=False),
        LibraryTerm("ux", 3, derivative_order=1, active_in_truth=False),
        LibraryTerm("uux", 4, derivative_order=1, active_in_truth=True),
        LibraryTerm("uxx", 5, derivative_order=2, active_in_truth=True),
    ]

    @property
    def name(self) -> str:
        """Return the library name."""
        return "burgers_base"

    @property
    def names(self) -> list[str]:
        """Return candidate term names."""
        return [term.name for term in self.terms]

    def build_strong(self, U: np.ndarray, diff: FourierSpectral1D) -> np.ndarray:
        """Build the strong-form design matrix."""
        ux = diff.dx_field(U)
        uxx = diff.dxx_field(U)
        return np.column_stack(
            [
                np.ones(U.size),
                U.ravel(),
                (U**2).ravel(),
                ux.ravel(),
                (U * ux).ravel(),
                uxx.ravel(),
            ]
        )


class BurgersLargeLibrary:
    """Large Burgers library with distractor terms."""

    terms = [
        "1",
        "u",
        "u2",
        "u3",
        "ux",
        "uux",
        "u2ux",
        "uxx",
        "uuxx",
        "ux2",
        "uxxx",
        "uxxxx",
    ]

    @property
    def name(self) -> str:
        """Return the library name."""
        return "burgers_large"

    @property
    def names(self) -> list[str]:
        """Return candidate term names."""
        return list(self.terms)

    def build_strong(self, U: np.ndarray, diff: FourierSpectral1D) -> np.ndarray:
        """Build the strong-form design matrix."""
        ux = diff.dx_field(U)
        uxx = diff.dxx_field(U)
        uxxx = diff.derivative_field(U, 3)
        uxxxx = diff.derivative_field(U, 4)
        return np.column_stack(
            [
                np.ones(U.size),
                U.ravel(),
                (U**2).ravel(),
                (U**3).ravel(),
                ux.ravel(),
                (U * ux).ravel(),
                ((U**2) * ux).ravel(),
                uxx.ravel(),
                (U * uxx).ravel(),
                (ux**2).ravel(),
                uxxx.ravel(),
                uxxxx.ravel(),
            ]
        )


class NS2DVorticityLibrary:
    """Minimal vorticity library for 2D incompressible Navier-Stokes."""

    terms = [
        "1",
        "omega",
        "omega_x",
        "omega_y",
        "uomega_x",
        "vomega_y",
        "lap_omega",
        "omega2",
        "omegaomega_x",
        "omegaomega_y",
    ]

    @property
    def name(self) -> str:
        """Return the library name."""
        return "ns2d_vorticity_minimal"

    @property
    def names(self) -> list[str]:
        """Return candidate term names."""
        return list(self.terms)

    def term_fields(self, omega: np.ndarray, diff: FourierSpectral2D) -> list[np.ndarray]:
        """Return candidate fields on the same spacetime grid as omega."""
        omega_x = diff.dx_field(omega)
        omega_y = diff.dy_field(omega)
        lap_omega = diff.laplacian_field(omega)
        u, v = diff.velocity_from_vorticity(omega)
        return [
            np.ones_like(omega),
            omega,
            omega_x,
            omega_y,
            u * omega_x,
            v * omega_y,
            lap_omega,
            omega**2,
            omega * omega_x,
            omega * omega_y,
        ]

    def build_strong(self, U: np.ndarray, diff: FourierSpectral2D) -> np.ndarray:
        """Build the strong-form design matrix."""
        return np.column_stack([field.ravel() for field in self.term_fields(U, diff)])


def library_from_config(config: dict):
    """Create a library object from configuration."""
    name = config.get("library", {}).get("name", "burgers_base")
    if name == "burgers_base":
        return BurgersBaseLibrary()
    if name == "burgers_large":
        return BurgersLargeLibrary()
    if name == "ns2d_vorticity_minimal":
        return NS2DVorticityLibrary()
    raise ValueError(f"Unknown library: {name}")
