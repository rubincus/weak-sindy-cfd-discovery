"""Tests for Fourier spectral derivatives."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.differentiators import FourierSpectral1D
from weak_sindy_robust.grids import Grid1D


def test_fourier_derivatives_on_sine() -> None:
    """d/dx and d2/dx2 reproduce exact trigonometric derivatives."""
    grid = Grid1D(L=2.0 * np.pi, Nx=128, T=1.0, Nt=2)
    diff = FourierSpectral1D(grid)
    k = 3
    u = np.sin(k * grid.x)
    assert np.max(np.abs(diff.dx(u) - k * np.cos(k * grid.x))) < 1.0e-10
    assert np.max(np.abs(diff.dxx(u) + (k**2) * np.sin(k * grid.x))) < 1.0e-9

