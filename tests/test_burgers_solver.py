"""Tests for the Burgers reference solver."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.metrics import kinetic_energy
from weak_sindy_robust.problems import Burgers1DSpec
from weak_sindy_robust.solvers import BurgersRK4Solver


def test_burgers_solver_shape_and_energy() -> None:
    """The reference solver returns finite data and dissipates energy."""
    spec = Burgers1DSpec()
    data = BurgersRK4Solver(spec).solve()
    assert data.U_clean.shape == (241, 128)
    assert np.isfinite(data.U_clean).all()
    energy = kinetic_energy(data.U_clean, spec.grid.dx)
    assert np.all(np.diff(energy) <= 1.0e-8)

