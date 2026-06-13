"""Tests for weak-form assembly."""

from __future__ import annotations

from weak_sindy_robust.libraries import BurgersBaseLibrary, BurgersLargeLibrary
from weak_sindy_robust.problems import Burgers1DSpec
from weak_sindy_robust.solvers import BurgersRK4Solver
from weak_sindy_robust.test_functions import TensorProductTestFunctionConfig
from weak_sindy_robust.weak_assembly import assemble_weak_system


def test_weak_assembly_shape() -> None:
    """Base weak assembly has 14*10 rows and six columns."""
    spec = Burgers1DSpec()
    data = BurgersRK4Solver(spec).solve()
    system = assemble_weak_system(
        data.U_clean,
        spec.grid,
        BurgersBaseLibrary(),
        TensorProductTestFunctionConfig(nxc=14, ntc=10),
    )
    assert system.A.shape == (140, 6)
    assert system.b.shape == (140,)


def test_weak_large_library_assembly_shape() -> None:
    """Large-library weak assembly has 14*10 rows and twelve columns."""
    spec = Burgers1DSpec()
    data = BurgersRK4Solver(spec).solve()
    system = assemble_weak_system(
        data.U_clean,
        spec.grid,
        BurgersLargeLibrary(),
        TensorProductTestFunctionConfig(nxc=14, ntc=10),
    )
    assert system.A.shape == (140, 12)
    assert system.b.shape == (140,)
    assert system.metadata["weak_variant"] == "integrated_by_parts_large_library"
