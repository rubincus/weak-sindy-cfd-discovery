"""Weak-form assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weak_sindy_robust.differentiators import FourierSpectral1D, FourierSpectral2D
from weak_sindy_robust.grids import Grid1D, Grid2D
from weak_sindy_robust.libraries import BurgersBaseLibrary, BurgersLargeLibrary, NS2DVorticityLibrary
from weak_sindy_robust.test_functions import (
    CompactCosineTestFunctionFamily,
    TensorProductTestFunctionConfig,
    compact_cosine,
    compact_cosine_d1,
)


@dataclass
class LinearSystem:
    """Linear system A xi = b plus feature metadata."""

    A: np.ndarray
    b: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]


def assemble_weak_system(
    U: np.ndarray,
    grid: Grid1D | Grid2D,
    library: BurgersBaseLibrary | BurgersLargeLibrary | NS2DVorticityLibrary,
    test_cfg: TensorProductTestFunctionConfig,
) -> LinearSystem:
    """Assemble a weak-form system for a supported library."""
    if isinstance(grid, Grid2D):
        return assemble_weak_ns2d_system(U, grid, library, test_cfg)
    if library.names == ["1", "u", "u2", "ux", "uux", "uxx"]:
        return assemble_weak_burgers_base_system(U, grid, library, test_cfg)
    if library.names == [
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
    ]:
        return assemble_weak_burgers_large_system(U, grid, library, test_cfg)
    raise NotImplementedError("Weak assembly is implemented for burgers_base, burgers_large, and ns2d_vorticity_minimal.")


def assemble_weak_burgers_base_system(
    U: np.ndarray,
    grid: Grid1D,
    library: BurgersBaseLibrary,
    test_cfg: TensorProductTestFunctionConfig,
) -> LinearSystem:
    """Assemble the weak-form system for the base Burgers library."""
    if library.names != ["1", "u", "u2", "ux", "uux", "uxx"]:
        raise NotImplementedError("Weak assembly is implemented for burgers_base only.")

    family = CompactCosineTestFunctionFamily(grid, test_cfg)
    xc, tc = family.centers()
    rows_G: list[list[float]] = []
    rows_b: list[float] = []
    U2 = U**2
    w = grid.dx * grid.dt

    for tm in tc:
        for xm in xc:
            evals = family.evaluate_all_for_center(xm=xm, tm=tm)
            psi = evals["psi"]
            psi_t = evals["psi_t"]
            psi_x = evals["psi_x"]
            psi_xx = evals["psi_xx"]

            b = -np.sum(U * psi_t) * w
            g1 = np.sum(psi) * w
            g2 = np.sum(U * psi) * w
            g3 = np.sum(U2 * psi) * w
            g4 = -np.sum(U * psi_x) * w
            g5 = -0.5 * np.sum(U2 * psi_x) * w
            g6 = np.sum(U * psi_xx) * w
            rows_G.append([g1, g2, g3, g4, g5, g6])
            rows_b.append(float(b))

    return LinearSystem(
        A=np.array(rows_G, dtype=float),
        b=np.array(rows_b, dtype=float),
        feature_names=library.names,
        metadata={
            "formulation": "weak",
            "n_test_functions": int(len(rows_b)),
            "ax": test_cfg.ax,
            "at": test_cfg.at,
            "nxc": test_cfg.nxc,
            "ntc": test_cfg.ntc,
        },
    )


def assemble_weak_burgers_large_system(
    U: np.ndarray,
    grid: Grid1D,
    library: BurgersLargeLibrary,
    test_cfg: TensorProductTestFunctionConfig,
) -> LinearSystem:
    """Assemble the weak-form system for the enlarged Burgers library."""
    if library.names != [
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
    ]:
        raise NotImplementedError("Weak large-library assembly is implemented for burgers_large only.")

    family = CompactCosineTestFunctionFamily(grid, test_cfg)
    xc, tc = family.centers()
    rows_G: list[list[float]] = []
    rows_b: list[float] = []
    U2 = U**2
    U3 = U**3
    ux = FourierSpectral1D(grid).dx_field(U)
    ux2 = ux**2
    w = grid.dx * grid.dt

    for tm in tc:
        for xm in xc:
            evals = family.evaluate_all_for_center(xm=xm, tm=tm)
            psi = evals["psi"]
            psi_t = evals["psi_t"]
            psi_x = evals["psi_x"]
            psi_xx = evals["psi_xx"]
            psi_xxx = evals["psi_xxx"]
            psi_xxxx = evals["psi_xxxx"]

            b = -np.sum(U * psi_t) * w
            g1 = np.sum(psi) * w
            g2 = np.sum(U * psi) * w
            g3 = np.sum(U2 * psi) * w
            g4 = np.sum(U3 * psi) * w
            g5 = -np.sum(U * psi_x) * w
            g6 = -0.5 * np.sum(U2 * psi_x) * w
            g7 = -(1.0 / 3.0) * np.sum(U3 * psi_x) * w
            g8 = np.sum(U * psi_xx) * w
            g9 = (0.5 * np.sum(U2 * psi_xx) - np.sum(ux2 * psi)) * w
            g10 = np.sum(ux2 * psi) * w
            g11 = -np.sum(U * psi_xxx) * w
            g12 = np.sum(U * psi_xxxx) * w
            rows_G.append([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12])
            rows_b.append(float(b))

    return LinearSystem(
        A=np.array(rows_G, dtype=float),
        b=np.array(rows_b, dtype=float),
        feature_names=library.names,
        metadata={
            "formulation": "weak",
            "weak_variant": "integrated_by_parts_large_library",
            "n_test_functions": int(len(rows_b)),
            "ax": test_cfg.ax,
            "at": test_cfg.at,
            "nxc": test_cfg.nxc,
            "ntc": test_cfg.ntc,
            "auxiliary_derivative_terms": "ux2,uuxx",
        },
    )


def _periodic_offsets(nodes: np.ndarray, center: float, length: float) -> np.ndarray:
    """Return shortest periodic offsets from a center."""
    return ((nodes - center + 0.5 * length) % length) - 0.5 * length


def assemble_weak_ns2d_system(
    omega: np.ndarray,
    grid: Grid2D,
    library: BurgersBaseLibrary | BurgersLargeLibrary | NS2DVorticityLibrary,
    test_cfg: TensorProductTestFunctionConfig,
) -> LinearSystem:
    """Assemble a local weak-time system for the 2D vorticity benchmark."""
    if library.names != [
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
    ]:
        raise NotImplementedError("2D weak assembly is implemented for ns2d_vorticity_minimal only.")

    ay = float(test_cfg.ay if test_cfg.ay is not None else test_cfg.ax)
    nyc = int(test_cfg.nyc if test_cfg.nyc is not None else test_cfg.nxc)
    xc = np.linspace(0.0, grid.Lx, test_cfg.nxc, endpoint=False)
    yc = np.linspace(0.0, grid.Ly, nyc, endpoint=False)
    tc = np.linspace(test_cfg.at, grid.T - test_cfg.at, test_cfg.ntc)

    diff = FourierSpectral2D(grid)
    term_fields = library.term_fields(omega, diff)
    rows_A: list[list[float]] = []
    rows_b: list[float] = []
    weight = grid.dx * grid.dy * grid.dt

    t_offsets_cache = {}
    x_weights_cache = {}
    y_weights_cache = {}

    for tm in tc:
        if tm not in t_offsets_cache:
            st = grid.t - tm
            t_offsets_cache[tm] = (
                compact_cosine(st, test_cfg.at)[:, None, None],
                compact_cosine_d1(st, test_cfg.at)[:, None, None],
            )
        pt, pt_t = t_offsets_cache[tm]
        for ym in yc:
            if ym not in y_weights_cache:
                sy = _periodic_offsets(grid.y, ym, grid.Ly)
                y_weights_cache[ym] = compact_cosine(sy, ay)[None, :, None]
            py = y_weights_cache[ym]
            for xm in xc:
                if xm not in x_weights_cache:
                    sx = _periodic_offsets(grid.x, xm, grid.Lx)
                    x_weights_cache[xm] = compact_cosine(sx, test_cfg.ax)[None, None, :]
                px = x_weights_cache[xm]
                psi = pt * py * px
                psi_t = pt_t * py * px
                rows_b.append(float(-np.sum(omega * psi_t) * weight))
                rows_A.append([float(np.sum(field * psi) * weight) for field in term_fields])

    return LinearSystem(
        A=np.array(rows_A, dtype=float),
        b=np.array(rows_b, dtype=float),
        feature_names=library.names,
        metadata={
            "formulation": "weak",
            "weak_variant": "local_window_time_integration",
            "n_test_functions": int(len(rows_b)),
            "ax": test_cfg.ax,
            "ay": ay,
            "at": test_cfg.at,
            "nxc": test_cfg.nxc,
            "nyc": nyc,
            "ntc": test_cfg.ntc,
        },
    )
