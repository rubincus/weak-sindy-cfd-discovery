"""Model rollouts for identified Burgers libraries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weak_sindy_robust.differentiators import FourierSpectral1D
from weak_sindy_robust.grids import Grid1D
from weak_sindy_robust.metrics import rmse
from weak_sindy_robust.problems import Burgers1DSpec
from weak_sindy_robust.solvers import BurgersRK4Solver


@dataclass
class RolloutResult:
    """Rollout field and diagnostics."""

    U: np.ndarray
    reference: np.ndarray | None
    rmse: float
    stable: bool
    grid: Grid1D
    metadata: dict


def _term_value(term: str, u: np.ndarray, diff: FourierSpectral1D) -> np.ndarray:
    """Evaluate one library term for a state."""
    ux = None
    uxx = None
    if term in {"ux", "uux", "u2ux", "ux2"}:
        ux = diff.dx(u)
    if term in {"uxx", "uuxx"}:
        uxx = diff.dxx(u)

    if term == "1":
        return np.ones_like(u)
    if term == "u":
        return u
    if term == "u2":
        return u**2
    if term == "u3":
        return u**3
    if term == "ux":
        return ux if ux is not None else diff.dx(u)
    if term == "uux":
        ux = ux if ux is not None else diff.dx(u)
        return u * ux
    if term == "u2ux":
        ux = ux if ux is not None else diff.dx(u)
        return (u**2) * ux
    if term == "uxx":
        return uxx if uxx is not None else diff.dxx(u)
    if term == "uuxx":
        uxx = uxx if uxx is not None else diff.dxx(u)
        return u * uxx
    if term == "ux2":
        ux = ux if ux is not None else diff.dx(u)
        return ux**2
    if term == "uxxx":
        return diff.derivative(u, 3)
    if term == "uxxxx":
        return diff.derivative(u, 4)
    raise ValueError(f"Unsupported rollout term: {term}")


def rollout_burgers(
    xi: np.ndarray,
    feature_names: list[str],
    spec: Burgers1DSpec,
    reference: np.ndarray | None = None,
    T_rollout: float | None = None,
    diffusion_policy: str = "raw",
    blowup_threshold: float = 50.0,
) -> RolloutResult:
    """Integrate an identified Burgers model without silent diffusion clipping."""
    if T_rollout is None:
        T_rollout = spec.T
    nt = int(round(T_rollout / spec.grid.dt)) + 1
    rollout_spec = Burgers1DSpec(nu=spec.nu, L=spec.L, Nx=spec.Nx, Nt=nt, T=float(T_rollout))
    grid = rollout_spec.grid
    diff = FourierSpectral1D(grid)

    if reference is None:
        if abs(T_rollout - spec.T) < 1.0e-14 and nt == spec.Nt:
            reference = None
        else:
            reference = BurgersRK4Solver(rollout_spec).solve().U_clean

    def rhs_model(u: np.ndarray) -> np.ndarray:
        rhs = np.zeros_like(u)
        for coeff, term in zip(xi, feature_names):
            c = float(coeff)
            if diffusion_policy == "positive" and term == "uxx":
                c = max(c, 0.0)
            rhs += c * _term_value(term, u, diff)
        return rhs

    U = np.zeros((grid.Nt, grid.Nx), dtype=float)
    U[0] = rollout_spec.initial_condition(grid.x)
    stable = True
    for n in range(grid.Nt - 1):
        u = U[n]
        try:
            k1 = rhs_model(u)
            k2 = rhs_model(u + 0.5 * grid.dt * k1)
            k3 = rhs_model(u + 0.5 * grid.dt * k2)
            k4 = rhs_model(u + grid.dt * k3)
            U[n + 1] = u + (grid.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        except FloatingPointError:
            stable = False
            U[n + 1 :] = np.nan
            break
        if not np.isfinite(U[n + 1]).all() or np.max(np.abs(U[n + 1])) > blowup_threshold:
            stable = False
            U[n + 1 :] = np.nan
            break

    if reference is not None and reference.shape == U.shape and np.isfinite(U).all():
        rollout_rmse = rmse(U, reference)
    else:
        rollout_rmse = float("nan")

    return RolloutResult(
        U=U,
        reference=reference,
        rmse=rollout_rmse,
        stable=stable,
        grid=grid,
        metadata={
            "T_rollout": float(T_rollout),
            "diffusion_policy": diffusion_policy,
            "blowup_threshold": float(blowup_threshold),
        },
    )

