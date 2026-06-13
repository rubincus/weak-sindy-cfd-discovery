"""Weak-form test functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weak_sindy_robust.grids import Grid1D


def compact_cosine(s: np.ndarray, a: float) -> np.ndarray:
    """Evaluate the compact cosine kernel."""
    out = np.zeros_like(s, dtype=float)
    mask = np.abs(s) <= a
    out[mask] = 0.5 * (1.0 + np.cos(np.pi * s[mask] / a))
    return out


def compact_cosine_d1(s: np.ndarray, a: float) -> np.ndarray:
    """Evaluate the first derivative of the compact cosine kernel."""
    out = np.zeros_like(s, dtype=float)
    mask = np.abs(s) <= a
    out[mask] = -0.5 * np.pi / a * np.sin(np.pi * s[mask] / a)
    return out


def compact_cosine_d2(s: np.ndarray, a: float) -> np.ndarray:
    """Evaluate the second derivative of the compact cosine kernel."""
    out = np.zeros_like(s, dtype=float)
    mask = np.abs(s) <= a
    out[mask] = -0.5 * (np.pi / a) ** 2 * np.cos(np.pi * s[mask] / a)
    return out


def compact_cosine_d3(s: np.ndarray, a: float) -> np.ndarray:
    """Evaluate the third derivative of the compact cosine kernel."""
    out = np.zeros_like(s, dtype=float)
    mask = np.abs(s) <= a
    out[mask] = 0.5 * (np.pi / a) ** 3 * np.sin(np.pi * s[mask] / a)
    return out


def compact_cosine_d4(s: np.ndarray, a: float) -> np.ndarray:
    """Evaluate the fourth derivative of the compact cosine kernel."""
    out = np.zeros_like(s, dtype=float)
    mask = np.abs(s) <= a
    out[mask] = 0.5 * (np.pi / a) ** 4 * np.cos(np.pi * s[mask] / a)
    return out


@dataclass(frozen=True)
class TensorProductTestFunctionConfig:
    """Configuration for tensor-product compact test functions."""

    ax: float = 0.90
    ay: float | None = None
    at: float = 0.22
    nxc: int = 14
    nyc: int | None = None
    ntc: int = 10
    family: str = "compact_cosine"
    center_policy: str = "interior_uniform"


class CompactCosineTestFunctionFamily:
    """Tensor-product compact cosine test functions."""

    def __init__(self, grid: Grid1D, cfg: TensorProductTestFunctionConfig):
        """Create the family on a grid."""
        if cfg.family != "compact_cosine":
            raise ValueError("Only compact_cosine is implemented.")
        if cfg.center_policy != "interior_uniform":
            raise ValueError("Only interior_uniform centers are implemented.")
        self.grid = grid
        self.cfg = cfg

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return spatial and temporal centers matching the original script."""
        xc = np.linspace(self.cfg.ax, self.grid.L - self.cfg.ax, self.cfg.nxc)
        tc = np.linspace(self.cfg.at, self.grid.T - self.cfg.at, self.cfg.ntc)
        return xc, tc

    def evaluate_all_for_center(self, xm: float, tm: float) -> dict[str, np.ndarray]:
        """Evaluate psi and derivatives for one center."""
        x = self.grid.x
        t = self.grid.t
        Xg, Tg = np.meshgrid(x, t)

        sx = Xg - xm
        st = Tg - tm

        px = compact_cosine(sx, self.cfg.ax)
        px_x = compact_cosine_d1(sx, self.cfg.ax)
        px_xx = compact_cosine_d2(sx, self.cfg.ax)
        px_xxx = compact_cosine_d3(sx, self.cfg.ax)
        px_xxxx = compact_cosine_d4(sx, self.cfg.ax)

        pt = compact_cosine(st, self.cfg.at)
        pt_t = compact_cosine_d1(st, self.cfg.at)

        return {
            "psi": px * pt,
            "psi_t": px * pt_t,
            "psi_x": px_x * pt,
            "psi_xx": px_xx * pt,
            "psi_xxx": px_xxx * pt,
            "psi_xxxx": px_xxxx * pt,
        }


def test_function_config_from_method(method_cfg: dict) -> TensorProductTestFunctionConfig:
    """Build a weak test-function config from a method dictionary."""
    cfg = method_cfg.get("test_functions", {})
    return TensorProductTestFunctionConfig(
        ax=float(cfg.get("ax", 0.90)),
        ay=float(cfg["ay"]) if cfg.get("ay") is not None else None,
        at=float(cfg.get("at", 0.22)),
        nxc=int(cfg.get("nxc", 14)),
        nyc=int(cfg["nyc"]) if cfg.get("nyc") is not None else None,
        ntc=int(cfg.get("ntc", 10)),
        family=cfg.get("family", "compact_cosine"),
        center_policy=cfg.get("center_policy", "interior_uniform"),
    )
