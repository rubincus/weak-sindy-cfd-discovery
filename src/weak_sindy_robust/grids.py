"""Grid and dataset containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Grid1D:
    """Uniform one-dimensional periodic grid."""

    L: float
    Nx: int
    T: float
    Nt: int
    periodic: bool = True

    def __post_init__(self) -> None:
        """Validate grid parameters."""
        if self.L <= 0.0:
            raise ValueError("L must be positive.")
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.Nx < 2:
            raise ValueError("Nx must be at least 2.")
        if self.Nt < 2:
            raise ValueError("Nt must be at least 2.")

    @property
    def x(self) -> np.ndarray:
        """Return spatial nodes with endpoint excluded."""
        return np.linspace(0.0, self.L, self.Nx, endpoint=False)

    @property
    def t(self) -> np.ndarray:
        """Return temporal nodes including the final time."""
        return np.linspace(0.0, self.T, self.Nt)

    @property
    def dx(self) -> float:
        """Return spatial grid spacing."""
        return self.L / self.Nx

    @property
    def dt(self) -> float:
        """Return temporal grid spacing."""
        return self.T / (self.Nt - 1)


@dataclass
class Dataset1D:
    """Clean/noisy one-dimensional field and metadata."""

    U_clean: np.ndarray
    U_noisy: np.ndarray | None
    x: np.ndarray
    t: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        """Validate array shapes and finiteness."""
        if self.U_clean.ndim != 2:
            raise ValueError("U_clean must be two-dimensional.")
        if self.U_clean.shape != (len(self.t), len(self.x)):
            raise ValueError("U_clean shape must be (len(t), len(x)).")
        if not np.isfinite(self.U_clean).all():
            raise ValueError("U_clean contains non-finite values.")
        if self.U_noisy is not None and self.U_noisy.shape != self.U_clean.shape:
            raise ValueError("U_noisy must have the same shape as U_clean.")

    @property
    def observed(self) -> np.ndarray:
        """Return noisy data when present, otherwise clean data."""
        return self.U_clean if self.U_noisy is None else self.U_noisy


@dataclass(frozen=True)
class Grid2D:
    """Uniform two-dimensional periodic grid with time samples."""

    Lx: float
    Ly: float
    Nx: int
    Ny: int
    T: float
    Nt: int
    periodic: bool = True

    def __post_init__(self) -> None:
        """Validate grid parameters."""
        if self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError("Lx and Ly must be positive.")
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Nx and Ny must be at least 2.")
        if self.Nt < 2:
            raise ValueError("Nt must be at least 2.")

    @property
    def x(self) -> np.ndarray:
        """Return x nodes with endpoint excluded."""
        return np.linspace(0.0, self.Lx, self.Nx, endpoint=False)

    @property
    def y(self) -> np.ndarray:
        """Return y nodes with endpoint excluded."""
        return np.linspace(0.0, self.Ly, self.Ny, endpoint=False)

    @property
    def t(self) -> np.ndarray:
        """Return temporal nodes including the final time."""
        return np.linspace(0.0, self.T, self.Nt)

    @property
    def dx(self) -> float:
        """Return x-grid spacing."""
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        """Return y-grid spacing."""
        return self.Ly / self.Ny

    @property
    def dt(self) -> float:
        """Return temporal grid spacing."""
        return self.T / (self.Nt - 1)


@dataclass
class Dataset2D:
    """Clean/noisy two-dimensional field and metadata."""

    U_clean: np.ndarray
    U_noisy: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        """Validate array shapes and finiteness."""
        if self.U_clean.ndim != 3:
            raise ValueError("U_clean must be three-dimensional.")
        if self.U_clean.shape != (len(self.t), len(self.y), len(self.x)):
            raise ValueError("U_clean shape must be (len(t), len(y), len(x)).")
        if not np.isfinite(self.U_clean).all():
            raise ValueError("U_clean contains non-finite values.")
        if self.U_noisy is not None and self.U_noisy.shape != self.U_clean.shape:
            raise ValueError("U_noisy must have the same shape as U_clean.")

    @property
    def observed(self) -> np.ndarray:
        """Return noisy data when present, otherwise clean data."""
        return self.U_clean if self.U_noisy is None else self.U_noisy
