"""Strong-form assembly and preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np

from weak_sindy_robust.differentiators import FourierSpectral1D, FourierSpectral2D
from weak_sindy_robust.grids import Grid1D, Grid2D
from weak_sindy_robust.weak_assembly import LinearSystem


def time_derivative_gradient(U: np.ndarray, grid: Grid1D) -> np.ndarray:
    """Estimate u_t with numpy.gradient to reproduce the original script."""
    return np.gradient(U, grid.dt, axis=0)


def smooth_temporal(U: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """Smooth along time using Savitzky-Golay when available, otherwise moving average."""
    window_length = int(max(3, window_length))
    if window_length % 2 == 0:
        window_length += 1
    try:
        from scipy.signal import savgol_filter  # type: ignore

        return savgol_filter(U, window_length=window_length, polyorder=polyorder, axis=0, mode="interp")
    except Exception:
        radius = window_length // 2
        pad_width = [(radius, radius)] + [(0, 0)] * (U.ndim - 1)
        padded = np.pad(U, pad_width, mode="edge")
        out = np.empty_like(U)
        for i in range(U.shape[0]):
            out[i] = np.mean(padded[i : i + window_length], axis=0)
        return out


def spectral_lowpass(U: np.ndarray, keep_fraction: float = 0.50) -> np.ndarray:
    """Apply a spatial Fourier low-pass filter row by row."""
    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))
    if U.ndim == 3:
        fft = np.fft.fft2(U, axes=(-2, -1))
        modes_x = np.fft.fftfreq(U.shape[-1])
        modes_y = np.fft.fftfreq(U.shape[-2])
        cutoff_x = keep_fraction * np.max(np.abs(modes_x))
        cutoff_y = keep_fraction * np.max(np.abs(modes_y))
        keep = (np.abs(modes_y)[:, None] <= cutoff_y) & (np.abs(modes_x)[None, :] <= cutoff_x)
        fft *= keep
        return np.real(np.fft.ifft2(fft, axes=(-2, -1)))
    fft = np.fft.fft(U, axis=1)
    modes = np.fft.fftfreq(U.shape[1])
    cutoff = keep_fraction * np.max(np.abs(modes))
    fft[:, np.abs(modes) > cutoff] = 0.0
    return np.real(np.fft.ifft(fft, axis=1))


def preprocess_for_method(U: np.ndarray, method_cfg: dict) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply method-specific preprocessing and return metadata."""
    name = method_cfg.get("name", "")
    preprocessing = method_cfg.get("preprocessing", {})
    meta: dict[str, Any] = {}

    if "savgol" in name or preprocessing.get("type") == "savgol":
        window = int(preprocessing.get("window_length", method_cfg.get("window_length", 9)))
        poly = int(preprocessing.get("polyorder", method_cfg.get("polyorder", 3)))
        meta.update({"preprocessing": "savgol", "window_length": window, "polyorder": poly})
        return smooth_temporal(U, window_length=window, polyorder=poly), meta

    if "spectral_filter" in name or preprocessing.get("type") == "spectral_lowpass":
        keep = float(preprocessing.get("keep_fraction", method_cfg.get("keep_fraction", 0.50)))
        meta.update({"preprocessing": "spectral_lowpass", "keep_fraction": keep})
        return spectral_lowpass(U, keep_fraction=keep), meta

    meta["preprocessing"] = "none"
    return U, meta


def assemble_strong_system(
    U: np.ndarray,
    grid: Grid1D | Grid2D,
    library,
    method_cfg: dict | None = None,
) -> LinearSystem:
    """Assemble A=Theta(U) and b=u_t for a strong-form method."""
    method_cfg = method_cfg or {}
    U_work, pre_meta = preprocess_for_method(U, method_cfg)
    diff = FourierSpectral2D(grid) if isinstance(grid, Grid2D) else FourierSpectral1D(grid)
    A = library.build_strong(U_work, diff)
    b = time_derivative_gradient(U_work, grid).ravel()
    return LinearSystem(
        A=A,
        b=b,
        feature_names=library.names,
        metadata={
            "formulation": "strong",
            "n_test_functions": 0,
            **pre_meta,
        },
    )
