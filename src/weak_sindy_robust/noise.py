"""Noise models for synthetic observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NoiseModel:
    """Base class for observation noise models."""

    @property
    def name(self) -> str:
        """Return a serializable noise model name."""
        raise NotImplementedError

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply noise to a clean field."""
        raise NotImplementedError


@dataclass
class GaussianIIDNoise(NoiseModel):
    """Independent Gaussian noise scaled by a global field statistic."""

    sigma: float
    relative_to: str = "std"

    @property
    def name(self) -> str:
        """Return the model name."""
        return "gaussian_iid"

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return U plus iid Gaussian perturbations."""
        scale = np.std(U) if self.relative_to == "std" else np.max(np.abs(U))
        return U + rng.normal(loc=0.0, scale=self.sigma * scale, size=U.shape)


def _gaussian_kernel1d(sigma_cells: float) -> np.ndarray:
    """Build a normalized Gaussian kernel in grid-cell units."""
    if sigma_cells <= 1.0e-12:
        return np.array([1.0])
    radius = max(1, int(np.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    return kernel / np.sum(kernel)


def _wrap_convolve_axis(A: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve with periodic wrapping along one axis."""
    if kernel.size == 1:
        return A.copy()
    radius = kernel.size // 2
    padded = np.pad(A, [(radius, radius) if i == axis else (0, 0) for i in range(A.ndim)], mode="wrap")
    moved = np.moveaxis(padded, axis, 0)
    out = np.empty_like(np.moveaxis(A, axis, 0), dtype=float)
    for i in range(out.shape[0]):
        window = moved[i : i + kernel.size]
        out[i] = np.tensordot(kernel, window, axes=(0, 0))
    return np.moveaxis(out, 0, axis)


def _normalize_noise(noise: np.ndarray, target_std: float) -> np.ndarray:
    """Center and scale a noise field to a requested standard deviation."""
    centered = noise - np.mean(noise)
    scale = np.std(centered)
    if scale <= 1.0e-15:
        return np.zeros_like(noise)
    return centered * (target_std / scale)


@dataclass
class SpatialCorrelatedGaussianNoise(NoiseModel):
    """Gaussian noise correlated along space."""

    sigma: float
    ell_x: float

    @property
    def name(self) -> str:
        """Return the model name."""
        return "spatial_correlated_gaussian"

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return U plus spatially correlated Gaussian perturbations."""
        white = rng.normal(0.0, 1.0, U.shape)
        smooth = _wrap_convolve_axis(white, _gaussian_kernel1d(self.ell_x), axis=1)
        return U + _normalize_noise(smooth, self.sigma * np.std(U))


@dataclass
class TemporalCorrelatedGaussianNoise(NoiseModel):
    """Gaussian noise correlated along time."""

    sigma: float
    ell_t: float

    @property
    def name(self) -> str:
        """Return the model name."""
        return "temporal_correlated_gaussian"

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return U plus temporally correlated Gaussian perturbations."""
        white = rng.normal(0.0, 1.0, U.shape)
        smooth = _wrap_convolve_axis(white, _gaussian_kernel1d(self.ell_t), axis=0)
        return U + _normalize_noise(smooth, self.sigma * np.std(U))


@dataclass
class SpatioTemporalCorrelatedGaussianNoise(NoiseModel):
    """Gaussian noise correlated in space and time."""

    sigma: float
    ell_x: float
    ell_t: float

    @property
    def name(self) -> str:
        """Return the model name."""
        return "spatiotemporal_correlated_gaussian"

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return U plus spatiotemporally correlated Gaussian perturbations."""
        white = rng.normal(0.0, 1.0, U.shape)
        smooth = _wrap_convolve_axis(white, _gaussian_kernel1d(self.ell_t), axis=0)
        smooth = _wrap_convolve_axis(smooth, _gaussian_kernel1d(self.ell_x), axis=1)
        return U + _normalize_noise(smooth, self.sigma * np.std(U))


@dataclass
class HeteroscedasticGaussianNoise(NoiseModel):
    """Gaussian noise whose scale follows the local amplitude."""

    sigma: float
    epsilon: float = 1.0e-8

    @property
    def name(self) -> str:
        """Return the model name."""
        return "heteroscedastic_gaussian"

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return U plus heteroscedastic Gaussian perturbations."""
        local_scale = self.sigma * (np.abs(U) + self.epsilon)
        return U + rng.normal(0.0, local_scale, size=U.shape)


@dataclass
class ImpulsiveOutlierNoise(NoiseModel):
    """IID Gaussian noise with sparse large outliers."""

    sigma: float
    outlier_probability: float
    outlier_scale: float = 10.0

    @property
    def name(self) -> str:
        """Return the model name."""
        return "impulsive_outliers"

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return U plus Gaussian perturbations and sparse outliers."""
        base = GaussianIIDNoise(self.sigma).apply(U, rng)
        mask = rng.random(U.shape) < self.outlier_probability
        outliers = rng.normal(0.0, self.outlier_scale * self.sigma * np.std(U), U.shape)
        base[mask] += outliers[mask]
        return base


@dataclass
class MissingDataMask:
    """Random mask for missing observations."""

    missing_probability: float

    def apply(self, U: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return a NaN-masked copy and the boolean observation mask."""
        mask = rng.random(U.shape) >= self.missing_probability
        U_masked = U.copy()
        U_masked[~mask] = np.nan
        return U_masked, mask


def fill_missing_nearest_mean(U: np.ndarray) -> np.ndarray:
    """Fill NaNs with column means and then the global mean if needed."""
    if not np.isnan(U).any():
        return U
    filled = U.copy()
    col_means = np.nanmean(filled, axis=0)
    global_mean = float(np.nanmean(filled)) if np.isfinite(np.nanmean(filled)) else 0.0
    col_means = np.where(np.isfinite(col_means), col_means, global_mean)
    rows, cols = np.where(np.isnan(filled))
    filled[rows, cols] = col_means[cols]
    return filled


def noise_model_from_config(case: dict) -> NoiseModel | MissingDataMask:
    """Create a noise model from a config dictionary."""
    kind = case.get("type", "gaussian_iid")
    sigma = float(case.get("sigma", 0.0))
    if kind == "gaussian_iid":
        return GaussianIIDNoise(sigma=sigma, relative_to=case.get("relative_to", "std"))
    if kind == "spatial_correlated_gaussian":
        return SpatialCorrelatedGaussianNoise(sigma=sigma, ell_x=float(case.get("ell_x", 2.0)))
    if kind == "temporal_correlated_gaussian":
        return TemporalCorrelatedGaussianNoise(sigma=sigma, ell_t=float(case.get("ell_t", 2.0)))
    if kind == "spatiotemporal_correlated_gaussian":
        return SpatioTemporalCorrelatedGaussianNoise(
            sigma=sigma,
            ell_x=float(case.get("ell_x", 2.0)),
            ell_t=float(case.get("ell_t", 2.0)),
        )
    if kind == "heteroscedastic_gaussian":
        return HeteroscedasticGaussianNoise(sigma=sigma)
    if kind == "impulsive_outliers":
        return ImpulsiveOutlierNoise(
            sigma=sigma,
            outlier_probability=float(case.get("outlier_probability", 0.02)),
            outlier_scale=float(case.get("outlier_scale", 10.0)),
        )
    if kind == "missing_data":
        return MissingDataMask(missing_probability=float(case.get("missing_probability", 0.05)))
    raise ValueError(f"Unknown noise type: {kind}")

