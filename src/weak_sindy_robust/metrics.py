"""Metrics for coefficient recovery, support and rollout diagnostics."""

from __future__ import annotations

import numpy as np


def coefficient_error(xi: np.ndarray, xi_true: np.ndarray) -> float:
    """Return relative coefficient error ||xi-xi*||/||xi*||."""
    denom = np.linalg.norm(xi_true)
    if denom <= 1.0e-15:
        return float(np.linalg.norm(xi - xi_true))
    return float(np.linalg.norm(xi - xi_true) / denom)


def support_from_coefficients(
    xi: np.ndarray,
    threshold_mode: str = "relative",
    threshold_relative: float = 1.0e-4,
    threshold_absolute: float = 1.0e-10,
) -> set[int]:
    """Return support using relative or absolute thresholding."""
    if threshold_mode == "relative":
        scale = max(float(np.max(np.abs(xi))), 1.0)
        threshold = threshold_relative * scale
    else:
        threshold = threshold_absolute
    return set(np.nonzero(np.abs(xi) > threshold)[0].tolist())


def support_scores(selected: set[int], true_support: set[int]) -> dict[str, float | int | bool]:
    """Compute support precision, recall, F1 and symmetric difference."""
    tp = len(selected & true_support)
    fp = len(selected - true_support)
    fn = len(true_support - selected)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not true_support else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    symmetric_difference = len(selected ^ true_support)
    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "support_exact": bool(selected == true_support),
        "support_symmetric_difference": int(symmetric_difference),
    }


def condition_number_subset(A: np.ndarray, support: set[int]) -> float:
    """Return condition number of the active columns."""
    if not support:
        return float("nan")
    cols = sorted(support)
    try:
        return float(np.linalg.cond(A[:, cols]))
    except np.linalg.LinAlgError:
        return float("inf")


def kinetic_energy(U: np.ndarray, dx: float) -> np.ndarray:
    """Return kinetic energy 0.5 integral u^2 dx for each time."""
    return 0.5 * np.sum(U**2, axis=1) * dx


def energy_violation_rate(U: np.ndarray, dx: float, eps: float = 1.0e-10) -> float:
    """Return fraction of time increments where energy increases."""
    E = kinetic_energy(U, dx)
    violations = np.diff(E) > eps
    return float(np.mean(violations)) if violations.size else 0.0


def rmse(A: np.ndarray, B: np.ndarray) -> float:
    """Return root mean square error for equally shaped arrays."""
    return float(np.sqrt(np.mean((A - B) ** 2)))


def bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20260612,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return percentile bootstrap confidence interval for the median."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(finite, size=finite.size, replace=True)
        meds[i] = np.median(sample)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

