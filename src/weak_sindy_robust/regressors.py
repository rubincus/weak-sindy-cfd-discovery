"""Sparse regressors used by the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from weak_sindy_robust.constraints import ConstraintReport, apply_positive_diffusion


@dataclass
class RegressorResult:
    """Result of a sparse regression fit."""

    xi: np.ndarray
    method: str
    support: set[int]
    residual_relative: float
    metadata: dict[str, Any] = field(default_factory=dict)
    constraints: ConstraintReport | None = None


def _column_norms(A: np.ndarray, normalize_columns: bool) -> np.ndarray:
    """Return safe column norms."""
    if not normalize_columns:
        return np.ones(A.shape[1], dtype=float)
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0.0] = 1.0
    return norms


def _solve_least_squares(A: np.ndarray, b: np.ndarray, ridge_alpha: float = 0.0) -> np.ndarray:
    """Solve least squares or ridge-stabilized least squares."""
    if ridge_alpha <= 0.0:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    lhs = A.T @ A + ridge_alpha * np.eye(A.shape[1])
    rhs = A.T @ b
    return np.linalg.solve(lhs, rhs)


def residual_relative(A: np.ndarray, b: np.ndarray, xi: np.ndarray) -> float:
    """Return ||A xi - b|| / ||b||."""
    denom = np.linalg.norm(b)
    if denom <= 1.0e-15:
        return float(np.linalg.norm(A @ xi - b))
    return float(np.linalg.norm(A @ xi - b) / denom)


def support_from_threshold(xi: np.ndarray, threshold: float = 1.0e-10) -> set[int]:
    """Return coefficient support using an absolute threshold."""
    return set(np.nonzero(np.abs(xi) > threshold)[0].tolist())


def stlsq(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    max_iter: int = 12,
    normalize_columns: bool = True,
    ridge_alpha: float = 0.0,
) -> RegressorResult:
    """Sequential thresholded least squares with optional ridge stabilization."""
    norms = _column_norms(A, normalize_columns)
    An = A / norms
    xi = _solve_least_squares(An, b, ridge_alpha=ridge_alpha)
    history: list[int] = []

    for _ in range(max_iter):
        small = np.abs(xi) < lam
        xi[small] = 0.0
        big = ~small
        history.append(int(np.sum(big)))
        if np.sum(big) == 0:
            break
        xi_big = _solve_least_squares(An[:, big], b, ridge_alpha=ridge_alpha)
        xi = np.zeros_like(xi)
        xi[big] = xi_big

    xi_unscaled = xi / norms
    return RegressorResult(
        xi=xi_unscaled,
        method="stlsq" if ridge_alpha <= 0.0 else "ridge_stlsq",
        support=support_from_threshold(xi_unscaled),
        residual_relative=residual_relative(A, b, xi_unscaled),
        metadata={
            "lambda": float(lam),
            "max_iter": int(max_iter),
            "normalize_columns": bool(normalize_columns),
            "ridge_alpha": float(ridge_alpha),
            "active_history": history,
        },
    )


def sr3(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    kappa: float = 1.0,
    max_iter: int = 100,
    normalize_columns: bool = True,
) -> RegressorResult:
    """Small deterministic SR3-style hard-thresholded relaxation."""
    norms = _column_norms(A, normalize_columns)
    An = A / norms
    p = An.shape[1]
    xi = _solve_least_squares(An, b)
    w = xi.copy()
    lhs = An.T @ An + kappa * np.eye(p)
    rhs_data = An.T @ b

    for _ in range(max_iter):
        xi = np.linalg.solve(lhs, rhs_data + kappa * w)
        w_new = xi.copy()
        w_new[np.abs(w_new) < lam] = 0.0
        if np.linalg.norm(w_new - w, ord=np.inf) < 1.0e-12:
            w = w_new
            break
        w = w_new

    xi_unscaled = w / norms
    return RegressorResult(
        xi=xi_unscaled,
        method="sr3",
        support=support_from_threshold(xi_unscaled),
        residual_relative=residual_relative(A, b, xi_unscaled),
        metadata={
            "lambda": float(lam),
            "kappa": float(kappa),
            "max_iter": int(max_iter),
            "normalize_columns": bool(normalize_columns),
        },
    )


def elasticnet(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    l1_ratio: float = 0.80,
    max_iter: int = 5000,
    normalize_columns: bool = True,
) -> RegressorResult:
    """Fit ElasticNet, using scikit-learn when available and a fallback otherwise."""
    norms = _column_norms(A, normalize_columns)
    An = A / norms
    try:
        from sklearn.linear_model import ElasticNet  # type: ignore

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=max_iter,
            tol=1.0e-10,
            selection="cyclic",
        )
        model.fit(An, b)
        coef = model.coef_
        backend = "sklearn"
    except Exception:
        coef = _elasticnet_coordinate_descent(An, b, alpha, l1_ratio, max_iter)
        backend = "numpy_coordinate_descent"

    xi = coef / norms
    return RegressorResult(
        xi=xi,
        method="elasticnet",
        support=support_from_threshold(xi, threshold=max(1.0e-10, alpha * 1.0e-3)),
        residual_relative=residual_relative(A, b, xi),
        metadata={
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
            "max_iter": int(max_iter),
            "normalize_columns": bool(normalize_columns),
            "backend": backend,
        },
    )


def _elasticnet_coordinate_descent(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> np.ndarray:
    """Coordinate descent fallback for a no-intercept ElasticNet objective."""
    n, p = A.shape
    coef = np.zeros(p, dtype=float)
    col_norm = np.sum(A * A, axis=0) / n
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    for _ in range(max_iter):
        old = coef.copy()
        residual = b - A @ coef
        for j in range(p):
            residual += A[:, j] * coef[j]
            rho = float(np.dot(A[:, j], residual) / n)
            if rho < -l1:
                coef[j] = (rho + l1) / (col_norm[j] + l2)
            elif rho > l1:
                coef[j] = (rho - l1) / (col_norm[j] + l2)
            else:
                coef[j] = 0.0
            residual -= A[:, j] * coef[j]
        if np.linalg.norm(coef - old, ord=np.inf) < 1.0e-10:
            break
    return coef


def stability_selection(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    n_repeats: int = 40,
    row_fraction: float = 0.70,
    probability_threshold: float = 0.60,
    seed: int = 0,
    normalize_columns: bool = True,
    max_iter: int = 12,
) -> RegressorResult:
    """Estimate support probabilities by row subsampling and refit selected terms."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    m = max(2, int(np.ceil(row_fraction * n)))
    counts = np.zeros(A.shape[1], dtype=float)

    for _ in range(n_repeats):
        rows = rng.choice(n, size=m, replace=False)
        fit = stlsq(
            A[rows],
            b[rows],
            lam=lam,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
        )
        for idx in fit.support:
            counts[idx] += 1.0

    probabilities = counts / max(1, n_repeats)
    selected = set(np.nonzero(probabilities >= probability_threshold)[0].tolist())
    xi = np.zeros(A.shape[1], dtype=float)
    if selected:
        cols = sorted(selected)
        xi_sub = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
        xi[cols] = xi_sub

    return RegressorResult(
        xi=xi,
        method="stability_selection",
        support=selected,
        residual_relative=residual_relative(A, b, xi),
        metadata={
            "lambda": float(lam),
            "n_repeats": int(n_repeats),
            "row_fraction": float(row_fraction),
            "probability_threshold": float(probability_threshold),
            "seed": int(seed),
            "support_probabilities": probabilities.tolist(),
        },
    )


def fit_regressor(A: np.ndarray, b: np.ndarray, feature_names: list[str], method_cfg: dict) -> RegressorResult:
    """Dispatch a method configuration to a sparse regressor."""
    name = method_cfg.get("name", "")
    regressor = method_cfg.get("regressor", "stlsq")
    lam = float(method_cfg.get("lambda", method_cfg.get("lam", 0.1)))
    max_iter = int(method_cfg.get("max_iter", 12))
    normalize_columns = bool(method_cfg.get("normalize_columns", True))

    if regressor == "sr3" or "sr3" in name:
        result = sr3(
            A,
            b,
            lam=lam,
            kappa=float(method_cfg.get("kappa", 1.0)),
            max_iter=int(method_cfg.get("max_iter", 100)),
            normalize_columns=normalize_columns,
        )
    elif regressor == "elasticnet" or "elasticnet" in name:
        result = elasticnet(
            A,
            b,
            alpha=float(method_cfg.get("alpha", lam)),
            l1_ratio=float(method_cfg.get("l1_ratio", 0.80)),
            max_iter=int(method_cfg.get("max_iter", 5000)),
            normalize_columns=normalize_columns,
        )
    elif regressor == "stability_selection" or "stability" in name:
        result = stability_selection(
            A,
            b,
            lam=lam,
            n_repeats=int(method_cfg.get("n_repeats", 40)),
            row_fraction=float(method_cfg.get("row_fraction", 0.70)),
            probability_threshold=float(method_cfg.get("probability_threshold", 0.60)),
            seed=int(method_cfg.get("seed", 0)),
            normalize_columns=normalize_columns,
            max_iter=max_iter,
        )
    else:
        result = stlsq(
            A,
            b,
            lam=lam,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            ridge_alpha=float(method_cfg.get("ridge_alpha", 0.0)) if "ridge" in name else 0.0,
        )

    result.metadata["configured_method"] = name
    result.metadata["configured_regressor"] = regressor

    constraints_cfg = method_cfg.get("constraints", {})
    if "constrained" in name or constraints_cfg.get("positive_diffusion", False):
        xi_new, report = apply_positive_diffusion(result.xi, feature_names)
        result.xi = xi_new
        result.support = support_from_threshold(xi_new)
        result.residual_relative = residual_relative(A, b, xi_new)
        result.constraints = report
    return result

