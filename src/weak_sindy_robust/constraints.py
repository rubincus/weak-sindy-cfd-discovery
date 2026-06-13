"""Explicit physical constraints for identified coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConstraintReport:
    """Serializable report for coefficient constraints."""

    active: bool
    names: list[str]
    changed: bool
    details: dict


def apply_positive_diffusion(
    xi: np.ndarray,
    feature_names: list[str],
    min_value: float = 0.0,
) -> tuple[np.ndarray, ConstraintReport]:
    """Project the uxx coefficient onto xi_uxx >= min_value when present."""
    xi_new = np.array(xi, dtype=float, copy=True)
    changed = False
    details: dict = {"min_value": float(min_value)}
    if "uxx" in feature_names:
        idx = feature_names.index("uxx")
        details["uxx_index"] = idx
        details["before"] = float(xi_new[idx])
        if xi_new[idx] < min_value:
            xi_new[idx] = min_value
            changed = True
        details["after"] = float(xi_new[idx])
    return xi_new, ConstraintReport(
        active=True,
        names=["positive_diffusion"],
        changed=changed,
        details=details,
    )

