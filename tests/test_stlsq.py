"""Tests for STLSQ."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.regressors import stlsq


def test_stlsq_recovers_sparse_support() -> None:
    """STLSQ recovers a simple sparse synthetic support."""
    rng = np.random.default_rng(123)
    A = rng.normal(size=(300, 5))
    xi_true = np.array([0.0, 1.5, 0.0, 0.0, -0.6])
    b = A @ xi_true + 1.0e-4 * rng.normal(size=A.shape[0])
    result = stlsq(A, b, lam=0.1, max_iter=12, normalize_columns=True)
    assert result.support == {1, 4}
    assert np.linalg.norm(result.xi - xi_true) < 1.0e-2

