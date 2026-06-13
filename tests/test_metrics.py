"""Tests for metrics."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.metrics import coefficient_error, support_scores


def test_support_metrics_and_coefficient_error() -> None:
    """Support metrics report exact recovery and small coefficient error."""
    scores = support_scores({4, 5}, {4, 5})
    assert scores["support_exact"] is True
    assert scores["support_f1"] == 1.0
    err = coefficient_error(np.array([0.0, -1.0, 0.075]), np.array([0.0, -1.0, 0.075]))
    assert err == 0.0

