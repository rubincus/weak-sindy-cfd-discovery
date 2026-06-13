"""Tests for deterministic configured runs."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.experiments import ExperimentRunner


def test_reproducibility_for_same_seed(tmp_path) -> None:
    """Running the same small experiment twice gives identical coefficients."""
    base_cfg = {
        "experiment": {
            "name": "repro_test",
            "output_dir": str(tmp_path / "run1"),
            "save_arrays": False,
            "save_figures": False,
        },
        "problem": {"grid": {"Nx": 32, "Nt": 61, "T": 1.20}},
        "noise": {"cases": [{"type": "gaussian_iid", "sigma": 0.03, "seed": 7}]},
        "identification": {
            "methods": [
                {
                    "name": "weak_stlsq",
                    "formulation": "weak",
                    "regressor": "stlsq",
                    "lambda": 0.1,
                    "normalize_columns": True,
                    "max_iter": 12,
                    "test_functions": {
                        "family": "compact_cosine",
                        "ax": 0.90,
                        "at": 0.22,
                        "nxc": 4,
                        "ntc": 3,
                    },
                }
            ]
        },
    }
    runner1 = ExperimentRunner(base_cfg, base_dir=tmp_path)
    runner1.run()
    cfg2 = dict(base_cfg)
    cfg2["experiment"] = dict(base_cfg["experiment"])
    cfg2["experiment"]["output_dir"] = str(tmp_path / "run2")
    runner2 = ExperimentRunner(cfg2, base_dir=tmp_path)
    runner2.run()
    xi1 = np.array([runner1.coefficient_rows[0][f"coef_{name}"] for name in ["1", "u", "u2", "ux", "uux", "uxx"]])
    xi2 = np.array([runner2.coefficient_rows[0][f"coef_{name}"] for name in ["1", "u", "u2", "ux", "uux", "uxx"]])
    assert np.linalg.norm(xi1 - xi2, ord=np.inf) < 1.0e-12

