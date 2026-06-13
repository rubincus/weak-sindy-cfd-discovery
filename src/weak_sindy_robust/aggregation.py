"""Aggregation utilities for experiment result folders."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from weak_sindy_robust.io import write_csv
from weak_sindy_robust.metrics import bootstrap_median_ci


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str) -> float:
    """Read a float cell with NaN fallback."""
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def aggregate_results(results_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Create standard summaries from metrics.csv."""
    results_dir = Path(results_dir)
    metrics = _read_csv(results_dir / "metrics.csv")
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        grouped[(row.get("method", ""), row.get("noise_type", ""), row.get("sigma", ""))].append(row)

    mc_rows: list[dict[str, Any]] = []
    for (method, noise_type, sigma), rows in sorted(grouped.items()):
        if not rows:
            continue
        for metric in ["E_xi", "rollout_rmse_train", "support_f1"]:
            values = np.array([_float(row, metric) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                med = iqr = ci_lo = ci_hi = float("nan")
            else:
                med = float(np.median(finite))
                q75, q25 = np.percentile(finite, [75, 25])
                iqr = float(q75 - q25)
                ci_lo, ci_hi = bootstrap_median_ci(finite, n_boot=300)
            mc_rows.append(
                {
                    "method": method,
                    "noise_type": noise_type,
                    "sigma": sigma,
                    "metric": metric,
                    "median": med,
                    "iqr": iqr,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "n": int(len(rows)),
                }
            )
    write_csv(summaries_dir / "mc_summary.csv", mc_rows)
    return {"mc_summary": mc_rows}

