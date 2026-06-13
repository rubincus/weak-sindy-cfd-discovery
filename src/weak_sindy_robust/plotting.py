"""Figure generation from CSV/NPZ artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_standard_figures(results_dir: str | Path) -> None:
    """Generate standard PNG/PDF figures when matplotlib is installed."""
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        (fig_dir / "README.txt").write_text(
            "matplotlib is not installed; figures can be regenerated after installing project dependencies.\n",
            encoding="utf-8",
        )
        return

    metrics = _read_csv(results_dir / "metrics.csv")
    coefficients = _read_csv(results_dir / "coefficients.csv")
    if metrics:
        _plot_metric_vs_sigma(metrics, fig_dir, "E_xi", "coefficient_error")
        _plot_metric_vs_sigma(metrics, fig_dir, "rollout_rmse_train", "rollout_rmse")
        _plot_metric_vs_sigma(metrics, fig_dir, "support_f1", "support_f1")
    if coefficients:
        _plot_coefficients(coefficients, fig_dir)
    plt.close("all")


def _plot_metric_vs_sigma(rows: list[dict[str, str]], fig_dir: Path, metric: str, name: str) -> None:
    """Plot one metric versus sigma."""
    import matplotlib.pyplot as plt  # type: ignore

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            sigma = float(row.get("sigma", "nan"))
            value = float(row.get(metric, "nan"))
        except ValueError:
            continue
        if not np.isfinite(sigma) or not np.isfinite(value):
            continue
        by_method.setdefault(row.get("method", ""), []).append((sigma, value))

    if not by_method:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for method, pairs in sorted(by_method.items()):
        sigmas = sorted(set(s for s, _ in pairs))
        medians = [np.median([v for s, v in pairs if s == sigma]) for sigma in sigmas]
        ax.plot(sigmas, medians, marker="o", label=method)
    ax.set_xlabel("sigma")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"{name}.{ext}")
    plt.close(fig)


def _plot_coefficients(rows: list[dict[str, str]], fig_dir: Path) -> None:
    """Plot coefficient bars for the first few result rows."""
    import matplotlib.pyplot as plt  # type: ignore

    coef_keys = [key for key in rows[0] if key.startswith("coef_")]
    if not coef_keys:
        return
    selected_rows = rows[: min(6, len(rows))]
    fig, axes = plt.subplots(len(selected_rows), 1, figsize=(7.0, 2.0 * len(selected_rows)), squeeze=False)
    for ax, row in zip(axes[:, 0], selected_rows):
        values = [float(row.get(key, "0") or 0.0) for key in coef_keys]
        labels = [key.replace("coef_", "") for key in coef_keys]
        ax.bar(labels, values)
        ax.set_title(f"{row.get('method')} sigma={row.get('sigma')} seed={row.get('seed')}", fontsize=9)
        ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"coefficient_bars.{ext}")
    plt.close(fig)

