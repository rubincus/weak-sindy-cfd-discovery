"""Input/output helpers for experiment artifacts."""

from __future__ import annotations

import csv
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def safe_slug(value: Any) -> str:
    """Return a filesystem-safe slug."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def jsonable(value: Any) -> Any:
    """Convert common numpy values to JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: str | Path, data: dict) -> None:
    """Write JSON with stable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a list of dictionaries to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    """Convert nested values for CSV cells."""
    value = jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def save_npz(path: str | Path, **arrays: Any) -> None:
    """Save arrays and JSON-compatible metadata to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = {}
    for key, value in arrays.items():
        if isinstance(value, (dict, list, tuple, set)):
            converted[key] = np.array(json.dumps(jsonable(value)), dtype=object)
        else:
            converted[key] = value
    np.savez_compressed(path, **converted)


def environment_report() -> str:
    """Return a plain-text report for the runtime environment."""
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"executable: {sys.executable}",
    ]
    for name in ["numpy", "scipy", "pandas", "yaml", "matplotlib", "sklearn", "tqdm"]:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "unknown")
            lines.append(f"{name}: {version}")
        except Exception:
            lines.append(f"{name}: not installed")
    return "\n".join(lines) + "\n"

