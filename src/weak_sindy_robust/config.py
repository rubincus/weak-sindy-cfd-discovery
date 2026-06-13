"""Configuration loading with a PyYAML fallback for the bundled runtime."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "burgers_reproduce",
        "output_dir": "results/burgers_reproduce",
        "seed_master": 20260612,
        "save_arrays": True,
        "save_figures": True,
    },
    "problem": {
        "name": "burgers_1d",
        "equation": "u_t = -u*u_x + nu*u_xx",
        "nu": 0.075,
        "domain": {"L": 6.283185307179586, "periodic": True},
        "grid": {"Nx": 128, "Nt": 241, "T": 1.20},
    },
    "solver": {
        "time_integrator": "rk4",
        "derivative_operator": "fourier_spectral",
        "dealiasing": False,
    },
    "library": {
        "name": "burgers_base",
        "terms": ["1", "u", "u2", "ux", "uux", "uxx"],
        "true_coefficients": [0.0, 0.0, 0.0, 0.0, -1.0, 0.075],
        "true_support": [4, 5],
    },
    "identification": {
        "methods": [
            {
                "name": "strong_stlsq",
                "formulation": "strong",
                "regressor": "stlsq",
                "lambda": 0.1,
                "normalize_columns": True,
                "max_iter": 12,
            },
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
                    "nxc": 14,
                    "ntc": 10,
                    "center_policy": "interior_uniform",
                },
            },
        ]
    },
    "noise": {
        "cases": [
            {"type": "gaussian_iid", "sigma": 0.0, "seed": 0},
            {"type": "gaussian_iid", "sigma": 0.05, "seed": 0},
        ]
    },
    "rollout": {
        "enabled": True,
        "T_rollout": 1.20,
        "dt": None,
        "diffusion_policy": "raw",
        "blowup_threshold": 50.0,
    },
    "metrics": {
        "support_threshold_mode": "relative",
        "support_threshold_relative": 1.0e-4,
        "compute_energy": True,
        "compute_condition_number": True,
        "compute_weak_residual": True,
    },
}


def deep_update(base: dict, override: dict) -> dict:
    """Return a deep merge of two dictionaries."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    """Load a YAML/JSON config and merge it with defaults."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = _loads_yaml_subset(text)
    return deep_update(DEFAULT_CONFIG, loaded)


def save_config(config: dict, path: str | Path) -> None:
    """Write a resolved config as YAML when possible, otherwise JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _strip_comment(line: str) -> str:
    """Strip comments from simple YAML lines."""
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _prepare_yaml_lines(text: str) -> list[tuple[int, str]]:
    """Return non-empty YAML lines with indentation."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        raw = _strip_comment(raw).rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    return lines


def _parse_scalar(value: str) -> Any:
    """Parse a scalar from a small YAML subset."""
    value = value.strip()
    if value in {"null", "Null", "NULL", "~", "None"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") or value.startswith("{"):
        normalized = (
            value.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        return ast.literal_eval(normalized)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_key_value(text: str) -> tuple[str, str]:
    """Split a YAML key-value line."""
    if ":" not in text:
        raise ValueError(f"Expected key: value line, got {text!r}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _loads_yaml_subset(text: str) -> dict:
    """Parse the simple YAML subset used by this project configs."""
    lines = _prepare_yaml_lines(text)
    if not lines:
        return {}

    def parse_block(i: int, indent: int) -> tuple[Any, int]:
        if i >= len(lines):
            return {}, i
        is_list = lines[i][0] == indent and lines[i][1].startswith("- ")
        if is_list:
            values = []
            while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                item = lines[i][1][2:].strip()
                i += 1
                if item == "":
                    if i < len(lines) and lines[i][0] > indent:
                        value, i = parse_block(i, lines[i][0])
                    else:
                        value = None
                elif ":" in item and not item.startswith(("'", '"')):
                    key, raw_value = _split_key_value(item)
                    value = {}
                    if raw_value:
                        value[key] = _parse_scalar(raw_value)
                    elif i < len(lines) and lines[i][0] > indent:
                        nested, i = parse_block(i, lines[i][0])
                        value[key] = nested
                    else:
                        value[key] = {}
                    if i < len(lines) and lines[i][0] > indent:
                        nested, i = parse_block(i, lines[i][0])
                        if isinstance(nested, dict):
                            value.update(nested)
                    values.append(value)
                else:
                    values.append(_parse_scalar(item))
            return values, i

        mapping = {}
        while i < len(lines) and lines[i][0] == indent and not lines[i][1].startswith("- "):
            key, raw_value = _split_key_value(lines[i][1])
            i += 1
            if raw_value:
                mapping[key] = _parse_scalar(raw_value)
            elif i < len(lines) and lines[i][0] > indent:
                mapping[key], i = parse_block(i, lines[i][0])
            else:
                mapping[key] = {}
        return mapping, i

    parsed, final_i = parse_block(0, lines[0][0])
    if final_i != len(lines):
        raise ValueError("Could not parse complete YAML file with fallback parser.")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML value must be a mapping.")
    return parsed

