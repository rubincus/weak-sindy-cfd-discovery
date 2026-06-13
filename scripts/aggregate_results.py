"""Aggregate metrics in a results directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from weak_sindy_robust.aggregation import aggregate_results  # noqa: E402


def main() -> None:
    """Parse CLI arguments and aggregate results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str)
    args = parser.parse_args()
    aggregate_results(args.results_dir)


if __name__ == "__main__":
    main()

