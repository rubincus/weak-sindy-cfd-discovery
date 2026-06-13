"""Regenerate figures from a results directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from weak_sindy_robust.plotting import make_standard_figures  # noqa: E402


def main() -> None:
    """Parse CLI arguments and create figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str)
    args = parser.parse_args()
    make_standard_figures(args.results_dir)


if __name__ == "__main__":
    main()

