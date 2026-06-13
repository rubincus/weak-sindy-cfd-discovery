"""Run a configured weak-sindy-robust experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from weak_sindy_robust.config import load_config  # noqa: E402
from weak_sindy_robust.experiments import ExperimentRunner  # noqa: E402


def main() -> None:
    """Parse CLI arguments and run an experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config, base_dir=ROOT)
    runner.run()


if __name__ == "__main__":
    main()

