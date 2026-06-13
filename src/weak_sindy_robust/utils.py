"""Small utilities used across the package."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TimerResult:
    """Elapsed time container."""

    elapsed_sec: float = 0.0


@contextmanager
def timed() -> Iterator[TimerResult]:
    """Measure elapsed wall-clock time for a code block."""
    result = TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_sec = time.perf_counter() - start

