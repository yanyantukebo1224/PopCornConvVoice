"""Small, dependency-light helpers for repeatable model latency measurements."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LatencyReport:
    iterations: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    audio_chunk_ms: float | None
    real_time_factor: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty list."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benchmark_callable(
    operation: Callable[[], Any],
    *,
    warmup: int = 20,
    iterations: int = 100,
    audio_chunk_ms: float | None = None,
    synchronize: Callable[[], None] | None = None,
) -> LatencyReport:
    """Measure a callable, optionally synchronizing an asynchronous accelerator.

    Pass a synchronization callback such as torch.cuda.synchronize when
    measuring CUDA work. The callback is called before and after each timed
    operation so asynchronous launch time is not mistaken for execution time.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if audio_chunk_ms is not None and audio_chunk_ms <= 0:
        raise ValueError("audio_chunk_ms must be > 0")

    sync = synchronize or (lambda: None)

    for _ in range(warmup):
        operation()
    sync()

    durations_ms: list[float] = []
    for _ in range(iterations):
        sync()
        started = time.perf_counter()
        operation()
        sync()
        durations_ms.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.fmean(durations_ms)
    rtf = mean_ms / audio_chunk_ms if audio_chunk_ms is not None else None

    return LatencyReport(
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean_ms,
        median_ms=statistics.median(durations_ms),
        p95_ms=_percentile(durations_ms, 95.0),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        audio_chunk_ms=audio_chunk_ms,
        real_time_factor=rtf,
    )
