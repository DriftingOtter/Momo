"""Shared benchmark harness: timing, memory, machine specs.

Methodology (see benchmarks/README.md):
- wall-clock time: best of N repeats after 1 warmup (time.perf_counter),
- peak Python allocation: tracemalloc around the measured region,
- peak process RSS: resource.getrusage high-water mark (absolute; deltas
  are reported against suite start since maxrss never decreases),
- correctness: every TensorLang result is asserted against native output,
- backend/materializations: taken from profile()/explain(), never guessed.
"""
from __future__ import annotations

import os
import platform
import time
import tracemalloc
from dataclasses import dataclass


@dataclass
class Measurement:
    workload: str
    size: str
    impl: str  # 'tensorlang' | 'native'
    seconds: float
    peak_python_alloc_mb: float
    peak_rss_mb: float
    backend: str = ""
    materializations: int = 0
    correct: bool = True
    note: str = ""


def measure(fn, repeats: int = 3):
    """Run ``fn`` (warmup + repeats); return (best_seconds, peak_py_mb, result)."""
    fn()
    best, peak, out = None, 0.0, None
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - start
        _, peak_now = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, peak_now / 1e6)
        best = elapsed if best is None else min(best, elapsed)
    return best, peak, out


def peak_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / 1e6
        return rss / 1e3
    except Exception:
        return float("nan")


def machine_specs() -> dict:
    import numpy
    import polars
    import sympy

    import tensorlang

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "tensorlang": tensorlang.__version__,
        "numpy": numpy.__version__,
        "polars": polars.__version__,
        "sympy": sympy.__version__,
    }
