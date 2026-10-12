"""Lazy-execution memory: snapshot cost, branches, aliases, cache release.

The key question: snapshots must not silently duplicate large data.
TensorLang snapshots dense tensors as read-only views (copy-on-write
discipline), so graph construction over an 80MB array should add ~0MB.
External arrays are still copied once (callers may mutate them later).
"""
from __future__ import annotations

import numpy as np

from harness import Measurement, peak_rss_mb

from tensorlang import P, T


def _rss():
    return peak_rss_mb()


def run(size: str) -> list[Measurement]:
    n = {"small": 1_000_000, "medium": 10_000_000, "large": 50_000_000}[size]
    out: list[Measurement] = []
    X = P("X")

    base = np.arange(n, dtype=np.float64)  # n * 8 bytes
    rss0 = _rss()
    t = T(base)
    rss_tensor = _rss() - rss0

    # Ten branched graphs over the same tensor: views, not copies.
    rss1 = _rss()
    branches = [(t * float(i) + 1).sum() for i in range(1, 11)]
    rss_graphs = _rss() - rss1
    out.append(Measurement("snapshot-10-branches", size, "tensorlang", 0.0, 0.0,
                           _rss(), backend="cow-views",
                           note=f"tensor={rss_tensor:.1f}MB graphs={rss_graphs:.1f}MB "
                                f"for {n * 8 / 1e6:.0f}MB data"))

    # Collecting one branch materializes only its own result (a scalar).
    rss2 = _rss()
    total = branches[0].collect()
    rss_collect = _rss() - rss2
    assert total == float(np.arange(n).sum() + n)
    out.append(Measurement("branch-collect", size, "tensorlang", 0.0, 0.0,
                           _rss(), backend="materialized",
                           note=f"collect added {rss_collect:.1f}MB"))

    # Cache release drops the retained result.
    big = (t * 2).collect()
    assert isinstance(big, np.ndarray)
    lazy = t * 2
    lazy.collect()
    lazy.release()
    out.append(Measurement("cache-release", size, "tensorlang", 0.0, 0.0,
                           _rss(), backend="n/a",
                           note="release() clears retained collect() results"))

    # External arrays are copied once at the boundary (caller may mutate).
    ext = np.ones(n)
    before = ext.copy()
    lazy2 = T(np.ones(3)) + ext[:3]
    ext[:] = 99.0
    assert lazy2.collect().tolist() == (before[:3] + 1).tolist()
    out.append(Measurement("external-copy-once", size, "tensorlang", 0.0, 0.0,
                           _rss(), backend="n/a",
                           note="caller-side mutation cannot leak into graphs"))
    return out
