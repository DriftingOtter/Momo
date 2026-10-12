"""Dense workloads: elementwise chains and matmul vs native NumPy."""
from __future__ import annotations

import numpy as np

from harness import Measurement, measure, peak_rss_mb

from tensorlang import P, T

SIZES = {"small": 100_000, "medium": 1_000_000, "large": 10_000_000}
MATMUL = {"small": 500, "medium": 1000, "large": 1500}


def run(size: str) -> list[Measurement]:
    from datasets import make_dense

    n = SIZES[size]
    a, b, c, d = make_dense(n)
    ta, tb, tc, td = T(a), T(b), T(c), T(d)
    out: list[Measurement] = []

    def tl_chain():
        return ((ta * tb + tc) / td).collect()

    def np_chain():
        return (a * b + c) / d

    secs, alloc, got = measure(tl_chain)
    _, _, ref = measure(np_chain)
    assert np.allclose(got, ref)
    out.append(Measurement("dense-chain", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="materialized",
                           materializations=1))
    secs, alloc, _ = measure(np_chain)
    out.append(Measurement("dense-chain", size, "native", secs, alloc,
                           peak_rss_mb(), backend="numpy"))

    X = P("X")
    f = X**2 + 1

    def tl_op():
        return f(ta).collect()

    def np_op():
        return a**2 + 1

    secs, alloc, got = measure(tl_op)
    _, _, ref = measure(np_op)
    assert np.allclose(got, ref)
    out.append(Measurement("dense-operator", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="materialized",
                           materializations=1))
    secs, alloc, _ = measure(np_op)
    out.append(Measurement("dense-operator", size, "native", secs, alloc,
                           peak_rss_mb(), backend="numpy"))

    m = MATMUL[size]
    rng = np.random.default_rng(3)
    A, B = rng.random((m, m)), rng.random((m, m))
    tA, tB = T(A), T(B)

    def tl_mm():
        return (tA @ tB).sum().collect()

    def np_mm():
        return (A @ B).sum()

    secs, alloc, got = measure(tl_mm)
    _, _, ref = measure(np_mm)
    assert np.allclose(got, ref)
    out.append(Measurement(f"matmul-{m}x{m}", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="materialized",
                           materializations=1))
    secs, alloc, _ = measure(np_mm)
    out.append(Measurement(f"matmul-{m}x{m}", size, "native", secs, alloc,
                           peak_rss_mb(), backend="numpy"))
    return out
