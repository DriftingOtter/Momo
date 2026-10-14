"""v0.6 M1 tests: executor memoization, structural dedupe, AD liveness.

Liveness bugs can only cost recomputation EXCEPT structural-key
over-unification (wrong values) and taint failures (stale values);
these tests pin the safety properties directly.
"""
import numpy as np
import pytest

from tensorlang import P, T, grad
from tensorlang.runtime import executor as _ex


def _count_evals(fn):
    counts = {}

    def hook(node):
        counts[type(node).__name__] = counts.get(type(node).__name__, 0) + 1
        return hook.orig(node)

    hook.orig = _ex._evaluate
    _ex._evaluate = hook
    try:
        out = fn()
    finally:
        _ex._evaluate = hook.orig
    return out, counts


def test_shared_subgraph_evaluates_once():
    X = P("X")
    y = X * 2  # one object, two consumers
    loss = (y + y).sum()
    _, counts = _count_evals(lambda: loss(T([1.0, 2.0])).collect())
    assert counts.get("BinOp", 0) == 2  # mul once, add once (not 3)


def test_structural_duplicates_unify():
    # Distinct objects, identical structure post-substitution: the
    # squared term is built twice on purpose.
    X = P("X")
    sq = X * X
    loss = (sq + (X * X)).sum()
    (value, counts) = _count_evals(lambda: loss(T([2.0, 3.0])).collect())
    assert value == pytest.approx(2 * (4.0 + 9.0))
    assert counts.get("BinOp", 0) == 2  # two muls unify to one, plus add


def test_taint_isolation_across_adjoint_applications():
    # Miniature of the gradient-descent divergence: two structurally
    # identical updates bound to different gradient operators must not
    # share values.
    X = P("X")
    g1 = grad((X ** 2).mean())(T([4.0]))
    g2 = grad((X ** 2).mean())(T([10.0]))
    step = (g1 * 0.5) + (g2 * 0.5)
    assert step.collect().tolist() == pytest.approx([4.0 + 10.0])


def test_cotangents_released_during_reverse():
    X = P("X")
    op = grad((X * X * X).sum())
    lazy = op(T([1.0, 2.0]))
    assert lazy.collect().tolist() == pytest.approx([3.0, 12.0])
    node = lazy._root
    while not hasattr(node, "shared"):
        node = node.args[0]
    assert node.shared.last_released_cotangents > 0
    # DiffVar cotangents survive for readout; values tape stays pinned
    # for repeated application (forward_runs == 1).
    assert lazy.collect().tolist() == pytest.approx([3.0, 12.0])
    assert node.shared.forward_runs == 1


def test_scan_memoization_bypass_preserves_counts(tmp_path):
    import polars as pl

    from tensorlang.runtime import io as _io

    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    path = str(tmp_path / "s.parquet")
    frame.write_parquet(path)
    before = _io.SCAN_EXECUTIONS
    scan = T.scan_parquet(path)
    total = (scan.a.sum() + scan.a.sum()).collect()
    assert total == pytest.approx(12.0)
    assert _io.SCAN_EXECUTIONS - before == 2


def test_eviction_is_correctness_neutral():
    # Deep chain with heavy sharing: memo + eviction must match plain
    # evaluation exactly (differential check against unoptimized path).
    X = P("X")
    y = X + 1.0
    z = y * y
    loss = (z + y).sum() / 3.0
    got = loss(T([1.0, 2.0, 3.0])).collect()
    assert got == pytest.approx(((np.array([1.0, 2.0, 3.0]) + 1.0) ** 2
                                 + np.array([1.0, 2.0, 3.0]) + 1.0).sum() / 3.0)
