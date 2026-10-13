"""v0.5 M3 tests: memory budgets, resource admission, concurrency control,
runtime observation, and structured diagnostics.

Model-based decisions are deterministic (no RSS dependence); only profile
field presence — never absolute RSS values — is asserted for observations.
"""
import subprocess
import sys

import numpy as np
import pytest
import zarr

from tensorlang import P, T, TLUnsupportedError
from tensorlang.runtime import chunk_plan as _cp


def mstore(path, data, chunks):
    arr = zarr.open_array(path, mode="w", shape=data.shape,
                          chunks=chunks, dtype="float64")
    arr[:] = data
    return path


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(31)
    data = rng.random((200, 8)) + 0.5
    path = mstore(str(tmp_path / "a.zarr"), data, (50, 8))
    return path, data


@pytest.fixture
def budgeted():
    old = _cp.set_chunk_policy(memory_budget_bytes=64 * 1024**2,
                               max_workers=2)
    try:
        yield {"memory_budget_bytes": 64 * 1024**2, "max_workers": 2}
    finally:
        _cp.clear_chunk_policy()


def test_policy_configure_reset_and_validation():
    old = _cp.set_chunk_policy(memory_budget_bytes=1024, max_workers=3)
    assert old["memory_budget_bytes"] is None
    assert _cp.current_policy()["memory_budget_bytes"] == 1024
    assert _cp.current_policy()["max_workers"] == 3
    assert _cp.current_policy()["cpu_count"] >= 1
    _cp.clear_chunk_policy()
    assert _cp.current_policy()["memory_budget_bytes"] is None
    assert _cp.current_policy()["max_workers"] is None
    with pytest.raises(TLUnsupportedError) as exc:
        _cp.set_chunk_policy(max_workers=0)
    assert exc.value.code == "TL1023"
    _cp.clear_chunk_policy()


def test_legacy_policy_keys_restore(tmp_path):
    rng = np.random.default_rng(4)
    data = rng.random((20, 4)) + 0.5
    path = mstore(str(tmp_path / "a.zarr"), data, (10, 4))
    old = _cp.set_chunk_policy(max_tasks=5)
    try:
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(path)).collect()
        assert exc.value.code == "TL1012"
    finally:
        _cp.set_chunk_policy(**old)
    assert _cp.current_policy()["max_tasks"] == 200_000


def test_elementwise_estimate_and_admit(store, budgeted):
    path, data = store
    X = P("X")
    got = ((X + 1.0) * 2)(T.scan_zarr(path)).collect()
    assert np.allclose(got, (data + 1.0) * 2)
    decision = dict(_cp.LAST_DECISION)
    assert decision["outcome"] == "admitted"
    assert decision["workers_used"] == 2
    # binary working set: left block + scalar + output block.
    assert decision["peak_task_bytes"] == 2 * 50 * 8 * 8 + 8
    assert "binary" in decision["peak_task_detail"]
    assert decision["estimated_peak_bytes"] <= 64 * 1024**2
    text = ((X + 1.0) * 2)(T.scan_zarr(path)).explain()
    assert "admitted" in text and "workers 2" in text


def test_broadcast_and_alignment_estimates(store, budgeted):
    path, data = store
    rng = np.random.default_rng(32)
    other = rng.random((200, 8)) + 0.5
    pb = mstore(str(path).replace("a.zarr", "b.zarr"), other, (25, 8))
    X, Y = P("X"), P("Y")
    got = (X + Y)(T.scan_zarr(path), T.scan_zarr(pb)).collect()
    assert np.allclose(got, data + other)
    assert _cp.LAST_DECISION["outcome"] == "admitted"
    assert "[rechunk]" in (X + Y)(T.scan_zarr(path), T.scan_zarr(pb)).explain()


def test_reduction_and_matmul_estimates(store, budgeted):
    path, data = store
    assert T.scan_zarr(path).sum().collect() == pytest.approx(data.sum())
    assert _cp.LAST_DECISION["peak_task_bytes"] >= 50 * 8 * 8
    rng = np.random.default_rng(33)
    A = rng.random((12, 8)) + 0.5
    B = rng.random((8, 4)) + 0.5
    pa = mstore(str(path).replace("a.zarr", "A.zarr"), A, (6, 8))
    pb = mstore(str(path).replace("a.zarr", "B.zarr"), B, (8, 4))
    X, Y = P("X"), P("Y")
    assert np.allclose((X @ Y)(T.scan_zarr(pa), T.scan_zarr(pb)).collect(),
                       A @ B)
    decision = dict(_cp.LAST_DECISION)
    assert decision["peak_task_bytes"] == (6 * 8 + 8 * 4 + 6 * 4) * 8
    assert "matmul" in decision["peak_task_detail"]


def test_reduced_concurrency_admission(store):
    path, data = store
    # collect needs peak*workers + output + overhead. With workers=8 the
    # model exceeds 40KB; with workers=1 it fits: admit reduced.
    _cp.set_chunk_policy(memory_budget_bytes=40 * 1024, max_workers=8)
    try:
        X = P("X")
        got = (X + 1)(T.scan_zarr(path)).collect()
        assert np.allclose(got, data + 1)
        decision = dict(_cp.LAST_DECISION)
        assert decision["outcome"] == "admitted-reduced-concurrency"
        assert decision["workers_used"] == 1
        assert decision["workers_requested"] == 8
    finally:
        _cp.clear_chunk_policy()


def test_impossible_budget_rejected_with_suggestion(store):
    path, _ = store
    _cp.set_chunk_policy(memory_budget_bytes=2048, max_workers=1)
    try:
        # output alone (12.8KB) exceeds 2KB: TL1020/TL1024 family with a
        # feasible suggestion, never silent materialization.
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(path)).collect()
        assert exc.value.code in ("TL1020", "TL1024")
        assert "budget_bytes" in exc.value.details
        assert "suggested_budget_bytes" in exc.value.details or \
            "suggestion" in exc.value.details
    finally:
        _cp.clear_chunk_policy()


def test_single_block_exceeds_budget_needs_rechunk(store):
    path, _ = store
    _cp.set_chunk_policy(memory_budget_bytes=1024, max_workers=1)
    try:
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(path)).collect()
        # output (12.8KB) > 1KB budget: materialization gate fires first.
        assert exc.value.code == "TL1024"
    finally:
        _cp.clear_chunk_policy()
    # Streaming under a budget below the output size but above the
    # working set succeeds: outputs may exceed the working budget.
    _cp.set_chunk_policy(memory_budget_bytes=12500, max_workers=1)
    try:
        import shutil

        out = str(path).replace("a.zarr", "s.zarr")
        shutil.rmtree(out, ignore_errors=True)
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(out)
        assert np.allclose(zarr.open_array(out, mode="r")[:],
                           zarr.open_array(path, mode="r")[:] + 1)
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(path)).collect()
        assert exc.value.code == "TL1024"
        assert "write_zarr" in str(exc.value)
    finally:
        _cp.clear_chunk_policy()


def test_unknown_requirements_never_zero(store):
    path, _ = store
    _cp.set_chunk_policy(memory_budget_bytes=1024**2)
    try:
        ctx = _cp._ChunkContext()
        graph = _cp.lower((P("X") + 1)(T.scan_zarr(path))._root, ctx, {})
        ctx.peak_task_bytes = -1  # simulate an unmeasurable working set
        with pytest.raises(_cp._NotChunkable) as exc:
            _cp.decide(graph, ctx, purpose="collect")
        assert exc.value.code == "TL1021"
    finally:
        _cp.clear_chunk_policy()


def test_pressure_classifier():
    # Within budget / below noise floor / near model: silent no-ops.
    _cp.assess_pressure(1024, 1024**2, 512.0)
    _cp.assess_pressure(10 * 1024**2, None, 512.0)
    _cp.assess_pressure(70 * 1024**2, 1024**2, 60 * 1024**2)
    # Far above both budget and model: TL1022 with structured details.
    with pytest.raises(TLUnsupportedError) as exc:
        _cp.assess_pressure(512 * 1024**2, 1024**2, 1024.0)
    assert exc.value.code == "TL1022"
    assert exc.value.details["budget_bytes"] == 1024**2


def test_failed_plan_leaves_no_cache_and_recovers(store):
    path, data = store
    _cp.set_chunk_policy(memory_budget_bytes=1024)
    lazy = (P("X") + 1)(T.scan_zarr(path))
    try:
        with pytest.raises(TLUnsupportedError):
            lazy.collect()
    finally:
        _cp.clear_chunk_policy()
    assert np.allclose(lazy.collect(), data + 1)


def test_policy_does_not_leak_into_dask_defaults(store, budgeted):
    import dask.config

    path, _ = store
    before = dict(dask.config.config)
    (P("X") + 1)(T.scan_zarr(path)).collect()
    assert dict(dask.config.config) == before
    assert _cp._effective_workers() == 2
    _cp.clear_chunk_policy()
    import os as _os

    assert _cp._effective_workers() == (_os.cpu_count() or 4)


def test_profile_reports_estimates_and_measurements(store, budgeted):
    path, _ = store
    prof = ((P("X") + 1)(T.scan_zarr(path))).profile()
    text = str(prof)
    assert prof.backend == "dask-chunked"
    assert prof.baseline_rss_mb is not None and prof.peak_rss_mb is not None
    assert prof.estimated_peak_mb is not None and prof.budget_mb is not None
    assert prof.workers_used == 2 and prof.task_count and prof.task_count > 0
    assert "estimated_peak" in text and "budget" in text
    assert "baseline_rss" in text and "peak_rss" in text
    assert prof.pressure_observed is False and prof.pressure_code is None


def test_abstract_carries_policy_without_reads(store, budgeted):
    path, _ = store
    summary = ((P("X") + 1)(T.scan_zarr(path))).abstract()
    assert summary["memory_policy"]["memory_budget_bytes"] == 64 * 1024**2
    assert summary["memory_policy"]["max_workers"] == 2


def test_budget_failure_in_subprocess(tmp_path):
    data_path = str(tmp_path / "s.zarr")
    rng = np.random.default_rng(3)
    arr = zarr.open_array(data_path, mode="w", shape=(200, 8),
                          chunks=(50, 8), dtype="float64")
    arr[:] = rng.random((200, 8))
    code = (
        "from tensorlang import P, T, set_chunk_policy, TLUnsupportedError\n"
        "set_chunk_policy(memory_budget_bytes=1024)\n"
        "try:\n"
        f"    (P('X') + 1)(T.scan_zarr({data_path!r})).collect()\n"
        "except TLUnsupportedError as e:\n"
        "    print('REJECTED', e.code)\n"
        "    raise SystemExit(7)\n"
        "print('ADMITTED')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 7
    assert "REJECTED TL1024" in proc.stdout


def test_scalar_reduction_profile(store, budgeted):
    # Regression: profiling a scalar chunked reduction must not crash on
    # the unwrapped Python float (0-d dense has no .shape).
    path, data = store
    prof = T.scan_zarr(path).mean().profile()
    assert prof.backend == "dask-chunked"
    assert prof.shape == ()
    assert "dense()" in str(prof)
