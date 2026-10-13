"""Chunked workloads: correctness plus bounded-memory evidence.

Correctness is asserted against NumPy on every run. Memory boundedness is
reported (peak RSS and Python allocation vs input bytes): chunk-local
execution must stay far below full materialization.
"""
from __future__ import annotations

import numpy as np
import zarr

from harness import Measurement, measure, peak_rss_mb

from tensorlang import P, T
from tensorlang import sin as _sin

SIZES = {"small": (200_000, 8), "medium": (1_000_000, 16), "large": (5_000_000, 32)}


def _make(path: str, rows: int, cols: int, chunk_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.random((rows, cols)) + 0.5
    arr = zarr.open_array(path, mode="w", shape=data.shape,
                          chunks=(chunk_rows, cols), dtype="float64")
    for start in range(0, rows, max(chunk_rows, 1)):
        stop = min(start + chunk_rows, rows)
        arr[start:stop] = data[start:stop]
    return data


def _tasks_of(lazy) -> int:
    import re

    match = re.search(r"(\d+) tasks", lazy.explain())
    return int(match.group(1)) if match else -1


def run(size: str, datadir: str) -> list[Measurement]:
    import os

    rows, cols = SIZES[size]
    out: list[Measurement] = []
    # Small fixed chunks: peak memory tracks chunk size, not data size.
    path = os.path.join(datadir, f"chunk-{size}.zarr")
    data = _make(path, rows, cols, 10_000, seed=11)
    X = P("X")
    transform = _sin(X) + X**2

    def tl_apply():
        return transform(T.scan_zarr(path)).collect()

    secs, alloc, got = measure(tl_apply)
    assert np.allclose(got, np.sin(data) + data**2)
    out.append(Measurement("chunked-apply", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="dask-chunked",
                           materializations=1,
                           note=f"input={data.nbytes / 1e6:.1f}MB"))

    def np_apply():
        whole = zarr.open_array(path, mode="r")[:]
        return np.sin(whole) + whole**2

    secs, alloc, ref = measure(np_apply)
    assert np.allclose(got, ref)
    out.append(Measurement("chunked-apply", size, "native", secs, alloc,
                           peak_rss_mb(), backend="numpy-direct-read",
                           note="native reads the whole store (unbounded)"))

    def tl_reduce():
        return transform(T.scan_zarr(path)).mean().collect()

    secs, alloc, mean = measure(tl_reduce)
    assert abs(mean - float((np.sin(data) + data**2).mean())) < 1e-9 * abs(mean)
    out.append(Measurement("chunked-reduce", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="dask-chunked",
                           materializations=1))

    def tl_write():
        transform(T.scan_zarr(path)).write_zarr(
            os.path.join(datadir, f"chunk-{size}-out.zarr"), overwrite=True)

    secs, alloc, _ = measure(tl_write)
    assert np.allclose(
        zarr.open_array(os.path.join(datadir, f"chunk-{size}-out.zarr"),
                        mode="r")[:],
        np.sin(data) + data**2)
    out.append(Measurement("chunked-write", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="dask-chunked",
                           materializations=1))
    out.extend(_run_m2(size, datadir))
    out.extend(_run_m3(size, datadir))
    out.extend(_run_m4(size, datadir))
    return out


def _run_m2(size: str, datadir: str) -> list[Measurement]:
    """Milestone 2: alignment, reshape/transpose, rechunk-heavy, matmul."""
    import os

    import dask.array as da

    from tensorlang import reshape as _reshape

    out: list[Measurement] = []
    rows, cols = SIZES[size]
    X, Y = P("X"), P("Y")

    # Aligned vs misaligned inputs (same values, different chunkings).
    rng = np.random.default_rng(31)
    data = rng.random((rows, cols)) + 0.5
    pa = os.path.join(datadir, f"m2-a-{size}.zarr")
    pb = os.path.join(datadir, f"m2-b-{size}.zarr")
    _make(pa, rows, cols, max(rows // 4, 1), seed=31)
    _make(pb, rows, cols, max(rows // 10, 1), seed=31)
    assert np.allclose(zarr.open_array(pa, mode="r")[:],
                       zarr.open_array(pb, mode="r")[:])
    la = (X + Y)(T.scan_zarr(pa), T.scan_zarr(pa))
    lb = (X + Y)(T.scan_zarr(pa), T.scan_zarr(pb))
    tasks_aligned = _tasks_of(la)
    tasks_misaligned = _tasks_of(lb)

    def tl_aligned():
        return la.collect()

    def tl_misaligned():
        return lb.collect()

    def np_native():
        return (da.from_zarr(pa) + da.from_zarr(pb)).compute()

    secs, alloc, got = measure(tl_aligned)
    assert np.allclose(got, data + data)
    out.append(Measurement("chunked-aligned", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="dask-chunked",
                           materializations=1,
                           note=f"tasks={tasks_aligned}"))
    secs, alloc, got = measure(tl_misaligned)
    assert np.allclose(got, data + data)
    mb_s = data.nbytes / 1e6
    out.append(Measurement("chunked-misaligned", size, "tensorlang", secs,
                           alloc, peak_rss_mb(), backend="dask-chunked",
                           materializations=1,
                           note=f"tasks={tasks_misaligned} "
                                f"throughput={mb_s / secs:.1f}MB/s"))
    secs, alloc, _ = measure(np_native)
    out.append(Measurement("chunked-misaligned", size, "native-dask", secs,
                           alloc, peak_rss_mb(), backend="dask",
                           note="raw dask baseline"))

    # Transpose / reshape vs native dask.
    from tensorlang import transpose as _transpose

    def tl_t():
        return _transpose(X)(T.scan_zarr(pa)).collect()

    secs, alloc, got = measure(tl_t)
    assert np.allclose(got, data.T)
    out.append(Measurement("chunked-transpose", size, "tensorlang", secs,
                           alloc, peak_rss_mb(), backend="dask-chunked",
                           materializations=1))
    flat_n = rows * cols

    def tl_r():
        return _reshape(X, (flat_n,))(T.scan_zarr(pa)).collect()

    secs, alloc, got = measure(tl_r)
    assert got.shape == (flat_n,)
    out.append(Measurement("chunked-reshape", size, "tensorlang", secs,
                           alloc, peak_rss_mb(), backend="dask-chunked",
                           materializations=1))

    # Bounded matmul subset vs native dask.
    k = min(cols, 8)
    A = rng.random((min(rows, 500), k)) + 0.5
    B = rng.random((k, 4)) + 0.5
    ma = os.path.join(datadir, f"m2-A-{size}.zarr")
    mb = os.path.join(datadir, f"m2-B-{size}.zarr")
    zarr.open_array(ma, mode="w", shape=A.shape,
                    chunks=(max(A.shape[0] // 2, 1), k),
                    dtype="float64")[:] = A
    zarr.open_array(mb, mode="w", shape=B.shape, chunks=(k, 4),
                    dtype="float64")[:] = B

    def tl_mm():
        return (X @ Y)(T.scan_zarr(ma), T.scan_zarr(mb)).collect()

    secs, alloc, got = measure(tl_mm)
    assert np.allclose(got, A @ B)
    mb_s = (A.nbytes + B.nbytes) / 1e6
    out.append(Measurement("chunked-matmul", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="dask-chunked",
                           materializations=1,
                           note=f"throughput={mb_s / secs:.1f}MB/s"))

    def np_mm():
        return (da.from_zarr(ma) @ da.from_zarr(mb)).compute()

    secs, alloc, _ = measure(np_mm)
    out.append(Measurement("chunked-matmul", size, "native-dask", secs, alloc,
                           peak_rss_mb(), backend="dask"))
    return out


def _run_m3(size: str, datadir: str) -> list[Measurement]:
    """Milestone 3: budget x chunk-size x concurrency; outputs larger than
    the working budget via streaming; native-dask parity under the same
    worker cap. Budgets are planning heuristics, never hard RSS claims."""
    import os

    import dask.array as da

    from tensorlang import clear_chunk_policy, set_chunk_policy
    from tensorlang.runtime import chunk_plan as _cp

    out: list[Measurement] = []
    rows, cols = SIZES[size]
    X = P("X")
    transform = _sin(X) + X**2

    for chunk_rows, workers, budget_mb in ((10_000, 4, 64),
                                           (2_000, 4, 16),
                                           (2_000, 1, 16)):
        tag = f"c{chunk_rows}-w{workers}-b{budget_mb}MB"
        path = os.path.join(datadir, f"m3-{tag}-{size}.zarr")
        data = _make(path, rows, cols, chunk_rows, seed=41)
        set_chunk_policy(memory_budget_bytes=budget_mb * 1024**2,
                         max_workers=workers)
        try:
            lazy = transform(T.scan_zarr(path))
            tasks = _tasks_of(lazy)
            # Purpose-dependent admission: small sizes admit; larger data
            # under the same fixed budget is rejected (TL1024/TL1020) —
            # the decisive budget demonstration, recorded not crashed on.
            try:
                secs, alloc, got = measure(lazy.collect)
                assert np.allclose(got, np.sin(data) + data**2)
                decision = dict(_cp.LAST_DECISION)
                out.append(Measurement(
                    "chunked-budget", size, "tensorlang", secs, alloc,
                    peak_rss_mb(), backend="dask-chunked", materializations=1,
                    note=f"{tag} tasks={tasks} "
                         f"outcome={decision.get('outcome')} "
                         f"est={decision.get('estimated_peak_bytes', 0) / 1e6:.2f}MB "
                         f"workers={decision.get('workers_used', '?')} "
                         f"throughput={data.nbytes / 1e6 / secs:.1f}MB/s"))

                def np_native(p=path, w=workers):
                    return (da.sin(da.from_zarr(p))
                            + da.from_zarr(p) ** 2).compute(num_workers=w)

                secs, alloc, _ = measure(np_native)
                out.append(Measurement("chunked-budget", size, "native-dask",
                                       secs, alloc, peak_rss_mb(),
                                       backend="dask",
                                       note=f"{tag} raw dask, same worker cap"))
            except Exception as exc:
                code = getattr(exc, "code", "?")
                assert code in ("TL1020", "TL1024"), str(exc)[:200]
                out.append(Measurement(
                    "chunked-budget", size, "tensorlang", 0.0, 0.0,
                    peak_rss_mb(), backend="dask-chunked", materializations=0,
                    note=f"{tag} tasks={tasks} rejected {code}: "
                         f"{data.nbytes / 1e6:.1f}MB output vs "
                         f"{budget_mb}MB budget"))
            # Reduction under the same budget admits at every data size
            # (scalar output; only the chunk-sized working set matters).
            secs, alloc, got = measure(lambda: lazy.mean().collect())
            assert abs(got - float((np.sin(data) + data**2).mean())) < 1e-6 * abs(
                float((np.sin(data) + data**2).mean())) + 1e-9
            out.append(Measurement(
                "chunked-budget-reduce", size, "tensorlang", secs, alloc,
                peak_rss_mb(), backend="dask-chunked", materializations=1,
                note=f"{tag} reduce {data.nbytes / 1e6:.1f}MB under "
                     f"{budget_mb}MB budget"))
        finally:
            clear_chunk_policy()

    # Output larger than the working budget: reduction admits, streaming
    # writes, full collect is rejected (TL1024) — same data, three purposes.
    big_path = os.path.join(datadir, f"m3-big-{size}.zarr")
    big = _make(big_path, rows, cols, 2_000, seed=42)
    set_chunk_policy(memory_budget_bytes=2 * 1024**2, max_workers=2)
    try:
        lazy = transform(T.scan_zarr(big_path))
        secs, alloc, got = measure(lambda: lazy.mean().collect())
        assert abs(got - float((np.sin(big) + big**2).mean())) < 1e-6 * abs(
            float((np.sin(big) + big**2).mean())) + 1e-9
        out.append(Measurement(
            "chunked-over-budget", size, "tensorlang", secs, alloc,
            peak_rss_mb(), backend="dask-chunked", materializations=1,
            note=f"reduce {big.nbytes / 1e6:.1f}MB under 2MB budget"))
        import shutil

        sink = os.path.join(datadir, f"m3-sink-{size}.zarr")
        shutil.rmtree(sink, ignore_errors=True)
        secs, alloc, _ = measure(
            lambda: lazy.write_zarr(sink, overwrite=True))
        out.append(Measurement(
            "chunked-over-budget", size, "tensorlang", secs, alloc,
            peak_rss_mb(), backend="dask-chunked", materializations=1,
            note=f"stream {big.nbytes / 1e6:.1f}MB output under 2MB budget"))
        try:
            lazy.collect()
            raise AssertionError("collect should have been rejected (TL1024)")
        except Exception as exc:
            assert "TL1024" in str(exc), str(exc)[:200]
    finally:
        clear_chunk_policy()
    return out


def _run_m4(size: str, datadir: str) -> list[Measurement]:
    """Milestone 4: staged publication overhead vs direct streaming.

    Same logical output both ways; reports wall time, Python allocation,
    RSS, throughput, publication split, transient disk overhead, and
    cleanup time after an injected mid-write failure. Overhead is
    reported, never hidden.
    """
    import os
    import shutil
    import time as _time

    from tensorlang.runtime import chunk_plan as _cp
    from tensorlang.runtime import execution as _ex

    out: list[Measurement] = []
    rows, cols = SIZES[size]
    X = P("X")
    transform = _sin(X) + X**2
    path = os.path.join(datadir, f"m4-{size}.zarr")
    data = _make(path, rows, cols, 10_000, seed=51)
    lazy = transform(T.scan_zarr(path))
    expected = np.sin(data) + data**2

    for staged, label in ((True, "staged"), (False, "direct")):
        sink = os.path.join(datadir, f"m4-sink-{label}-{size}.zarr")
        shutil.rmtree(sink, ignore_errors=True)

        def tl_write(s=sink, st=staged):
            return lazy.write_zarr(s, overwrite=True, staged=st)

        secs, alloc, _ = measure(tl_write)
        assert np.allclose(zarr.open_array(sink, mode="r")[:], expected)
        rec = _ex.last_record()
        store_mb = expected.nbytes / 1e6
        disk_note = (f"final={store_mb:.1f}MB transient~{2 * store_mb:.1f}MB"
                     if staged else f"final={store_mb:.1f}MB transient~0MB")
        out.append(Measurement(
            "chunked-write-mode", size, "tensorlang", secs, alloc,
            peak_rss_mb(), backend="dask-chunked", materializations=1,
            note=f"{label} publish={rec['timings_ms'].get('publish_ms', 0.0):.1f}ms "
                 f"validate={rec['timings_ms'].get('validate_ms', 0.0):.1f}ms "
                 f"{disk_note} throughput={store_mb / secs:.1f}MB/s"))

    # Cleanup time after an injected mid-write failure (staged).
    sink = os.path.join(datadir, f"m4-fail-{size}.zarr")
    shutil.rmtree(sink, ignore_errors=True)
    _cp._TEST_HOOKS["fail_after_partitions"] = 1
    try:
        start = _time.perf_counter()
        try:
            lazy.write_zarr(sink)
        except Exception as exc:  # noqa: BLE001 - measuring failure path
            assert "TL1031" in str(exc), str(exc)[:200]
        cleanup_secs = _time.perf_counter() - start
    finally:
        _cp._TEST_HOOKS["fail_after_partitions"] = None
    assert not os.path.exists(sink)
    assert _cp.list_stale_staging(sink) == []
    out.append(Measurement("chunked-write-cleanup", size, "tensorlang",
                           cleanup_secs, 0.0, peak_rss_mb(),
                           backend="dask-chunked", materializations=0,
                           note="fail-after-1-partition abort+cleanup, "
                                "destination absent, no staging leaked"))
    return out
