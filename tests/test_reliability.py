"""v0.5 M4 tests: execution reliability — lifecycle states, staged
publication integrity, failure injection, interruption, source
consistency, and resource isolation.

Failure injection uses the test-only hooks in chunk_plan (reset by the
``hooks`` fixture below); production paths never consult them.
"""
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import zarr

from tensorlang import P, T, TLDeclarationError, TLUnsupportedError
from tensorlang.runtime import chunk_plan as _cp
from tensorlang.runtime import execution as _ex


def mstore(path, data, chunks):
    arr = zarr.open_array(path, mode="w", shape=data.shape,
                          chunks=chunks, dtype="float64")
    arr[:] = data
    return path


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(41)
    data = rng.random((40, 8)) + 0.5
    path = mstore(str(tmp_path / "a.zarr"), data, (10, 8))
    return path, data


@pytest.fixture
def hooks():
    saved = dict(_cp._TEST_HOOKS)
    _cp._TEST_HOOKS.update({"fail_before_write": False,
                            "fail_after_partitions": None,
                            "fail_validation": False,
                            "fail_publication": False,
                            "raise_interrupt_at": None,
                            "slow_partition_secs": 0.0})
    try:
        yield _cp._TEST_HOOKS
    finally:
        _cp._TEST_HOOKS.update(saved)
        _cp.clear_chunk_policy()


def _stale(dest):
    return _cp.list_stale_staging(dest)


def test_staged_write_lifecycle_and_splits(store, hooks):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data + 1)
    rec = _ex.last_record()
    assert rec["stage"] == "completed"
    assert rec["kind"] == "write" and rec["strategy"] == "staged"
    assert {"write_ms", "validate_ms", "publish_ms"} <= set(rec["timings_ms"])
    assert rec["error"] == {}
    assert _stale(dest) == []


def test_collect_lifecycle_completed(store, hooks):
    path, data = store
    assert np.allclose((P("X") + 1)(T.scan_zarr(path)).collect(), data + 1)
    rec = _ex.last_record()
    assert rec["stage"] == "completed" and rec["kind"] == "collect"
    assert rec["timings_ms"]["compute_ms"] >= 0.0


def test_run_ids_unique_and_per_run(store, hooks):
    path, _ = store
    (P("X") + 1)(T.scan_zarr(path)).collect()
    first = _ex.last_record()["run_id"]
    (P("X") + 2)(T.scan_zarr(path)).collect()
    second = _ex.last_record()["run_id"]
    assert first and second and first != second


def test_failure_before_first_write(store, hooks):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    hooks["fail_before_write"] = True
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1031"
    assert not os.path.exists(dest)
    assert _stale(dest) == []
    assert _ex.last_record()["stage"] == "failed"


def test_failure_after_some_chunks(store, hooks):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    hooks["fail_after_partitions"] = 2  # 2 of 4 partitions staged, then abort
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1031"
    assert exc.value.details["written_partitions"] == 2
    assert not os.path.exists(dest)
    assert _stale(dest) == []  # partial staging cleaned, not leaked
    # Recovery works after the failed run.
    hooks["fail_after_partitions"] = None
    (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data + 1)


def test_failure_during_validation(store, hooks):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    hooks["fail_validation"] = True
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1034"
    assert not os.path.exists(dest)
    assert _stale(dest) == []


def test_failure_during_publication_retains_validated(store, hooks):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    hooks["fail_publication"] = True
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1031"
    assert not os.path.exists(dest)
    stale = _stale(dest)
    assert len(stale) == 1  # complete output retained, not deleted
    assert np.allclose(zarr.open_array(stale[0], mode="r")[:], data + 1)
    report = _cp.recover_stale_staging(dest)
    assert report["removed"] == stale and _stale(dest) == []


def test_destination_exists_preserved(store, hooks):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    before = zarr.open_array(dest, mode="r")[:]
    with pytest.raises(Exception) as exc:
        (P("X") + 2)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1002"
    assert np.allclose(zarr.open_array(dest, mode="r")[:], before)
    assert _stale(dest) == []


def test_overwrite_failure_preserves_previous(store, hooks):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    before = zarr.open_array(dest, mode="r")[:]
    hooks["fail_after_partitions"] = 1
    with pytest.raises(TLUnsupportedError):
        (P("X") + 2)(T.scan_zarr(path)).write_zarr(dest, overwrite=True)
    # The previous valid destination survived; no backup debris remains.
    assert np.allclose(zarr.open_array(dest, mode="r")[:], before)
    assert _stale(dest) == []
    assert [p for p in os.listdir(os.path.dirname(dest))
            if ".tlbackup-" in p] == []
    hooks["fail_after_partitions"] = None
    (P("X") + 2)(T.scan_zarr(path)).write_zarr(dest, overwrite=True)
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data + 2)


def test_concurrent_publication_one_winner(store, hooks):
    import shutil

    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    hooks["slow_partition_secs"] = 0.05
    outcomes, errors = [], []

    def attempt():
        try:
            (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
            outcomes.append("published")
        except Exception as exc:  # noqa: BLE001 - asserting below
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == ["published"] and len(errors) == 1
    assert errors[0].code == "TL1002"
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data + 1)
    # Loser's validated staging is recoverable, then gone.
    stale = _stale(dest)
    assert len(stale) == 1
    assert "staging" in errors[0].details
    shutil.rmtree(stale[0], ignore_errors=True)
    assert _stale(dest) == []


def test_invalid_destination(store, hooks):
    path, _ = store
    dest = os.path.join(str(path) + "-is-a-file", "out.zarr")
    with open(str(path) + "-is-a-file", "w"):
        pass
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1031"
    assert exc.value.details["destination"] == dest


def test_direct_mode_documents_partial(store, hooks):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    hooks["fail_after_partitions"] = 2
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest, staged=False)
    assert exc.value.code == "TL1031"
    # Direct streaming leaves the documented weaker state: a partial store.
    assert os.path.exists(dest)
    text = (P("X") + 1)(T.scan_zarr(path)).describe_write(dest, staged=False)
    assert "direct" in text


def test_validation_rejects_corrupt_store(tmp_path):
    from tensorlang.backends import dask_backend as _be

    rng = np.random.default_rng(5)
    data = rng.random((20, 4))
    src = mstore(str(tmp_path / "src.zarr"), data, (10, 4))
    report = _be.validate_store(src, [20, 4], "float64", 2)
    assert report["problems"] == []
    # Corrupt: remove one chunk file -> incomplete census, metadata intact.
    chunks = []
    for root, _dirs, files in os.walk(src):
        chunks.extend(os.path.join(root, name) for name in files
                      if name not in ("zarr.json", ".zgroup", ".zarray",
                                      ".zattrs"))
    assert chunks, os.listdir(src)
    os.remove(sorted(chunks)[0])
    report = _be.validate_store(src, [20, 4], "float64", 2)
    assert any("incomplete" in problem for problem in report["problems"])


def test_source_shape_change_rejected(store, hooks):
    path, _ = store
    (P("X") + 1)(T.scan_zarr(path)).collect()  # populates metadata cache
    rng = np.random.default_rng(6)
    new = rng.random((25, 8))
    mstore(path, new, (10, 8))  # external rewrite, cache untouched
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).collect()
    assert exc.value.code == "TL1030"
    assert exc.value.details["path"] == path
    assert _ex.last_record()["stage"] == "failed"


def test_source_mtime_change_rejected(store, hooks):
    import time as _time

    path, data = store
    (P("X") + 1)(T.scan_zarr(path)).collect()
    # Same shape, new bytes, forced stamp movement: best-effort detection.
    mstore(path, data + 100.0, (10, 8))
    stamp = _time.time() + 30
    os.utime(path, (stamp, stamp))
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).collect()
    assert exc.value.code == "TL1030"


def test_in_memory_snapshot_stable(store, hooks):
    from tensorlang import T as _T

    snap = _T(np.ones((4, 4)))
    hooks["fail_before_write"] = True
    with pytest.raises(TLUnsupportedError):
        (P("X") + 1)(_T.scan_zarr(store[0])).write_zarr(
            str(store[0]).replace("a.zarr", "out.zarr"))
    assert np.allclose(snap.collect(), np.ones((4, 4)))


def test_interrupted_execution_cancelled(store, hooks):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    with _cp.injected_failures(raise_interrupt_at=1):
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    assert exc.value.code == "TL1032"
    assert not os.path.exists(dest)
    assert _stale(dest) == []
    assert _ex.last_record()["stage"] == "cancelled"
    assert _cp._TEST_HOOKS["raise_interrupt_at"] is None


def test_injected_failures_rejects_unknown_keys():
    with pytest.raises(KeyError):
        with _cp.injected_failures(no_such_hook=True):
            pass


def test_cleanup_removes_reappearing_staging(tmp_path, monkeypatch):
    """Regression: an interrupted native call may finish creating the
    staging directory after the first removal (its event loop was only
    interrupted out of ``wait``, not cancelled). Cleanup verifies and
    removes the reappearance instead of leaving it behind."""
    from tensorlang.backends import dask_backend as _be

    staging = tmp_path / "dest.zarr.tlstage-x"
    staging.mkdir()
    calls = []

    real_remove = _be.remove_path

    def flaky_remove(path):
        calls.append(path)
        real_remove(path)
        if len(calls) == 1:
            # Simulate the native loop thread completing store
            # creation just after cleanup removed the directory.
            os.makedirs(path, exist_ok=True)

    monkeypatch.setattr(_be, "remove_path", flaky_remove)
    _cp._cleanup_staging(str(staging))  # must not raise
    assert calls and not os.path.exists(staging)


def test_cleanup_missing_staging_is_silent(tmp_path):
    _cp._cleanup_staging(str(tmp_path / "dest.zarr.tlstage-absent"))
    _cp._cleanup_staging(None)
    _cp._cleanup_staging("")


@pytest.mark.skipif(sys.platform == "win32",
                        reason="POSIX signal delivery; Windows untested")
def test_interruption_subprocess_preserves_destination(tmp_path):
    src = str(tmp_path / "src.zarr")
    dest = str(tmp_path / "dest.zarr")
    rng = np.random.default_rng(9)
    data = rng.random((60, 8)) + 0.5
    mstore(src, data, (6, 8))  # 10 partitions
    child = (
        "import time\n"
        "from tensorlang import P, T\n"
        "from tensorlang.runtime import chunk_plan as cp\n"
        "cp._TEST_HOOKS['slow_partition_secs'] = 0.3\n"
        f"(P('X') + 1)(T.scan_zarr({src!r})).write_zarr({dest!r})\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", child])
    try:
        # Wait until staging exists: interrupt is guaranteed mid-write.
        for _ in range(200):
            if [p for p in os.listdir(tmp_path)
                    if p.startswith("dest.zarr.tlstage-")]:
                break
            time.sleep(0.05)
        proc.send_signal(2)  # SIGINT
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode != 0
    assert not os.path.exists(dest)  # never published partial output
    # SIGINT runs Python-level cleanup (unlike SIGKILL): staging removed.
    assert [p for p in os.listdir(tmp_path)
            if p.startswith("dest.zarr.tlstage-")] == []
    from tensorlang.backends import dask_backend as _be

    assert _be.list_stale_staging(dest) == []


@pytest.mark.skipif(sys.platform == "win32",
                        reason="POSIX signals; Windows untested")
def test_sigkill_leaves_recoverable_staging(tmp_path):
    src = str(tmp_path / "src.zarr")
    dest = str(tmp_path / "dest.zarr")
    rng = np.random.default_rng(11)
    mstore(src, rng.random((60, 8)) + 0.5, (6, 8))
    child = (
        "from tensorlang import P, T\n"
        "from tensorlang.runtime import chunk_plan as cp\n"
        "cp._TEST_HOOKS['slow_partition_secs'] = 0.3\n"
        f"(P('X') + 1)(T.scan_zarr({src!r})).write_zarr({dest!r})\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", child])
    try:
        for _ in range(200):
            if [p for p in os.listdir(tmp_path)
                    if p.startswith("dest.zarr.tlstage-")]:
                break
            time.sleep(0.05)
        proc.kill()  # SIGKILL: no cleanup runs by design
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert not os.path.exists(dest)
    report = _cp.recover_stale_staging(dest, remove=False)
    assert len(report["stale"]) == 1 and report["removed"] == []
    report = _cp.recover_stale_staging(dest)
    assert len(report["removed"]) == 1
    assert _cp.list_stale_staging(dest) == []


def test_failed_run_preserves_policy_and_release(store, hooks):
    from tensorlang import set_chunk_policy as _set

    path, data = store
    _set(memory_budget_bytes=1024)
    lazy = (P("X") + 1)(T.scan_zarr(path))
    with pytest.raises(TLUnsupportedError):
        lazy.collect()
    assert _cp.current_policy()["memory_budget_bytes"] == 1024
    assert lazy.release() is lazy
    _cp.clear_chunk_policy()
    assert np.allclose(lazy.collect(), data + 1)


def test_zero_size_staged_write(tmp_path, hooks):
    data = np.zeros((0, 4))
    src = mstore(str(tmp_path / "z.zarr"), data, (4, 4))
    dest = str(tmp_path / "out.zarr")
    (P("X") + 1)(T.scan_zarr(src)).write_zarr(dest)
    assert zarr.open_array(dest, mode="r").shape == (0, 4)


def test_describe_write_is_read_only(store, hooks):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    lazy = (P("X") + 1)(T.scan_zarr(path))
    text = lazy.describe_write(dest)
    assert "staged" in text and "admission" in text
    assert not os.path.exists(dest)  # nothing created, nothing executed
    assert "direct" in lazy.describe_write(dest, staged=False)
    mstore(dest, np.ones((4, 2)), (2, 2))
    assert "exists" in lazy.describe_write(dest)
    assert "TL1002" in lazy.describe_write(dest)


def test_describe_write_under_declare_is_declarative(tmp_path):
    from tensorlang import run_source

    src = mstore(str(tmp_path / "s.zarr"), np.ones((4, 2)), (2, 2))
    ns = run_source(
        "from tensorlang import T\n"
        f"d := T.scan_zarr({str(src)!r})\n"
        "w := (d + 1).describe_write('/tmp/m4-desc-check.zarr')\n")
    assert "staged" in ns["w"]


def test_write_zarr_under_declare_rejected():
    from tensorlang import run_source

    with pytest.raises(TLDeclarationError) as exc:
        run_source("from tensorlang import T\n"
                   "t := T({'a': [1]})\n"
                   "p := t.write_zarr('/tmp/m4-decl-check.zarr')\n")
    assert "effectful" in str(exc.value)


def test_error_details_carry_provenance(store, hooks):
    path, _ = store
    dest = os.path.join(str(path) + "-is-a-file", "out.zarr")
    with open(str(path) + "-is-a-file", "w"):
        pass
    try:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(dest)
    except TLUnsupportedError as exc:
        assert exc.details["destination"] == dest
        assert "backend_error" in exc.details
        assert exc.details["operation"] == "write_zarr"
    else:  # pragma: no cover - must fail
        raise AssertionError("expected TL1031")
