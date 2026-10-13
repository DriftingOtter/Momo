"""v0.5 M5 tests: recovery state table, hook isolation, and release audit."""
import os
import shutil

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
    rng = np.random.default_rng(43)
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


def _validated_staging(src_data_path, dest, chunks=(10, 8)):
    """A complete, census-valid staging dir not yet published."""
    from tensorlang.backends import dask_backend as _be

    staging = _be.staging_dir_for(dest)
    arr = zarr.open_array(staging, mode="w", shape=(40, 8),
                          chunks=chunks, dtype="float64")
    arr[:] = np.ones((40, 8))
    return staging


def test_recover_clean_state(tmp_path):
    dest = str(tmp_path / "nothing.zarr")
    report = _cp.recover(dest)
    assert report["removed"] == [] and report["retained"] == []
    assert report["published"] is None
    assert any("clean" in action for action in report["actions"])


def test_recover_removes_incomplete_staging(store):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    staging = _validated_staging(path, dest)
    # Corrupt it: drop a chunk file -> incomplete census.
    files = []
    for root, _dirs, names in os.walk(staging):
        files.extend(os.path.join(root, n) for n in names
                     if n not in ("zarr.json", ".zgroup", ".zarray"))
    os.remove(sorted(files)[0])
    report = _cp.recover(dest)
    assert report["removed"] == [staging]
    assert not os.path.exists(staging)
    assert not os.path.exists(dest)


def test_recover_retains_only_valid_copy(store):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    staging = _validated_staging(path, dest)
    report = _cp.recover(dest)
    assert report["retained"] == [staging]  # never auto-delete only copy
    assert os.path.exists(staging)
    # Idempotent: second run reports the same.
    again = _cp.recover(dest)
    assert again["retained"] == [staging]


def test_recover_publishes_validated_when_absent(store):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    staging = _validated_staging(path, dest)
    report = _cp.recover(dest, publish_validated=True)
    assert report["published"] == staging
    assert np.allclose(zarr.open_array(dest, mode="r")[:], np.ones((40, 8)))
    assert _cp.recover(dest)["actions"] != []


def test_recover_publish_never_clobbers_without_overwrite(store):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    mstore(dest, data, (10, 8))
    staging = _validated_staging(path, dest)
    report = _cp.recover(dest, publish_validated=True)
    assert report["published"] is None
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data)
    report = _cp.recover(dest, publish_validated=True, overwrite=True)
    assert report["published"] == staging
    assert np.allclose(zarr.open_array(dest, mode="r")[:], np.ones((40, 8)))


def test_recover_restores_backup_when_dest_absent(store):
    import shutil as _shutil

    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    mstore(dest, data, (10, 8))
    from tensorlang.backends import dask_backend as _be

    backup = _be.backup_path_for(dest)
    os.rename(dest, backup)  # simulate crash between the two renames
    assert not os.path.exists(dest)
    report = _cp.recover(dest)
    assert any("restored backup" in a for a in report["actions"])
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data)
    # Idempotent: destination now valid, nothing left to do.
    again = _cp.recover(dest)
    assert again["destination_valid"] is True


def test_recover_drops_backup_after_valid_dest(store):
    path, data = store
    dest = str(path).replace("a.zarr", "out.zarr")
    mstore(dest, data, (10, 8))
    from tensorlang.backends import dask_backend as _be

    backup = _be.backup_path_for(dest)
    shutil.copytree(dest, backup)  # crash after publish, before cleanup
    report = _cp.recover(dest)
    assert report["removed"] == [backup]
    assert np.allclose(zarr.open_array(dest, mode="r")[:], data)


def test_recover_ambiguous_keeps_everything_valid(store):
    path, _ = store
    dest = str(path).replace("a.zarr", "out.zarr")
    first = _validated_staging(path, dest)
    second = _validated_staging(path, dest)
    report = _cp.recover(dest)
    assert sorted(report["retained"]) == sorted([first, second])
    assert report["removed"] == []


def test_self_overwrite_rejected(store, hooks):
    path, _ = store
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(path, overwrite=True)
    assert exc.value.code == "TL1031"
    assert exc.value.details["conflict"] == "source==destination"
    # Without overwrite the existence gate fires first, source intact.
    with pytest.raises(Exception) as exc2:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(path)
    assert exc2.value.code == "TL1002"


def test_hooks_inaccessible_through_public_api(store, hooks):
    # Public write paths accept no hook/failure arguments.
    import inspect as _inspect

    from tensorlang import Tensor as _Tensor

    params = set(_inspect.signature(_Tensor.write_zarr).parameters)
    assert params == {"self", "path", "overwrite", "staged"}
    lazy_params = set(_inspect.signature(
        __import__("tensorlang").Lazy.write_zarr).parameters)
    assert lazy_params == {"self", "path", "overwrite", "staged"}
    # Hooks live behind a private name and reset cleanly.
    assert _cp._TEST_HOOKS["fail_after_partitions"] is None


def test_write_zarr_staged_kwarg_validated(store, hooks):
    path, data = store
    remote = "s3://bucket/out.zarr"
    # describe_write never raises: unavailability is reported as text.
    text = (P("X") + 1)(T.scan_zarr(path)).describe_write(remote, staged=True)
    assert "unavailable" in text
    text = (P("X") + 1)(T.scan_zarr(path)).describe_write(remote)
    assert "direct" in text and "non-local" in text
    # The execution boundary itself rejects staged-on-remote (TL1031).
    with pytest.raises(TLUnsupportedError) as exc:
        (P("X") + 1)(T.scan_zarr(path)).write_zarr(remote, staged=True)
    assert exc.value.code == "TL1031"
