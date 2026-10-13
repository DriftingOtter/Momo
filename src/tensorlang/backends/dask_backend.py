"""Dask/Zarr backend: persistent chunked array sources and sinks.

Chunking is a physical execution property — operators always see one
logical tensor. Nothing here reads array data at plan time; only store
metadata (shape/dtype/chunks) is inspected, and it is cached per path.
"""
from __future__ import annotations

import functools
import os

from ..core.errors import TLTypeError
from ..core.types import ChunkSpec


def _require_libs():
    try:
        import dask.array as da
        import zarr
    except ImportError as exc:
        raise TLTypeError(
            "chunked execution needs the zarr and dask packages; "
            f"install them ({exc})") from exc
    return da, zarr


@functools.lru_cache(maxsize=128)
def describe(path: str) -> ChunkSpec:
    """Store metadata without reading array data."""
    _, zarr = _require_libs()
    try:
        store = zarr.open_array(path, mode="r")
    except Exception as exc:
        raise TLTypeError(f"cannot open chunked store '{path}': {exc}",
                          code="TL1003", details={"path": path}) from exc
    shape = tuple(int(d) for d in store.shape)
    chunks = tuple(int(c) for c in (store.chunks or shape)) or shape
    return ChunkSpec(path=path, shape=shape, dtype=str(store.dtype),
                     chunks=chunks)


def from_spec(spec: ChunkSpec):
    """A lazy dask collection over the store (no data read)."""
    da, _ = _require_libs()
    try:
        return da.from_zarr(spec.path, chunks=spec.chunks)
    except Exception as exc:
        raise TLTypeError(f"cannot scan chunked store '{spec.path}': {exc}",
                          code="TL1003", details={"path": spec.path}) from exc


def from_array(array, chunks="auto"):
    """A lazy dask collection over in-memory data (no copy at plan time)."""
    da, _ = _require_libs()
    import numpy as _np

    return da.from_array(_np.asarray(array), chunks=chunks)


def read_all(path: str):
    """Explicit boundary: materialize a whole store (may exceed memory)."""
    import numpy as _np

    return _np.asarray(from_spec(describe(path)).compute())


def write_array(graph, path: str, overwrite: bool = False,
                num_workers: int | None = None) -> str:
    """Stream a dask graph into a store without full materialization.

    ``num_workers`` caps worker threads for this sink only (per-call
    argument; never global config). ``None`` keeps the default path.
    Writes are incremental, not atomic: an interrupted write may leave a
    partial store behind.
    """
    import dask.array as _da

    if os.path.exists(path) and not overwrite:
        raise TLTypeError(
            f"chunked write target '{path}' already exists; pass "
            f"overwrite=True to replace it or choose a fresh path",
            code="TL1002", details={"path": path})
    try:
        if num_workers is None:
            _da.to_zarr(graph, path, overwrite=overwrite, compute=True)
        else:
            delayed = _da.to_zarr(graph, path, overwrite=overwrite,
                                  compute=False)
            try:
                delayed.compute(num_workers=int(num_workers))
            except TypeError:
                delayed.compute()  # backend ignores the cap
    except TLTypeError:
        raise
    except Exception as exc:
        if "exists" in str(exc).lower() or "contains" in str(exc).lower():
            raise TLTypeError(
                f"chunked write target '{path}' already exists; pass "
                f"overwrite=True to replace it",
                code="TL1002", details={"path": path}) from exc
        raise TLTypeError(f"cannot write chunked store '{path}': {exc}",
                          code="TL1003", details={"path": path}) from exc
    describe.cache_clear()  # the store may have been replaced
    return path


def task_count(graph) -> int:
    try:
        return len(dict(graph.__dask_graph__()))
    except Exception:  # pragma: no cover - defensive
        return -1


# -- persistent integrity (v0.5 M4) ------------------------------------------------------
# Staged publication for local stores; metadata-only validation. Remote
# (URI-scheme) stores keep direct streaming with a documented warning —
# no distributed transaction system is introduced.

_REMOTE_PREFIXES = ("s3://", "gs://", "gcs://", "az://", "abfs://",
                    "hdfs://", "http://", "https://")


def is_local_store(path: str) -> bool:
    lowered = str(path).lower()
    if "://" in str(path) and lowered.split("://", 1)[0].isalpha():
        return False
    return not str(path).startswith(_REMOTE_PREFIXES)


def fingerprint(path: str) -> dict:
    """Fresh store metadata + filesystem stamp (no data read).

    Used to detect external source changes between planning and execution.
    Best-effort: shape/dtype/chunks changes are always caught; content
    changes without metadata or mtime movement may not be.
    """
    _, zarr = _require_libs()
    try:
        store = zarr.open_array(str(path), mode="r")
        meta = {"shape": tuple(int(d) for d in store.shape),
                "dtype": str(store.dtype),
                "chunks": tuple(int(c) for c in (store.chunks or store.shape)),
                "nchunks": int(store.nchunks)}
    except Exception as exc:
        raise TLTypeError(f"cannot fingerprint store '{path}': {exc}",
                          code="TL1003", details={"path": str(path)}) from exc
    try:
        stamp = os.stat(str(path))
        meta["mtime_ns"] = stamp.st_mtime_ns
        meta["size_bytes"] = stamp.st_size
    except OSError:
        meta["mtime_ns"] = -1
        meta["size_bytes"] = -1
    meta["path"] = str(path)
    return meta


def staging_dir_for(dest: str) -> str:
    import uuid as _uuid

    return f"{dest}.tlstage-{_uuid.uuid4().hex[:8]}"


def backup_path_for(dest: str) -> str:
    import uuid as _uuid

    return f"{dest}.tlbackup-{_uuid.uuid4().hex[:8]}"


def list_stale_staging(dest: str) -> list[str]:
    """Staging/backup artifacts left by interrupted runs (no reads)."""
    import glob as _glob

    found = []
    for pattern in (f"{dest}.tlstage-*", f"{dest}.tlbackup-*"):
        found.extend(p for p in _glob.glob(pattern) if os.path.exists(p))
    return sorted(found)


def census_complete(path: str) -> tuple[bool, dict]:
    """Self-consistency check: every chunk the store claims exists is
    initialized (metadata only; NOT a checksum — garbage bytes inside a
    present chunk file are undetectable here). Returns (ok, report)."""
    try:
        report = validate_store(path, None, None, -1)  # type: ignore[arg-type]
    except TLTypeError as exc:
        return False, {"path": str(path), "unreadable": str(exc)}
    actual_nchunks = report["actual_nchunks"]
    initialized = report["initialized"]
    ok = initialized == actual_nchunks
    return ok, {"path": str(path), "nchunks": actual_nchunks,
                "initialized": initialized,
                "shape": report["actual_shape"],
                "dtype": report["actual_dtype"]}


def remove_path(path: str) -> None:
    import shutil as _shutil

    if os.path.isdir(path) and not os.path.islink(path):
        _shutil.rmtree(path, ignore_errors=False)
    elif os.path.exists(path) or os.path.islink(path):
        os.remove(path)


def disk_free_bytes(path: str) -> int:
    import shutil as _shutil

    parent = os.path.dirname(os.path.abspath(str(path))) or "."
    try:
        return _shutil.disk_usage(parent).free
    except OSError:  # pragma: no cover - defensive
        return -1


def validate_store(path: str, shape, dtype_str: str,
                   expected_nchunks: int) -> dict:
    """Metadata-only output validation (never reads array data).

    Returns a report; raises TLTypeError(TL1003) when unreadable. Chunk
    presence is counted via chunk-key existence, not data reads. Pass
    ``shape=None`` for a self-consistency census only (no expected
    metadata to compare against).
    """
    _, zarr = _require_libs()
    try:
        store = zarr.open_array(str(path), mode="r")
        actual = {"shape": tuple(int(d) for d in store.shape),
                  "dtype": str(store.dtype),
                  "nchunks": int(store.nchunks),
                  "initialized": int(store.nchunks_initialized)}
    except Exception as exc:
        raise TLTypeError(f"cannot validate store '{path}': {exc}",
                          code="TL1003", details={"path": str(path)}) from exc
    report = {"path": str(path),
              "expected_shape": list(shape) if shape is not None else None,
              "actual_shape": list(actual["shape"]),
              "expected_dtype": (str(dtype_str)
                                 if dtype_str is not None else None),
              "actual_dtype": actual["dtype"],
              "expected_nchunks": (int(expected_nchunks)
                                   if expected_nchunks is not None and
                                   int(expected_nchunks) >= 0 else None),
              "actual_nchunks": actual["nchunks"],
              "initialized": actual["initialized"]}
    problems = []
    if shape is not None and list(actual["shape"]) != list(shape):
        problems.append(f"shape {list(actual['shape'])} != {list(shape)}")
    if dtype_str is not None and actual["dtype"] != str(dtype_str):
        problems.append(f"dtype {actual['dtype']} != {dtype_str}")
    if (expected_nchunks is not None and int(expected_nchunks) >= 0
            and actual["initialized"] != int(expected_nchunks)):
        problems.append(f"{actual['initialized']}/{expected_nchunks} chunks "
                        f"initialized (incomplete output)")
    report["problems"] = problems
    return report


def publish_staged(staging: str, dest: str, overwrite: bool) -> str:
    """Publish a validated staging store (same filesystem only).

    Each ``os.rename`` is atomic on POSIX, but a backup-based overwrite
    (rename aside, rename into place, remove backup) is a recoverable
    multi-step replacement — NOT one continuously atomic overwrite (see
    LANGUAGE_SPEC §16). Remote stores are never published this way.
    """
    dest, staging = str(dest), str(staging)
    if os.path.exists(dest) and not overwrite:
        raise TLTypeError(
            f"chunked write target '{dest}' already exists; pass "
            f"overwrite=True to replace it or choose a fresh path",
            code="TL1002", details={"path": dest})
    backup = None
    try:
        if os.path.exists(dest) and overwrite:
            backup = backup_path_for(dest)
            os.rename(dest, backup)
        os.rename(staging, dest)
    except TLTypeError:
        raise
    except Exception as exc:
        # A rename onto an existing destination means it appeared between
        # our precheck and publication: concurrent-publication loss, not a
        # generic I/O error (staging is complete and retained by callers).
        if os.path.exists(dest) and not overwrite:
            raise TLTypeError(
                f"chunked write target '{dest}' appeared during publication "
                f"(concurrent run won); staging retained",
                code="TL1002",
                details={"path": dest}) from exc
        raise TLTypeError(f"cannot publish staged output to '{dest}': {exc}",
                          code="TL1003", details={"path": dest}) from exc
    if backup is not None:
        try:
            remove_path(backup)
        except Exception:  # backup removal is best-effort; data is safe
            pass
    describe.cache_clear()
    return dest
