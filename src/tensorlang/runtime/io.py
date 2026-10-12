"""Scan execution and dataset writing.

Scans are declarative descriptors until an explicit execution boundary
(``collect()`` / ``write_*`` / ``profile()``) materializes them.  Fully
lowered pipelines use Polars streaming ``sink_*`` and never materialize
intermediate frames in Python at all.
"""
from __future__ import annotations

import polars as pl

from ..core.errors import TLTypeError
from ..core.types import ScanSpec

# Counts effectful scan executions (drives profile() honestly).
SCAN_EXECUTIONS = 0


def _bump() -> None:
    global SCAN_EXECUTIONS
    SCAN_EXECUTIONS += 1


def lazy_scan(spec: ScanSpec) -> pl.LazyFrame:
    """Build (but do not execute) a Polars scan for ``spec``."""
    try:
        if spec.format == "parquet":
            return pl.scan_parquet(list(spec.paths), **spec.kwargs())
        if spec.format == "csv":
            return pl.scan_csv(list(spec.paths), **spec.kwargs())
    except TypeError as exc:
        raise TLTypeError(f"unsupported scan option for {spec.describe()}: {exc}")
    raise TLTypeError(f"unsupported scan format '{spec.format}'")  # pragma: no cover


def materialize_scan(spec: ScanSpec) -> pl.DataFrame:
    """Eagerly read a scan (explicit boundary only)."""
    _bump()
    try:
        return lazy_scan(spec).collect()
    except Exception as exc:
        raise TLTypeError(f"cannot scan {spec.describe()}: {exc}") from exc


def scan_schema(spec: ScanSpec) -> dict:
    """Schema metadata without reading row data."""
    try:
        return dict(lazy_scan(spec).collect_schema())
    except Exception as exc:
        raise TLTypeError(f"cannot describe {spec.describe()}: {exc}") from exc


def write_value(value, kind: str, path: str, format: str) -> str:
    """Write an already-materialized value to ``path``."""
    if kind == "dense":
        raise TLTypeError(
            f"cannot write dense arrays to {format}; select structured fields "
            f"first or use to_numpy() and NumPy I/O explicitly"
        )
    if kind == "series":
        frame = value.to_frame()
    elif kind == "frame":
        frame = value
    elif kind == "scalar":
        raise TLTypeError(f"cannot write a scalar value to {format}")
    else:  # pragma: no cover - defensive
        raise TLTypeError(f"cannot write '{kind}' data to {format}")
    try:
        if format == "parquet":
            frame.write_parquet(path)
        else:
            frame.write_csv(path)
    except Exception as exc:
        raise TLTypeError(f"cannot write {format} to '{path}': {exc}") from exc
    return path
