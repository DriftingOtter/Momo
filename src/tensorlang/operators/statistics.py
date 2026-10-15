"""Reductions and statistics: sum, mean, min, max, std, var."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..backends import numpy_backend as _np_be
from ..backends import polars_backend as _pl_be
from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError


def apply(op: str, value, kind: str, by, evaluate, axis=None,
            keepdims: bool = False):
    """Evaluate a reduction. Returns ``(value, kind)``.

    - dense: NumPy reduction (``axis``/``keepdims`` per NumPy; ``by``
      rejected; scalar kind only for rank-0 results without keepdims).
    - series: scalar (``axis`` restricted to None/0/-1, ``keepdims``
      rejected).
    - frame: one-row frame over applicable numerical fields, or a grouped
      frame when ``by`` is given (``axis`` rejected as ambiguous).
    """
    if kind == "dense":
        if by is not None:
            if axis is not None:
                raise TLTypeError(
                    f"reduction '{op}' cannot combine grouped 'by' with "
                    f"'axis'; use one or the other",
                    code="TL0511",
                    details={"op": op, "axis": repr(axis)})
            raise TLTypeError(
                f"grouped reduction '{op}' is only defined for structured tensors"
            )
        arr = np.asarray(value)
        axes = ir.resolve_axes(axis, arr.ndim, op, arr.shape)
        out = _np_be.apply_reduce(op, arr, axes, keepdims)
        if np.ndim(out) == 0 and not keepdims:
            return out, "scalar"
        return out, "dense"
    if kind == "series":
        if by is None:
            if keepdims:
                raise TLTypeError(
                    f"reduction '{op}' keepdims has no series result form; "
                    f"collect() to dense first for shaped reductions",
                    code="TL0511",
                    details={"op": op, "keepdims": True})
            if axis is not None and tuple(axis) not in ((),):
                # Rank-1 validation only (axis 0/-1); anything valid here
                # reduces the whole series, matching NumPy semantics.
                ir.resolve_axes(axis, 1, op, (len(value),))
            return _series_reduce(op, value), "scalar"
        if axis is not None:
            raise TLTypeError(
                f"reduction '{op}' cannot combine grouped 'by' with 'axis'; "
                f"use one or the other",
                code="TL0511",
                details={"op": op, "axis": repr(axis)})
        key = _resolve_series_key(by, value, evaluate)
        return _pl_be.grouped_series(op, value, key), "frame"
    if kind == "frame":
        if axis is not None:
            raise TLTypeError(
                f"reduction '{op}' axis={list(axis)} is ambiguous for "
                f"structured frames (rows vs fields); reduce fields "
                f"explicitly or collect() to dense first",
                code="TL0511",
                details={"op": op, "axes": list(axis)})
        key = _resolve_by(by, value, evaluate)
        out = _pl_be.apply_reduce(op, value, key)
        if isinstance(out, pl.DataFrame):
            return out, "frame"
        return out, "scalar"
    if kind == "scalar":
        if by is not None:
            raise TLTypeError("cannot group a scalar reduction")
        if axis is not None:
            ir.resolve_axes(axis, 0, op, ())
        if op == "sum" and axis is None:
            return value, "scalar"
        if op == "count":
            arr = np.asarray(value)
            if arr.dtype.kind in "fc" and bool(np.isnan(arr)):
                return 0, "scalar"
            return 1, "scalar"
        return _np_be.apply_reduce(op, np.asarray(value), (), keepdims), \
            "scalar"
    raise TLTypeError(f"unsupported reduction operand")  # pragma: no cover


def _series_reduce(op: str, series: pl.Series):
    table = {
        "sum": series.sum, "mean": series.mean, "min": series.min,
        "max": series.max, "std": series.std, "var": series.var,
        "count": series.count,
    }
    try:
        return table[op]()
    except Exception as exc:
        raise TLTypeError(f"cannot compute '{op}' of series: {exc}") from exc


def _resolve_by(by, frame: pl.DataFrame, evaluate):
    if by is None:
        return None
    if isinstance(by, ir.Const) and isinstance(by.value, str):
        return by.value
    value, kind = evaluate(by)
    if kind == "series":
        return value
    if kind == "scalar":
        if isinstance(value, str):
            return value
        raise TLTypeError("group key must be a field name or one value per row")
    if kind == "dense":
        return value
    raise TLTypeError("group key must be a field name or one value per row")


# -- covariance, correlation, quantiles (v0.3) ------------------------------------


def cov(x, rowvar: bool | None = None):
    """Covariance matrix.

    ``rowvar`` follows NumPy for dense data (default True).  Structured
    frames always use columns-as-variables and return a labeled frame.
    """
    from ..core.tensor import Lazy, expr_of

    root = x.root if isinstance(x, Lazy) else expr_of(x, "cov")
    return Lazy(ir.Cov(root, True if rowvar is None else bool(rowvar)))


def corr(x, rowvar: bool | None = None):
    """Correlation matrix (same conventions as :func:`cov`)."""
    from ..core.tensor import Lazy, expr_of

    root = x.root if isinstance(x, Lazy) else expr_of(x, "corr")
    return Lazy(ir.Corr(root, True if rowvar is None else bool(rowvar)))


def quantile(x, q=0.5):
    """Quantiles at ``q`` (scalar or list of probabilities in [0, 1])."""
    from ..core.tensor import Lazy, expr_of

    probs = _norm_q(q)
    root = x.root if isinstance(x, Lazy) else expr_of(x, "quantile")
    return Lazy(ir.Quantile(root, probs))


def _norm_q(q) -> tuple[float, ...]:
    vals = list(q) if isinstance(q, (list, tuple)) else [q]
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise TLTypeError(f"quantile probabilities must be numbers, got {v!r}") from None
        if not 0.0 <= f <= 1.0:
            raise TLTypeError(f"quantile probabilities must lie in [0, 1], got {v!r}")
        out.append(f)
    if not out:
        raise TLTypeError("quantile needs at least one probability")
    return tuple(out)


def evaluate_cov(value, kind: str, rowvar: bool):
    matrix, names = _var_matrix(value, kind, "cov", rowvar)
    try:
        # Dense follows the rowvar flag; frames are columns-as-variables
        # by construction (records are rows).
        out = np.cov(matrix, rowvar=rowvar if names is None else False)
    except ValueError as exc:
        raise TLShapeError(f"cov failed: {exc}") from exc
    return _labeled(out, names, kind)


def evaluate_corr(value, kind: str, rowvar: bool):
    matrix, names = _var_matrix(value, kind, "corr", rowvar)
    with np.errstate(all="ignore"):
        out = np.corrcoef(matrix, rowvar=rowvar if names is None else False)
    out = np.asarray(out, dtype=np.float64)
    return _labeled(out, names, kind)


def _var_matrix(value, kind: str, name: str, rowvar: bool):
    if kind == "series":
        raise TLTypeError(
            f"{name} needs at least two variables; got a single field series")
    if kind == "dense":
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise TLShapeError(f"{name} needs a 2-D matrix, got shape {arr.shape}")
        return arr, None
    if kind == "frame":
        from ..backends import polars_backend as _pl_be

        numeric = _pl_be.numeric_columns(value)
        if len(numeric) < 2:
            if numeric and value.width > len(numeric):
                raise TLTypeError(
                    f"{name} needs an unambiguous numeric feature selection; "
                    f"non-numeric fields present — select explicitly, e.g. "
                    f"data[{numeric}]")
            raise TLShapeError(f"{name} needs at least two variables")
        return value.select(numeric).to_numpy().astype(np.float64), numeric
    raise TLTypeError(f"{name} needs matrix data")


def _labeled(out: np.ndarray, names, kind: str):
    out = np.atleast_2d(np.asarray(out, dtype=np.float64))
    if names is None:
        return out, "dense"
    cols = {"variable": names}
    cols.update({name: out[:, j].tolist() for j, name in enumerate(names)})
    return pl.DataFrame(cols), "frame"


def evaluate_quantile(value, kind: str, q: tuple[float, ...]):
    if kind == "series":
        arr = np.asarray(value.to_list(), dtype=np.float64)
        out = np.quantile(arr, list(q))
        return out.tolist()[0] if len(q) == 1 else out, \
            ("scalar" if len(q) == 1 else "dense")
    if kind == "dense":
        arr = np.asarray(value, dtype=np.float64)
        out = np.quantile(arr, list(q))
        if len(q) == 1:
            return float(out.flat[0]), "scalar"
        return np.asarray(out), "dense"
    if kind == "frame":
        from ..backends import polars_backend as _pl_be

        numeric = _pl_be.numeric_columns(value)
        if not numeric:
            raise TLTypeError("quantile has no applicable numerical fields")
        rows = []
        for prob in q:
            rows.append([float(np.quantile(
                np.asarray(value[c].to_numpy(), dtype=np.float64), prob))
                for c in numeric])
        frame = pl.DataFrame(
            {"quantile": list(q),
             **{c: [r[j] for r in rows] for j, c in enumerate(numeric)}})
        return frame, "frame"
    raise TLTypeError("quantile needs dense data or a structured selection")


def _resolve_series_key(by, series: pl.Series, evaluate):
    if isinstance(by, ir.Const) and isinstance(by.value, str):
        raise TLTypeError(
            f"unknown group field '{by.value}': operand is a single field series"
        )
    value, kind = evaluate(by)
    if kind == "series":
        return value
    if kind == "dense":
        return value
    raise TLTypeError("group key must be one value per row")
