"""Row-wise transformations: conditional replacement, clipping, grouped
broadcasts (``over``), and sliding-window (``rolling``) aggregation."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError


def _operand_names(ops: list) -> list[str]:
    names: list[str] = []
    for op in ops:
        for p in op.params:
            if p not in names:
                names.append(p)
    return names


def where(cond, a, b):
    """Elementwise ``where(cond, a, b)``: values from ``a`` where ``cond``
    holds, else from ``b``.  Works on dense data, field series, and —
    with a Boolean frame condition — whole frames."""
    from ..core.operator import Operator, fresh_uid
    from ..core.tensor import Lazy, Tensor, expr_of

    ops = [x for x in (cond, a, b) if isinstance(x, Operator)]
    if ops:
        uid = fresh_uid()
        params = tuple(_operand_names(ops))

        def node(x: Any) -> ir.Expr:
            if isinstance(x, Operator):
                missing = [p for p in x.params if p not in params]
                if missing:  # pragma: no cover - defensive (union covers)
                    raise TLTypeError(f"unbound params {missing} in where()")
                return ir.remap_params(x.root, uid)
            if isinstance(x, Tensor):
                return x._snapshot_node()
            if isinstance(x, Lazy):
                return x.root
            return ir.Const(x)

        return Operator(params,
                        ir.Where(node(cond), node(a), node(b)), uid)
    return Lazy(ir.Where(expr_of(cond, "where"), expr_of(a, "where"),
                        expr_of(b, "where")))


def clip(x, lo=None, hi=None):
    """Clamp values to ``[lo, hi]``; either bound may be None (one-sided)."""
    from ._lift import lift_unary_operand
    from ..core.tensor import expr_of

    if lo is None and hi is None:
        raise TLTypeError("clip needs at least one bound")
    return lift_unary_operand(
        x,
        lambda root: ir.Clip(
            root,
            expr_of(lo, "clip") if lo is not None else ir.Const(None),
            expr_of(hi, "clip") if hi is not None else ir.Const(None),
        ),
    )


def over(x, by, op: str = "mean"):
    """Grouped transformation: per-group ``op`` broadcast back to rows.

    ``by`` is a field name (inside lazy structured pipelines), a field
    series, or a per-row key array.  Unlike ``Reduce(..., by=)`` this keeps
    row shape — e.g. demeaned values per group.
    """
    from ..core.operator import Operator

    if op not in ir.REDUCTIONS:
        raise TLTypeError(f"unknown grouped operator '{op}'")
    if isinstance(by, Operator):
        raise TLTypeError("group keys must be field names or key data, not operators")
    return _over_lift(x, by, op)


def _over_lift(x, by, op: str):
    from ..core.operator import Operator, fresh_uid
    from ..core.tensor import Lazy, Tensor, expr_of

    if isinstance(x, Operator):
        uid = fresh_uid()
        if isinstance(by, str):
            by_node: ir.Expr = ir.Const(by)
        elif isinstance(by, (Tensor, Lazy)):
            by_node = by._snapshot_node() if isinstance(by, Tensor) else by.root
        else:
            by_node = ir.Const(by)
        return Operator(x.params,
                        ir.Over(op, ir.remap_params(x.root, uid), by_node), uid)
    if isinstance(by, str):
        by_node = ir.Const(by)
    else:
        by_node = expr_of(by, "group-by")
    root = x._snapshot_node() if isinstance(x, Tensor) else (
        x.root if isinstance(x, Lazy) else expr_of(x, "over"))
    return Lazy(ir.Over(op, root, by_node))


def rolling(x, window: int, op: str = "mean", min_periods: int | None = None):
    """Sliding-window aggregation over 1-D dense data or a field series."""
    from ._lift import lift_unary_operand

    if window < 1:
        raise TLTypeError("rolling window must be >= 1")
    if op not in ("sum", "mean", "min", "max"):
        raise TLTypeError(f"unknown rolling operator '{op}'")
    if min_periods is not None and min_periods < 1:
        raise TLTypeError("rolling min_periods must be >= 1")
    return lift_unary_operand(
        x, lambda root: ir.Rolling(root, int(window), op, min_periods)
    )


# -- fallback evaluation -------------------------------------------------------


def evaluate_where(cond, ckind: str, a, akind: str, b, bkind: str):
    if ckind == "frame":
        if akind == "frame" and bkind == "frame":
            return _frame_where(cond, a, b)
        if akind == "scalar" and bkind == "scalar":
            return _frame_where_scalar(cond, a, b)
        raise TLTypeError(
            "frame conditions need scalar or same-schema frame branches; "
            "select fields explicitly otherwise"
        )
    c = _bool_array(cond, ckind)
    av = np.asarray(a) if akind != "scalar" else a
    bv = np.asarray(b) if bkind != "scalar" else b
    try:
        out = np.where(c, av, bv)
    except ValueError as exc:
        raise TLShapeError(f"where() branches do not broadcast: {exc}") from exc
    if akind == "series" or bkind == "series":
        name = (a.name if akind == "series" else
                (b.name if bkind == "series" else "where"))
        return pl.Series(str(name), out.tolist()), "series"
    return out, "dense"


def _bool_array(cond, ckind: str) -> np.ndarray:
    if ckind == "series":
        if cond.dtype != pl.Boolean:
            raise TLTypeError("where() needs a Boolean condition series")
        return np.asarray(cond.to_list(), dtype=bool)
    arr = np.asarray(cond)
    if arr.dtype != bool:
        raise TLTypeError("where() needs a Boolean condition")
    return arr


def _frame_where(cond: pl.DataFrame, a: pl.DataFrame, b: pl.DataFrame):
    if cond.columns != a.columns or a.columns != b.columns:
        raise TLShapeError(
            "frame where() needs matching columns for condition and branches"
        )
    cols = []
    for c in a.columns:
        if c not in _pl_bool_cols(cond):
            raise TLTypeError(
                f"frame where() needs Boolean condition fields; '{c}' is not Boolean"
            )
        cols.append(
            pl.Series(c, np.where(np.asarray(cond[c].to_list(), dtype=bool),
                                  np.asarray(a[c].to_list()),
                                  np.asarray(b[c].to_list())).tolist())
        )
    return pl.DataFrame(cols), "frame"


def _pl_bool_cols(df: pl.DataFrame) -> list[str]:
    return [n for n, d in df.schema.items() if d == pl.Boolean]


def _frame_where_scalar(cond: pl.DataFrame, a, b):
    cols = []
    for c in cond.columns:
        if c not in _pl_bool_cols(cond):
            raise TLTypeError(
                f"frame where() needs Boolean condition fields; '{c}' is not Boolean"
            )
        cols.append(pl.Series(c, np.where(np.asarray(cond[c].to_list(), dtype=bool),
                                          a, b).tolist()))
    return pl.DataFrame(cols), "frame"


def evaluate_clip(value, kind: str, lo, hi):
    lo_v = None if lo is None else (lo.item() if isinstance(lo, np.generic) else lo)
    hi_v = None if hi is None else (hi.item() if isinstance(hi, np.generic) else hi)
    if kind == "dense":
        try:
            return np.clip(np.asarray(value), lo_v, hi_v), "dense"
        except (TypeError, ValueError) as exc:
            raise TLTypeError(f"cannot clip: {exc}") from exc
    if kind == "series":
        try:
            return value.clip(lo_v, hi_v), "series"
        except Exception as exc:
            raise TLTypeError(f"cannot clip series: {exc}") from exc
    if kind == "frame":
        from ..backends import polars_backend as _pl_be

        cols = _pl_be.numeric_columns(value)
        if not cols:
            raise TLTypeError("clip has no applicable numerical fields")
        out = value.clone()
        for c in cols:
            out = out.with_columns(value[c].clip(lo_v, hi_v).alias(c))
        return out, "frame"
    raise TLTypeError("cannot clip a scalar value")


def evaluate_over(node: ir.Over, evaluate):
    """Grouped broadcast. ``by`` may be a field name when the operand node
    is a field projection (resolved against its frame), else key data."""
    operand_value, operand_kind = evaluate(node.operand)
    by_value, by_kind = (None, None) if node.by is None else evaluate(node.by)
    if node.by is None:
        raise TLTypeError("over() needs an explicit group key via by=")
    if by_kind == "scalar" and isinstance(by_value, str):
        if not (isinstance(node.operand, ir.FieldAccess)):
            raise TLTypeError(
                f"group field '{by_value}' needs a structured operand; "
                f"use a key series or array otherwise"
            )
        frame, _ = evaluate(node.operand.obj)
        if not isinstance(frame, pl.DataFrame):
            raise TLTypeError("over() field keys need a structured operand")
        return _series_over_polars(frame[node.operand.field], node.op,
                                   frame[by_value], by_value)
    key = by_value if by_kind == "series" else np.asarray(by_value)
    if operand_kind == "series":
        return _series_over_polars(operand_value, node.op, key, None)
    if operand_kind == "dense":
        return _dense_over(np.asarray(operand_value), node.op, key)
    raise TLTypeError("over() needs a field series or dense data")


def _key_series(key, n: int, name: str | None) -> pl.Series:
    if isinstance(key, pl.Series):
        if len(key) != n:
            raise TLShapeError(
                f"group key has length {len(key)} but data has {n} rows"
            )
        return key
    arr = np.asarray(key)
    if arr.ndim != 1 or len(arr) != n:
        raise TLShapeError("group key must be one value per row")
    return pl.Series(name or "group", arr.tolist())


def _series_over_polars(series: pl.Series, op: str, key, key_name=None):
    k = _key_series(key, len(series), key_name or (key.name if isinstance(key, pl.Series) else None))
    tmp = pl.DataFrame({"__tl_k__": k, "__tl_v__": series})
    out = tmp.with_columns(
        getattr(pl.col("__tl_v__"), op)().over("__tl_k__").alias("__tl_v__")
    )["__tl_v__"]
    out = out.rename(series.name or "over")
    return out, "series"


def _dense_over(values: np.ndarray, op: str, key) -> tuple[np.ndarray, str]:
    if values.ndim != 1:
        raise TLTypeError("dense over() is defined for 1-D data only")
    arr = np.asarray(key)
    if arr.ndim != 1 or len(arr) != len(values):
        raise TLShapeError("group key must be one value per row")
    _, inverse = np.unique(arr, return_inverse=True)
    vals = values.astype(float)
    if op == "sum":
        agg = np.bincount(inverse, weights=vals)
    elif op == "mean":
        agg = np.bincount(inverse, weights=vals) / np.bincount(inverse)
    elif op == "min":
        agg = np.full(inverse.max() + 1, np.inf)
        np.minimum.at(agg, inverse, vals)
    elif op == "max":
        agg = np.full(inverse.max() + 1, -np.inf)
        np.maximum.at(agg, inverse, vals)
    elif op == "var":
        m = np.bincount(inverse, weights=vals) / np.bincount(inverse)
        m2 = np.bincount(inverse, weights=vals**2) / np.bincount(inverse)
        agg = m2 - m**2
    elif op == "std":
        m = np.bincount(inverse, weights=vals) / np.bincount(inverse)
        m2 = np.bincount(inverse, weights=vals**2) / np.bincount(inverse)
        agg = np.sqrt(np.maximum(m2 - m**2, 0.0))
    else:  # pragma: no cover - guarded at construction
        raise TLTypeError(f"unknown grouped operator '{op}'")
    return agg[inverse], "dense"


def evaluate_rolling(value, kind: str, window: int, op: str, min_periods):
    mp = window if min_periods is None else min_periods
    if kind == "series":
        fn = {"sum": "rolling_sum", "mean": "rolling_mean",
              "min": "rolling_min", "max": "rolling_max"}[op]
        try:
            return getattr(value, fn)(window_size=window, min_samples=mp), "series"
        except Exception as exc:
            raise TLTypeError(f"cannot roll series: {exc}") from exc
    if kind == "dense":
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise TLTypeError("dense rolling() is defined for 1-D data only")
        return _dense_rolling(arr.astype(float), window, op, mp), "dense"
    raise TLTypeError("rolling() needs 1-D dense data or a field series")


def _dense_rolling(vals: np.ndarray, window: int, op: str, mp: int):
    n = len(vals)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    table = {"sum": np.sum, "mean": np.mean, "min": np.min, "max": np.max}
    fn = table[op]
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        if n >= window:
            out[window - 1:] = fn(sliding_window_view(vals, window), axis=-1)
    except ValueError:  # pragma: no cover - defensive
        pass
    for i in range(min(mp - 1, n), min(window - 1, n)):
        out[i] = fn(vals[: i + 1])
    return out
