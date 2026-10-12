"""Basic data science: missing-value handling and grouped aggregation."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..backends import numpy_backend as _np_be
from ..backends import polars_backend as _pl_be
from ..core import expression as ir
from ..core.errors import TLTypeError
from ._lift import lift_unary_operand


def is_null(x):
    """Boolean mask of missing values (dense / series / frame)."""
    return lift_unary_operand(x, lambda root: ir.FuncCall("is_null", (root,)))


def not_null(x):
    """Boolean mask of present values."""
    return lift_unary_operand(x, lambda root: ir.FuncCall("not_null", (root,)))


def fill_null(x, value):
    """Replace missing values with ``value`` (explicit null handling)."""
    from ..core.tensor import expr_of

    return lift_unary_operand(
        x, lambda root: ir.FillNull(root, expr_of(value, "fill_null"))
    )


def drop_null(x):
    """Drop missing values (dense 1-D / series / frame rows)."""
    return lift_unary_operand(x, lambda root: ir.FuncCall("drop_null", (root,)))


def apply(func: str, value, kind: str):
    if kind == "dense":
        arr = np.asarray(value)
        if func == "is_null":
            return _np_be.is_null(arr), "dense"
        if func == "not_null":
            return ~_np_be.is_null(arr), "dense"
        if func == "drop_null":
            return _dense_drop_null(arr)
        raise TLTypeError(f"unknown function '{func}'")  # pragma: no cover
    if kind in ("series", "frame"):
        table = {
            "is_null": _pl_be.is_null, "not_null": _pl_be.not_null,
            "drop_null": _pl_be.drop_null,
        }
        out = table[func](value)
        if isinstance(out, pl.DataFrame):
            return out, "frame"
        return out, "series"
    raise TLTypeError(f"'{func}' is undefined for scalar values")


def apply_fill(value, kind: str, fill):
    if kind == "dense":
        arr = np.array(np.asarray(value), copy=True)
        mask = _np_be.is_null(arr)
        try:
            arr[mask] = fill
        except (TypeError, ValueError) as exc:
            raise TLTypeError(f"cannot fill_null with {fill!r}: {exc}") from exc
        return arr, "dense"
    if kind in ("series", "frame"):
        return _pl_be.fill_null(value, fill), kind
    raise TLTypeError("'fill_null' is undefined for scalar values")


def _dense_drop_null(arr: np.ndarray):
    mask = _np_be.is_null(arr)
    if arr.ndim == 1:
        return arr[~mask], "dense"
    if arr.ndim == 2:
        return arr[~mask.any(axis=1)], "dense"
    raise TLTypeError("drop_null is defined for 1-D/2-D dense tensors only")
