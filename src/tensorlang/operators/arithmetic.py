"""Arithmetic operators: evaluation dispatch across representations."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..backends import numpy_backend as _np_be
from ..backends import polars_backend as _pl_be
from ..core.errors import TLTypeError

_DENSE = (np.ndarray,)
_COMPARISONS = {"eq", "ne", "gt", "lt", "ge", "le"}
_LOGICALS = {"and", "or"}
def apply(op: str, left, right):
    """Evaluate a binary op over materialized values.

    Returns ``(value, kind)`` where kind is one of
    ``dense | frame | series | scalar``.
    """
    lkind = _kind_of(left)
    rkind = _kind_of(right)

    if lkind == "scalar" and rkind == "scalar":
        return _scalar_binop(op, left, right), "scalar"

    if op == "matmul":
        if lkind != "dense" or rkind != "dense":
            raise TLTypeError(
                "matrix multiplication requires compatible numerical operands; "
                f"got {lkind} and {rkind}"
            )
        return _np_be.matmul(np.asarray(left), np.asarray(right)), "dense"

    if lkind == "frame" or rkind == "frame":
        if (lkind == "dense" and rkind == "frame") or (
            lkind == "frame" and rkind == "dense"
        ):
            raise TLTypeError(
                f"cannot combine a dense array with a whole structured tensor "
                f"for '{op}': select numerical fields explicitly"
            )
        return _frame_result(_pl_be.apply_binop(op, left, right))

    if lkind == "series" or rkind == "series":
        if (lkind == "dense" and rkind == "series") or (
            lkind == "series" and rkind == "dense"
        ):
            left, right = _align_series_operand(left, lkind, right, rkind)
        out = _pl_be.apply_binop(op, left, right)
        if isinstance(out, pl.DataFrame):
            return out, "frame"
        if isinstance(out, pl.Series):
            return out, "series"
        return _np_be.apply_binop(op, np.asarray(left), np.asarray(right)), "dense"

    # dense / scalar mixing with NumPy-compatible broadcasting.
    try:
        return _np_be.apply_binop(op, np.asarray(left), np.asarray(right)), "dense"
    except TypeError as exc:
        raise TLTypeError(f"cannot apply '{op}': {exc}") from exc


def _frame_result(out):
    import polars as _pl

    if isinstance(out, _pl.DataFrame):
        return out, "frame"
    if isinstance(out, _pl.Series):
        return out, "series"
    return out, "dense"


def _align_series_operand(left, lkind, right, rkind):
    import polars as _pl

    if lkind == "dense":
        arr = np.asarray(left)
        if arr.ndim != 1 or len(arr) != len(right):
            raise TLTypeError(
                f"cannot combine a dense array of shape {arr.shape} with a "
                f"series of length {len(right)} for elementwise ops"
            )
        return _pl.Series(arr.tolist()), right
    arr = np.asarray(right)
    if arr.ndim != 1 or len(arr) != len(left):
        raise TLTypeError(
            f"cannot combine a series of length {len(left)} with a dense "
            f"array of shape {arr.shape} for elementwise ops"
        )
    return left, _pl.Series(arr.tolist())


def _scalar_binop(op: str, left, right):
    import operator as _pyop

    table = {
        "add": _pyop.add, "sub": _pyop.sub, "mul": _pyop.mul,
        "div": _pyop.truediv, "pow": _pyop.pow, "mod": _pyop.mod,
        "eq": _pyop.eq, "ne": _pyop.ne, "gt": _pyop.gt, "lt": _pyop.lt,
        "ge": _pyop.ge, "le": _pyop.le, "and": _pyop.and_, "or": _pyop.or_,
    }
    if op == "matmul":
        raise TLTypeError("matrix multiplication needs array operands")
    try:
        result = table[op](left, right)
    except ZeroDivisionError:
        raise
    except Exception as exc:
        raise TLTypeError(f"cannot apply '{op}': {exc}") from exc
    if isinstance(result, (np.bool_, bool)):
        return bool(result)
    return result


def apply_unary(op: str, operand):
    kind = _kind_of(operand)
    if kind == "scalar":
        table = {"neg": lambda x: -x, "not": lambda x: not x, "abs": abs}
        return table[op](operand), "scalar"
    if kind == "dense":
        return _np_be.apply_unop(op, np.asarray(operand)), "dense"
    if kind in ("frame", "series"):
        out = _pl_be.apply_unop(op, operand)
        return out, ("frame" if isinstance(out, pl.DataFrame) else "series")
    raise TLTypeError(f"unsupported operand for '{op}'")  # pragma: no cover


def _kind_of(value) -> str:
    if isinstance(value, np.ndarray):
        return "dense"
    if isinstance(value, pl.DataFrame):
        return "frame"
    if isinstance(value, pl.Series):
        return "series"
    return "scalar"
