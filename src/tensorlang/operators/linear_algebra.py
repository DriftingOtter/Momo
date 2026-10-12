"""Tensor manipulation: transpose, reshape, flatten, concat, stack."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..backends import numpy_backend as _np_be
from ..backends import polars_backend as _pl_be
from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError, TLUnsupportedError
from ._lift import lift_unary_operand


def transpose(x):
    """Transpose (dense only; undefined for structured tensors)."""
    return lift_unary_operand(x, lambda root: ir.Transpose(root))


def reshape(x, *shape):
    """Reshape (dense only)."""
    return lift_unary_operand(x, lambda root: ir.Reshape(root, _norm(shape)))


def flatten(x):
    """Flatten to one dimension (dense and series)."""
    return lift_unary_operand(x, lambda root: ir.Flatten(root))


def concat(parts, axis: int = 0):
    """Concatenate tensors along ``axis`` (dense) or vertically (frames)."""
    from ..core.tensor import Lazy, Tensor, expr_of

    seq = list(parts)
    if any(isinstance(p, _OperatorLike()) for p in seq):
        return _operator_concat(seq, axis)
    return Lazy(ir.Concat(tuple(expr_of(p, "concat") for p in seq), axis))


def stack(parts, axis: int = 0):
    """Stack tensors along a new ``axis`` (dense and series)."""
    from ..core.tensor import Lazy, Tensor, expr_of

    seq = list(parts)
    if any(isinstance(p, _OperatorLike()) for p in seq):
        return _operator_stack(seq, axis)
    return Lazy(ir.Stack(tuple(expr_of(p, "concat") for p in seq), axis))


def _OperatorLike():
    from ..core.operator import Operator

    return Operator


def _operator_concat(seq, axis: int):
    from ..core.operator import Operator, fresh_uid

    ops = [p for p in seq if isinstance(p, _OperatorLike())]
    others = [p for p in seq if not isinstance(p, _OperatorLike())]
    if others:
        raise TLTypeError("cannot mix operators and data in concat()")
    names: list[str] = []
    for op in ops:
        for p in op.params:
            if p not in names:
                names.append(p)
    uid = fresh_uid()
    parts = tuple(ir.remap_params(op.root, uid) for op in ops)
    first = ops[0].params
    if any(tuple(op.params) != tuple(first) for op in ops):
        raise TLTypeError("concat() needs operators with identical parameters")
    return Operator(tuple(first), ir.Concat(parts, axis), uid)


def _operator_stack(seq, axis: int):
    from ..core.operator import Operator, fresh_uid

    ops = [p for p in seq if isinstance(p, _OperatorLike())]
    if any(not isinstance(p, _OperatorLike()) for p in seq):
        raise TLTypeError("cannot mix operators and data in stack()")
    names: list[str] = []
    for op in ops:
        for p in op.params:
            if p not in names:
                names.append(p)
    uid = fresh_uid()
    parts = tuple(ir.remap_params(op.root, uid) for op in ops)
    first = ops[0].params
    if any(tuple(op.params) != tuple(first) for op in ops):
        raise TLTypeError("stack() needs operators with identical parameters")
    return Operator(tuple(first), ir.Stack(parts, axis), uid)


def _norm(shape) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    try:
        return tuple(int(s) for s in shape)
    except (TypeError, ValueError) as exc:
        raise TLShapeError(f"invalid reshape shape {shape!r}") from exc


# -- evaluation --------------------------------------------------------------


def apply_transpose(value, kind: str):
    if kind == "dense":
        return _np_be.transpose(np.asarray(value)), "dense"
    if kind == "series":
        return value, "series"
    raise TLUnsupportedError(
        "transpose is undefined for heterogeneous structured tensors: "
        "select numerical fields explicitly"
    )


def apply_reshape(value, kind: str, shape: tuple[int, ...]):
    if kind == "dense":
        return _np_be.reshape(np.asarray(value), shape), "dense"
    raise TLUnsupportedError("reshape is undefined for structured tensors")


def apply_flatten(value, kind: str):
    if kind == "dense":
        return np.asarray(value).flatten(), "dense"
    if kind == "series":
        return value, "series"
    raise TLUnsupportedError("flatten is undefined for structured tensors")


def apply_concat(parts: list[tuple[Any, str]], axis: int):
    kinds = {k for _, k in parts}
    if kinds == {"dense"}:
        return _np_be.concat([np.asarray(v) for v, _ in parts], axis), "dense"
    if kinds == {"series"}:
        try:
            return pl.concat([v for v, _ in parts]), "series"
        except Exception as exc:
            raise TLShapeError(f"cannot concatenate series: {exc}") from exc
    if kinds == {"frame"}:
        return _pl_be.concat_frames([v for v, _ in parts]), "frame"
    if "scalar" in kinds and len(parts) > 1:
        return _np_be.concat(
            [np.asarray(v) for v, _ in parts], axis
        ), "dense"
    raise TLTypeError(
        f"cannot concatenate mixed representations {sorted(kinds)}: "
        f"convert explicitly first"
    )


def apply_stack(parts: list[tuple[Any, str]], axis: int):
    kinds = {k for _, k in parts}
    if kinds <= {"dense", "scalar"}:
        return _np_be.stack([np.asarray(v) for v, _ in parts], axis), "dense"
    if kinds == {"series"}:
        try:
            import numpy as _np

            return _np.stack([v.to_numpy() for v, _ in parts], axis), "dense"
        except Exception as exc:
            raise TLShapeError(f"cannot stack series: {exc}") from exc
    raise TLTypeError(
        f"stack is undefined for representations {sorted(kinds)}: "
        f"stack dense tensors or series only"
    )
