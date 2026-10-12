"""Relational operators: joins, sorting, unique values.

These work through the same first-class operator model: applied to data
they build lazy graphs; applied to operators they build new operators.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError
from ..core.types import JOIN_HOW
from ._lift import lift_unary_operand


def _norm_on(on) -> tuple[str, ...]:
    if isinstance(on, str):
        return (on,)
    keys = tuple(on)
    if not keys or not all(isinstance(k, str) for k in keys):
        raise TLTypeError(f"join keys must be field names, got {on!r}")
    return keys


def join(left, right, on=None, how: str = "inner", suffix: str = "_right"):
    """Relational join on explicit keys (frames only).

    ``how`` is one of inner/left/right/full/outer/cross/semi/anti.
    ``cross`` needs no keys; every other mode requires ``on``.
    """
    from ..core.operator import Operator, fresh_uid
    from ..core.tensor import Lazy, Tensor, expr_of

    keys = _norm_on(on) if on is not None else ()
    if how not in JOIN_HOW:
        raise TLTypeError(f"unknown join type '{how}'; use one of {sorted(JOIN_HOW)}",
                          code="TL0430",
                          details={"got": how, "valid": sorted(JOIN_HOW)})
    if how != "cross" and not keys:
        raise TLTypeError(f"join '{how}' needs explicit keys via on=",
                          code="TL0430",
                          details={"how": how})
    if isinstance(left, Operator) or isinstance(right, Operator):
        ops = [x for x in (left, right) if isinstance(x, Operator)]
        names: list[str] = []
        for op in ops:
            for p in op.params:
                if p not in names:
                    names.append(p)
        uid = fresh_uid()

        def node(x: Any) -> ir.Expr:
            if isinstance(x, Operator):
                return ir.remap_params(x.root, uid)
            if isinstance(x, Tensor):
                return x._snapshot_node()
            if isinstance(x, Lazy):
                return x.root
            raise TLTypeError(
                f"cannot join {type(x).__name__}; join tensors, lazy results "
                f"or operators sharing params {names}"
            )

        return Operator(tuple(names),
                        ir.Join(node(left), node(right), keys, how, suffix), uid)
    return Lazy(
        ir.Join(expr_of(left, "join"), expr_of(right, "join"), keys, how, suffix)
    )


def sort(x, by=None, descending: bool = False):
    """Explicit ordering. Frame sorts need key field(s); dense sorts values.

    Lazy/streaming execution preserves no incidental order, so ordering is
    only guaranteed where ``sort`` is applied explicitly.
    """
    from ..core.operator import Operator

    if by is not None and isinstance(by, Operator):
        raise TLTypeError("sort keys must be field names, not operators")
    keys = _norm_sort_keys(by)
    return lift_unary_operand(
        x, lambda root: ir.SortBy(root, keys, bool(descending))
    )


def _norm_sort_keys(by) -> tuple[str, ...]:
    if by is None:
        return ()
    if isinstance(by, str):
        return (by,)
    keys = tuple(by)
    if not all(isinstance(k, str) for k in keys):
        raise TLTypeError(f"sort keys must be field names, got {by!r}")
    return keys


def unique(x):
    """Unique values (dense/series, sorted) or unique rows (frames, ordered)."""
    return lift_unary_operand(x, lambda root: ir.Unique(root))


# -- fallback evaluation (used when Polars-lazy lowering does not apply) -------


def evaluate_join(left, lkind: str, right, rkind: str,
                  on: tuple[str, ...], how: str, suffix: str):
    if lkind != "frame" or rkind != "frame":
        raise TLTypeError(
            f"join needs two structured tensors, got {lkind} and {rkind}"
        )
    for key in on:
        if key not in left.columns or key not in right.columns:
            raise TLTypeError(
                f"join key '{key}' must exist on both sides; have "
                f"{left.columns} and {right.columns}",
                code="TL0431",
                details={"key": key, "left": list(left.columns),
                         "right": list(right.columns)},
            )
    try:
        if how == "cross":
            out = left.join(right, how="cross", suffix=suffix)
        else:
            out = left.join(right, on=list(on), how=how, suffix=suffix)
    except Exception as exc:
        raise TLShapeError(f"join failed: {exc}",
                           code="TL0510",
                           details={"on": list(on), "how": how}) from exc
    return out, "frame"


def evaluate_sort(value, kind: str, keys: tuple[str, ...], descending: bool):
    if kind == "dense":
        if keys:
            raise TLTypeError(
                f"cannot sort dense data by fields {list(keys)}; named fields "
                f"exist only on structured tensors"
            )
        arr = np.asarray(value)
        out = np.sort(arr, axis=-1)
        if descending:
            out = np.flip(out, axis=-1)
        return np.ascontiguousarray(out), "dense"
    if kind == "series":
        if keys:
            raise TLTypeError("cannot sort a series by fields")
        return value.sort(descending=descending), "series"
    if kind == "frame":
        if not keys:
            raise TLTypeError(
                "sorting a structured tensor needs explicit key fields via by="
            )
        missing = [k for k in keys if k not in value.columns]
        if missing:
            raise TLTypeError(
                f"unknown sort keys {missing}; have {value.columns}"
            )
        return value.sort(list(keys), descending=descending), "frame"
    raise TLTypeError("cannot sort a scalar value")


def evaluate_unique(value, kind: str):
    if kind == "dense":
        return np.unique(np.asarray(value)), "dense"
    if kind == "series":
        return value.unique(maintain_order=True), "series"
    if kind == "frame":
        return value.unique(maintain_order=True), "frame"
    raise TLTypeError("cannot take unique values of a scalar")
