"""Tensor algebra: Einstein summation, tensor dots, axis movement."""
from __future__ import annotations

import numpy as np

from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError


def _operand(x, name: str) -> ir.Expr:
    from ..core.tensor import Lazy, Tensor, expr_of

    if isinstance(x, Lazy):
        return x.root
    if isinstance(x, Tensor):
        return x._snapshot_node()
    if isinstance(x, (np.ndarray, int, float, bool)):
        return expr_of(x, name)
    from ..core.operator import Operator

    if isinstance(x, Operator):
        return None  # handled by the n-ary merge below
    raise TLTypeError(
        f"{name} needs tensor data, got {type(x).__name__}"
    )


def _merge_parts(name: str, spec: str, parts: list):
    """Build an n-ary node, merging operator scopes by parameter name."""
    from ..core.operator import Operator, fresh_uid

    ops = [p for p in parts if isinstance(p, Operator)]
    if not ops:
        return None
    uid = fresh_uid()
    names: list[str] = []
    for op in ops:
        for p in op.params:
            if p not in names:
                names.append(p)

    def node(x):
        from ..core.tensor import Lazy, Tensor

        if isinstance(x, Operator):
            return ir.remap_params(x.root, uid)
        if isinstance(x, Tensor):
            return x._snapshot_node()
        if isinstance(x, Lazy):
            return x.root
        return ir.Const(x)

    return Operator(tuple(names), spec([node(p) for p in parts]), uid)


def einsum(spec: str, *operands):
    """Einstein summation (NumPy conventions, ``optimize=True``)."""
    if not isinstance(spec, str) or "->" not in spec.replace(" ", ""):
        raise TLTypeError(
            f"einsum needs a subscript specification like 'ij,jk->ik', "
            f"got {spec!r}"
        )
    if not operands:
        raise TLTypeError("einsum needs at least one operand")
    merged = _merge_parts("einsum", lambda ps: ir.Einsum(spec, tuple(ps)),
                          list(operands))
    if merged is not None:
        return merged
    from ..core.tensor import Lazy

    return Lazy(ir.Einsum(spec, tuple(_operand(o, "einsum") for o in operands)))


def tensordot(a, b, axes=2):
    """Tensor dot product along ``axes`` (NumPy conventions)."""
    merged = _merge_parts(
        "tensordot", lambda ps: ir.Tensordot(ps[0], ps[1], _norm_axes(axes)),
        [a, b])
    if merged is not None:
        return merged
    from ..core.tensor import Lazy

    return Lazy(ir.Tensordot(_operand(a, "tensordot"), _operand(b, "tensordot"),
                             _norm_axes(axes)))


def _norm_axes(axes):
    if isinstance(axes, int):
        if axes < 0:
            raise TLTypeError(f"invalid tensordot axes {axes!r}")
        return axes
    try:
        return tuple(tuple(x) if isinstance(x, list) else x for x in axes)
    except TypeError:
        raise TLTypeError(f"invalid tensordot axes {axes!r}") from None


def moveaxis(x, source, destination):
    """Move array axes (NumPy conventions)."""
    from ._lift import lift_unary_operand

    return lift_unary_operand(
        x,
        lambda root: ir.MoveAxis(root, _norm_ax(source, "source"),
                                 _norm_ax(destination, "destination")),
    )


def _norm_ax(ax, name: str) -> tuple[int, ...]:
    if isinstance(ax, (int, np.integer)):
        return (int(ax),)
    try:
        out = tuple(int(v) for v in ax)
    except TypeError:
        raise TLTypeError(f"invalid moveaxis {name} {ax!r}") from None
    return out


# -- evaluation -----------------------------------------------------------------


def _dense_only(value, kind: str, name: str) -> np.ndarray:
    if kind != "dense":
        raise TLTypeError(
            f"{name} is defined for dense tensors; select numeric fields "
            f"explicitly for structured data"
        )
    return np.asarray(value)


def evaluate_einsum(spec: str, parts: list):
    arrays = [_dense_only(v, k, "einsum") for v, k in parts]
    try:
        return np.einsum(spec, *arrays, optimize=True), "dense"
    except ValueError as exc:
        raise TLShapeError(f"einsum failed: {exc}") from exc


def evaluate_tensordot(a, akind: str, b, bkind: str, axes):
    ma = _dense_only(a, akind, "tensordot")
    mb = _dense_only(b, bkind, "tensordot")
    try:
        return np.tensordot(ma, mb, axes=axes), "dense"
    except ValueError as exc:
        raise TLShapeError(f"tensordot failed: {exc}") from exc


def evaluate_moveaxis(value, kind: str, source, destination):
    arr = _dense_only(value, kind, "moveaxis")
    try:
        return np.moveaxis(arr, source, destination), "dense"
    except (ValueError, TypeError) as exc:
        raise TLShapeError(f"moveaxis failed: {exc}") from exc
