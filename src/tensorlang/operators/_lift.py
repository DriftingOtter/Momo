"""Shared helpers for building operators and lazy graphs from functions."""
from __future__ import annotations

from typing import Any

from ..core import expression as ir
from ..core.operator import Operator, fresh_uid


def _merge_names(ops: list[Operator]) -> list[str]:
    names: list[str] = []
    for op in ops:
        for p in op.params:
            if p not in names:
                names.append(p)
    return names


def lift_unary_operand(value: Any, build, const_build=None):
    """Apply a single-operand ``build(root) -> Expr`` to Operator/Tensor/Lazy.

    Returns an ``Operator`` for operator input, a ``Lazy`` otherwise.
    """
    from ..core.tensor import Lazy, Tensor, expr_of

    if isinstance(value, Operator):
        uid = fresh_uid()
        return Operator(value.params, build(ir.remap_params(value.root, uid)), uid)
    if isinstance(value, (Tensor, Lazy)):
        root = value._snapshot_node() if isinstance(value, Tensor) else value.root
        return Lazy(build(root))
    if const_build is not None:
        return const_build(value)
    return Lazy(build(expr_of(value, "function")))


def lift_binary(a: Any, b: Any, opcode: str):
    """Combine two operands with a binary opcode (operator-aware merge)."""
    from ..core.tensor import Lazy, Tensor, expr_of

    ops = [x for x in (a, b) if isinstance(x, Operator)]
    if ops:
        uid = fresh_uid()
        params = tuple(_merge_names(ops))

        def node(x: Any) -> ir.Expr:
            if isinstance(x, Operator):
                return ir.remap_params(x.root, uid)
            if isinstance(x, Tensor):
                return x._snapshot_node()
            if isinstance(x, Lazy):
                return x.root
            return ir.Const(x)

        return Operator(params, ir.BinOp(opcode, node(a), node(b)), uid)
    return Lazy(ir.BinOp(opcode, expr_of(a, opcode), expr_of(b, opcode)))
