"""Execution planner: backend selection and safe constant folding.

The initial planner is deliberately simple: it inspects the expression
graph, assigns each node the backend that will execute it (``numpy``,
``polars``, ``sympy`` or ``scalar``), and folds constant-only subtrees
where that is provably safe (pure scalar arithmetic over literals).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import expression as ir


@dataclass
class Plan:
    root: ir.Expr
    backends: dict[int, str] = field(default_factory=dict)
    node_count: int = 0

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for backend in self.backends.values():
            counts[backend] = counts.get(backend, 0) + 1
        return counts


def plan(root: ir.Expr) -> Plan:
    """Build an execution plan for a (possibly open) expression graph."""
    assignment: dict[int, str] = {}
    count = 0

    def visit(node: ir.Expr) -> str:
        nonlocal count
        count += 1
        if isinstance(node, ir.Literal):
            backend = {"dense": "numpy", "frame": "polars",
                       "series": "polars", "scalar": "scalar"}[node.kind]
        elif isinstance(node, ir.Param):
            backend = "symbolic"
        elif isinstance(node, ir.Const):
            backend = "scalar" if _is_scalar_const(node.value) else "numpy"
        elif isinstance(node, ir.ApplyOp):
            for a in node.args:
                visit(a)
            backend = "mixed"
        else:
            for child in _children(node):
                visit(child)
            backend = _backend_for(type(node).__name__)
        assignment[id(node)] = backend
        return backend

    visit(root)
    return Plan(root=fold_constants(root), backends=assignment, node_count=count)


def _children(node: ir.Expr) -> list[ir.Expr]:
    if isinstance(node, ir.BinOp):
        return [node.left, node.right]
    if isinstance(node, ir.UnOp):
        return [node.operand]
    if isinstance(node, ir.FuncCall):
        return list(node.args)
    if isinstance(node, ir.FillNull):
        return [node.operand, node.value]
    if isinstance(node, ir.FieldAccess):
        return [node.obj]
    if isinstance(node, ir.GetItem):
        return [node.obj, node.index]
    if isinstance(node, (ir.Transpose, ir.Reshape, ir.Flatten)):
        return [node.obj]
    if isinstance(node, (ir.Concat, ir.Stack)):
        return list(node.parts)
    if isinstance(node, ir.Reduce):
        return [node.operand] + ([node.by] if node.by is not None else [])
    if isinstance(node, ir.WithField):
        return [node.obj, node.value]
    if isinstance(node, ir.Join):
        return [node.left, node.right]
    if isinstance(node, (ir.SortBy, ir.Unique)):
        return [node.obj]
    if isinstance(node, ir.Where):
        return [node.cond, node.a, node.b]
    if isinstance(node, ir.Clip):
        return [node.obj, node.lo, node.hi]
    if isinstance(node, ir.Over):
        return [node.operand] + ([node.by] if node.by is not None else [])
    if isinstance(node, ir.Rolling):
        return [node.operand]
    if isinstance(node, ir.Scan):
        return []
    if isinstance(node, ir.Chunked):
        return []
    if isinstance(node, ir.FittedTransform):
        return [node.arg]
    if isinstance(node, (ir.SvdPart, ir.EighPart)):
        return []
    if isinstance(node, ir.Solve):
        return [node.a, node.b]
    if isinstance(node, (ir.Norm, ir.Cov, ir.Corr, ir.Quantile)):
        return [node.obj]
    if isinstance(node, ir.Einsum):
        return list(node.parts)
    if isinstance(node, ir.Tensordot):
        return [node.a, node.b]
    if isinstance(node, (ir.MoveAxis, ir.OneHot, ir.Polynomial)):
        return [node.obj]
    if isinstance(node, ir.DiffVar):
        return [node.value]
    if isinstance(node, (ir.AdjointOf, ir.ForwardOf)):
        return []
    return []


def _backend_for(node_type: str) -> str:
    if node_type in ("FieldAccess",):
        return "polars"
    if node_type in ("Transpose", "Reshape", "Flatten", "Concat", "Stack"):
        return "numpy"
    if node_type == "Reduce":
        return "mixed"
    return "mixed"


def _is_scalar_const(value) -> bool:
    return isinstance(value, (int, float, bool, str, complex, slice)) or (
        isinstance(value, np.generic)
    )


def fold_constants(node: ir.Expr) -> ir.Expr:
    """Fold pure-scalar constant subtrees (safe: no data, no side effects)."""
    import operator as _pyop

    table = {
        "add": _pyop.add, "sub": _pyop.sub, "mul": _pyop.mul,
        "div": _pyop.truediv, "pow": _pyop.pow, "mod": _pyop.mod,
        "eq": _pyop.eq, "ne": _pyop.ne, "gt": _pyop.gt, "lt": _pyop.lt,
        "ge": _pyop.ge, "le": _pyop.le, "and": _pyop.and_, "or": _pyop.or_,
    }

    def visit(n: ir.Expr) -> ir.Expr:
        if isinstance(n, ir.BinOp):
            left, right = visit(n.left), visit(n.right)
            if (
                isinstance(left, ir.Const)
                and isinstance(right, ir.Const)
                and isinstance(left.value, (int, float, bool))
                and isinstance(right.value, (int, float, bool))
                and n.op in table
                and not (n.op == "div" and right.value == 0)
            ):
                try:
                    return ir.Const(table[n.op](left.value, right.value))
                except Exception:
                    return ir.BinOp(n.op, left, right)
            return ir.BinOp(n.op, left, right)
        if isinstance(n, ir.UnOp):
            operand = visit(n.operand)
            if isinstance(operand, ir.Const) and isinstance(
                operand.value, (int, float, bool)
            ):
                if n.op == "neg":
                    return ir.Const(-operand.value)
                if n.op == "not":
                    return ir.Const(not operand.value)
            return ir.UnOp(n.op, operand)
        if isinstance(n, ir.FuncCall):
            return ir.FuncCall(n.func, tuple(visit(a) for a in n.args))
        if isinstance(n, ir.GetItem):
            return ir.GetItem(visit(n.obj), visit(n.index))
        if isinstance(n, ir.Reduce):
            return ir.Reduce(n.op, visit(n.operand),
                             visit(n.by) if n.by is not None else None,
                             n.axis, n.keepdims)
        return n

    return visit(node)
