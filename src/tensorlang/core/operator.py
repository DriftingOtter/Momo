"""First-class operators: symbolic parameters, application, composition.

An ``Operator`` pairs a tuple of lexically scoped parameter names with an
immutable expression tree.  Every combinator mints a fresh owner uid and
rebinds parameters by name, so identically named parameters from different
definitions (``f := {X}**2`` vs ``g := {X} + 1``) never collide, while
repeated references within one definition (``({X} > 10) & ({X} < 100)``)
unify to the same parameter.
"""
from __future__ import annotations

import itertools

import numpy as np
from typing import Any

from . import expression as ir
from .errors import TLShapeError, TLTypeError
from .tensor import Lazy, Tensor, expr_of

_uid_counter = itertools.count(1)


def fresh_uid() -> str:
    return f"op{next(_uid_counter)}"


def P(name: str) -> "Operator":
    """Create a symbolic parameter (plain-Python spelling of ``{X}``)."""
    _check_param_name(name)
    uid = fresh_uid()
    return Operator((name,), ir.Param(uid, name), uid)


def _check_param_name(name: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise TLTypeError(f"invalid parameter name {name!r}")


def _collect_names(operands: list["Operator"]) -> list[str]:
    names: list[str] = []
    for op in operands:
        for p in op.params:
            if p not in names:
                names.append(p)
    return names


def _combine(opcode: str, left: Any, right: Any) -> "Operator":
    ops = [x for x in (left, right) if isinstance(x, Operator)]
    uid = fresh_uid()
    params = tuple(_collect_names(ops))

    def node(x: Any) -> ir.Expr:
        if isinstance(x, Operator):
            # Merge by parameter name: repeated references to the same name
            # within one definition unify; names from different definitions
            # are rebound to this definition's fresh owner uid.
            return ir.remap_params(x.root, uid)
        value = _const_val(x, opcode)
        return value if isinstance(value, ir.Expr) else ir.Const(value)

    return Operator(params, ir.BinOp(opcode, node(left), node(right)), uid)


def _const_val(value: Any, opcode: str) -> Any:
    if isinstance(value, Tensor):
        return value._snapshot_node()  # type: ignore[return-value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, bool, str, complex)):
        return value
    if "sympy" in type(value).__module__:
        return value
    if isinstance(value, Lazy):
        raise TLTypeError(
            f"cannot use a pending computation inside an operator definition "
            f"('{opcode}'); apply operators to data, not the reverse"
        )
    raise TLTypeError(
        f"unsupported operand of type '{type(value).__name__}' for '{opcode}'"
    )


class Operator:
    """A reusable computational transformation over symbolic parameters."""

    def __init__(self, params: tuple[str, ...], root: ir.Expr, uid: str) -> None:
        for p in params:
            _check_param_name(p)
        self.params = tuple(params)
        self.root = root
        self.uid = uid
        self._bound_name: str | None = None

    # -- introspection ------------------------------------------------------------
    @property
    def arity(self) -> int:
        return len(self.params)

    def display_params(self) -> list[str]:
        base = self._bound_name or "op"
        return [f"{base}_{p.lower()}" for p in self.params]

    def __repr__(self) -> str:
        name = f"{self._bound_name} := " if self._bound_name else ""
        return f"{name}Operator(params={list(self.params)}, root={self.root!r})"

    # -- definition combinators ------------------------------------------------------
    def _unary(self, opcode: str) -> "Operator":
        uid = fresh_uid()
        root = ir.remap_params(self.root, uid)
        return Operator(self.params, ir.UnOp(opcode, root), uid)

    def __add__(self, other): return _combine("add", self, other)
    def __radd__(self, other): return _combine("add", other, self)
    def __sub__(self, other): return _combine("sub", self, other)
    def __rsub__(self, other): return _combine("sub", other, self)
    def __mul__(self, other): return _combine("mul", self, other)
    def __rmul__(self, other): return _combine("mul", other, self)
    def __truediv__(self, other): return _combine("div", self, other)
    def __rtruediv__(self, other): return _combine("div", other, self)
    def __pow__(self, other): return _combine("pow", self, other)
    def __rpow__(self, other): return _combine("pow", other, self)
    def __mod__(self, other): return _combine("mod", self, other)
    def __rmod__(self, other): return _combine("mod", other, self)
    def __matmul__(self, other): return _combine("matmul", self, other)
    def __eq__(self, other): return _combine("eq", self, other)  # type: ignore[override]
    def __ne__(self, other): return _combine("ne", self, other)  # type: ignore[override]
    def __gt__(self, other): return _combine("gt", self, other)
    def __lt__(self, other): return _combine("lt", self, other)
    def __ge__(self, other): return _combine("ge", self, other)
    def __le__(self, other): return _combine("le", self, other)
    def __and__(self, other): return _combine("and", self, other)
    def __rand__(self, other): return _combine("and", other, self)
    def __or__(self, other): return _combine("or", self, other)
    def __ror__(self, other): return _combine("or", other, self)
    def __neg__(self): return self._unary("neg")
    def __invert__(self): return self._unary("not")
    def __abs__(self): return self._unary("abs")

    def _reduce(self, op: str, by=None) -> "Operator":
        uid = fresh_uid()
        root = ir.remap_params(self.root, uid)
        if by is None:
            by_node = None
        elif isinstance(by, str):
            by_node = ir.Const(by)
        elif isinstance(by, Operator):
            missing = [p for p in by.params if p not in self.params]
            if missing:
                raise TLTypeError(
                    f"cannot group by params {missing}: unbound in this definition"
                )
            by_node = ir.remap_params(by.root, uid)
        else:
            raise TLTypeError(
                f"group key must be a field name or operator, got "
                f"{type(by).__name__}")
        return Operator(self.params, ir.Reduce(op, root, by_node), uid)

    def sum(self, by=None) -> "Operator":
        """Reduction over the operator expression (e.g. loss means)."""
        return self._reduce("sum", by)

    def mean(self, by=None) -> "Operator":
        return self._reduce("mean", by)

    def min(self, by=None) -> "Operator":
        return self._reduce("min", by)

    def max(self, by=None) -> "Operator":
        return self._reduce("max", by)

    def std(self, by=None) -> "Operator":
        return self._reduce("std", by)

    def var(self, by=None) -> "Operator":
        return self._reduce("var", by)

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        raise TLTypeError(
            "truth value of an operator is undefined; apply it to data first"
        )

    def __getattr__(self, name: str) -> "Operator":
        # Field projection, e.g. {D}.amount.  Only called when normal lookup
        # misses, so real methods/properties always win.
        if name.startswith("_"):
            raise AttributeError(name)
        uid = fresh_uid()
        root = ir.remap_params(self.root, uid)
        return Operator(self.params, ir.FieldAccess(root, name), uid)

    def __getitem__(self, index) -> "Operator":
        uid = fresh_uid()
        root = ir.remap_params(self.root, uid)
        if isinstance(index, Operator):
            missing = [p for p in index.params if p not in self.params]
            if missing:
                raise TLTypeError(
                    f"cannot use operator with params {list(index.params)} as "
                    f"an index: params {missing} are unbound"
                )
            idx = ir.remap_params(index.root, uid)
        else:
            idx = _operator_index(index)
        return Operator(self.params, ir.GetItem(root, idx), uid)

    # -- composition -------------------------------------------------------------------
    def __rshift__(self, other: "Operator") -> "Operator":
        """``f >> g``: left-to-right composition, ``(f >> g)(x) = g(f(x))``."""
        if getattr(other, "_tl_fittable_", False):
            # Fittable right-hand side: let it build a fittable pipeline
            # (avoids importing estimator machinery into core).
            return other._tl_rcompose(self)
        if not isinstance(other, Operator):
            raise TLTypeError(
                f"cannot compose an operator with {type(other).__name__}; "
                f"both sides of '>>' must be operators"
            )
        if len(other.params) != 1:
            raise TLTypeError(
                f"composition requires a unary right-hand operator, got params "
                f"{list(other.params)}"
            )
        if len(self.params) == 0:
            raise TLTypeError("cannot compose a zero-parameter operator on the left")
        uid = fresh_uid()
        params = self.params
        inner = ir.ApplyOp(self, tuple(ir.Param(uid, p) for p in params))
        body = ir.substitute(other.root, {(other.uid, other.params[0]): inner})
        return Operator(params, body, uid)

    # -- application ----------------------------------------------------------------------
    def __call__(self, *args: Any) -> Lazy:
        if len(args) != len(self.params):
            raise TLShapeError(
                f"operator expects {len(self.params)} argument(s) "
                f"{list(self.params)}, got {len(args)}"
            )
        env: dict[tuple[str, str], ir.Expr] = {}
        target: Tensor | None = None
        if len(args) == 1 and isinstance(args[0], Tensor):
            target = args[0]
        for name, value in zip(self.params, args):
            if isinstance(value, Operator):
                raise TLTypeError("cannot apply an operator to another operator")
            env[(self.uid, name)] = _bind_arg(value)
        return Lazy(ir.substitute(self.root, env), target)

    def collect(self):
        """Materialize a zero-parameter (constant) operator, e.g. after
        exact definite integration."""
        if self.params:
            raise TLTypeError(
                f"collect() needs operator arguments; call the operator "
                f"with data for params {list(self.params)} first"
            )
        if isinstance(self.root, ir.Const):
            # Bare constants keep their exact representation (e.g. 125/3).
            return self.root.value
        from ..runtime.executor import evaluate, collect_value

        value, kind = evaluate(self.root)
        return collect_value(value, kind)

    def fit(self, data):
        """Learn parameters from data. Only fittable operators
        (``standardize()``, ``pca(2)``, … and their ``>>`` pipelines)
        support fitting; fitting is an explicit runtime boundary, so the
        result must be bound with ``=``, not ``:=``."""
        raise TLTypeError(
            "only fittable operators support .fit(): build one from "
            "standardize(), minmax(), normalize(), impute() or pca(), "
            "optionally composed with '>>'"
        )

    def explain(self) -> str:
        """Logical graph of this operator (params stay symbolic)."""
        from ..runtime.inspect import format_node
        from ..runtime.planner import plan

        graph = plan(self.root)
        lines = [
            f"operator(params={list(self.params)}"
            + (f", bound as '{self._bound_name}'" if self._bound_name else "")
            + "):",
            format_node(self.root),
            "",
            f"graph: {graph.node_count} nodes; backends {graph.summary()}",
        ]
        return "\n".join(lines)


def _operator_index(index: Any) -> ir.Expr:
    if isinstance(index, slice):
        return ir.Const(index)
    if isinstance(index, tuple):
        return ir.Const(tuple(index))
    if isinstance(index, list):
        if all(isinstance(x, str) for x in index):
            return ir.Const(list(index))
        raise TLTypeError(f"unsupported operator index {index!r}")
    if isinstance(index, (int, str, bool)):
        return ir.Const(index)
    raise TLTypeError(f"unsupported operator index {index!r}")


def _bind_arg(value: Any) -> ir.Expr:
    if isinstance(value, Lazy):
        return value.root
    if isinstance(value, Tensor):
        return value._snapshot_node()
    return expr_of(value, "operator application")
