"""TensorLang intermediate representation.

Immutable, typed expression nodes describing computations and dependencies.
``Param`` leaves are identified by ``(owner, name)`` where ``owner`` is the
uid of the operator definition that introduced them, so identically named
parameters from different definitions never collide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import ScanSpec


# Operator codes shared by the frontend, planner and backends.
BINOPS = frozenset({
    "add", "sub", "mul", "div", "pow", "mod", "matmul",
    "eq", "ne", "gt", "lt", "ge", "le", "and", "or",
})
UNOPS = frozenset({"neg", "not", "abs"})
MATH_FUNCS = frozenset({"sin", "cos", "tan", "exp", "log", "sqrt"})
REDUCTIONS = frozenset({"sum", "mean", "min", "max", "std", "var"})


@dataclass(frozen=True)
class Expr:
    """Base class for all IR nodes."""


@dataclass(frozen=True)
class Const(Expr):
    """A literal scalar (number, bool, str, or exact sympy value)."""

    value: Any


@dataclass(frozen=True)
class Literal(Expr):
    """An immutable snapshot of runtime data (ndarray / DataFrame / Series)."""

    value: Any
    kind: str  # 'dense' | 'frame' | 'series' | 'scalar'


@dataclass(frozen=True)
class Param(Expr):
    """A lexically scoped symbolic operator parameter, e.g. ``{X}``."""

    owner: str
    name: str

    def display(self, bound: str | None = None) -> str:
        base = bound if bound else "op"
        return f"{base}_{self.name.lower()}"


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in BINOPS:
            raise ValueError(f"unknown binary operator {self.op!r}")


@dataclass(frozen=True)
class UnOp(Expr):
    op: str
    operand: Expr

    def __post_init__(self) -> None:
        if self.op not in UNOPS:
            raise ValueError(f"unknown unary operator {self.op!r}")


@dataclass(frozen=True)
class FuncCall(Expr):
    func: str
    args: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if self.func not in MATH_FUNCS and self.func not in {
            "is_null", "not_null", "drop_null",
        }:
            raise ValueError(f"unknown function {self.func!r}")


@dataclass(frozen=True)
class FillNull(Expr):
    operand: Expr
    value: Expr


@dataclass(frozen=True)
class FieldAccess(Expr):
    """``{D}.amount`` — named-field projection of a structured value."""

    obj: Expr
    field: str


@dataclass(frozen=True)
class GetItem(Expr):
    """Indexing / slicing / masking / field selection."""

    obj: Expr
    index: Expr


@dataclass(frozen=True)
class Transpose(Expr):
    obj: Expr


@dataclass(frozen=True)
class Reshape(Expr):
    obj: Expr
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Flatten(Expr):
    obj: Expr


@dataclass(frozen=True)
class Concat(Expr):
    parts: tuple[Expr, ...]
    axis: int = 0


@dataclass(frozen=True)
class Stack(Expr):
    parts: tuple[Expr, ...]
    axis: int = 0


@dataclass(frozen=True)
class Reduce(Expr):
    op: str
    operand: Expr
    by: Optional[Expr] = None
    axis: object = None
    keepdims: bool = False

    def __post_init__(self) -> None:
        if self.op not in REDUCTIONS:
            raise ValueError(f"unknown reduction {self.op!r}")
        object.__setattr__(self, "axis", _normalize_axis(self.axis))
        if not isinstance(self.keepdims, bool):
            raise TLTypeError(
                f"reduction keepdims must be bool, got {self.keepdims!r}",
                code="TL0511",
                details={"op": self.op, "keepdims": repr(self.keepdims)})


def _normalize_axis(axis) -> tuple | None:
    """Validate axis *types* (not rank); rank checks need input shape."""
    import operator as _operator

    from .errors import TLTypeError as _TLTypeError

    def _bad() -> _TLTypeError:
        return _TLTypeError(f"reduction axis must be int or tuple of ints, "
                            f"got {axis!r}", code="TL0511",
                            details={"axis": repr(axis)})

    if axis is None:
        return None
    if isinstance(axis, bool):
        raise _bad()
    if isinstance(axis, (tuple, list)):
        items = list(axis)
    else:
        items = [axis]
    out = []
    for ax in items:
        if isinstance(ax, bool):
            raise _bad()
        try:
            out.append(_operator.index(ax))
        except TypeError:
            raise _bad() from None
    return tuple(out)


def resolve_axes(axis: tuple | None, ndim: int, op: str,
                 shape: tuple) -> tuple:
    """Normalize axes against rank (negative wrap, duplicate/empty drop).

    Returns canonical sorted axes. Raises TL0511 (op, axes, rank, shape)
    for out-of-range entries.
    """
    from .errors import TLShapeError as _TLShapeError

    if axis is None:
        return tuple(range(ndim))
    if len(axis) == 0:
        return ()  # NumPy: axis=() reduces nothing, even on rank-0 input
    if ndim == 0:
        raise _TLShapeError(
            f"reduction '{op}' over axes {list(axis)} of rank-0 input "
            f"(shape {list(shape)}): no axes to reduce",
            code="TL0511",
            details={"op": op, "axes": list(axis), "rank": ndim,
                     "shape": list(shape)})
    resolved = []
    for ax in axis:
        norm = ax + ndim if ax < 0 else ax
        if norm < 0 or norm >= ndim:
            raise _TLShapeError(
                f"reduction '{op}' axis {ax} out of range for rank {ndim} "
                f"(shape {list(shape)})",
                code="TL0511",
                details={"op": op, "axes": list(axis), "rank": ndim,
                         "shape": list(shape)})
        if norm not in resolved:
            resolved.append(norm)
    return tuple(sorted(resolved))


@dataclass(frozen=True)
class ApplyOp(Expr):
    """Application of a (sub-)operator embedded inside an expression.

    Used by operator composition (``>>``) and by embedding one operator
    inside another operator's definition. ``args`` are bound positionally
    to the embedded operator's parameters at evaluation time.
    """

    operator: Any  # Operator; typed as Any to avoid a circular import
    args: tuple[Expr, ...] = field(default_factory=tuple)


# -- v0.2 data-processing nodes -------------------------------------------------


@dataclass(frozen=True)
class Scan(Expr):
    """A lazy file-source descriptor (declarative: reads nothing)."""

    spec: ScanSpec


@dataclass(frozen=True)
class Chunked(Expr):
    """A reference to persistent chunked storage (path only; metadata
    loads lazily at planning time, data only at execution boundaries)."""

    path: str


@dataclass(frozen=True)
class WithField(Expr):
    """Named-field assignment as a lazy graph node (``t["net"] = ...``)."""

    obj: Expr
    name: str
    value: Expr


@dataclass(frozen=True)
class Join(Expr):
    """Relational join with explicit keys (``how`` one of JOIN_HOW)."""

    left: Expr
    right: Expr
    on: tuple[str, ...]
    how: str = "inner"
    suffix: str = "_right"


@dataclass(frozen=True)
class SortBy(Expr):
    """Explicit ordering (lazy/streaming execution preserves no order)."""

    obj: Expr
    keys: tuple[str, ...] = ()
    descending: bool = False


@dataclass(frozen=True)
class Unique(Expr):
    """Unique values (dense/series) or unique rows (frames, order kept)."""

    obj: Expr


@dataclass(frozen=True)
class Where(Expr):
    """Conditional replacement: ``where(cond, a, b)`` elementwise."""

    cond: Expr
    a: Expr
    b: Expr


@dataclass(frozen=True)
class Clip(Expr):
    """Clamp values to ``[lo, hi]`` (either bound may be None)."""

    obj: Expr
    lo: Expr
    hi: Expr


@dataclass(frozen=True)
class Over(Expr):
    """Grouped transformation: per-group aggregate broadcast to rows."""

    op: str
    operand: Expr
    by: Optional[Expr] = None

    def __post_init__(self) -> None:
        if self.op not in REDUCTIONS:
            raise ValueError(f"unknown window operator {self.op!r}")


@dataclass(frozen=True)
class Rolling(Expr):
    """Sliding-window aggregation over dense data or field series."""

    operand: Expr
    window: int
    op: str = "mean"
    min_periods: Optional[int] = None

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("rolling window must be >= 1")
        if self.op not in {"sum", "mean", "min", "max"}:
            raise ValueError(f"unknown rolling operator {self.op!r}")


# -- v0.3 ML and numerical linear algebra nodes ---------------------------------


@dataclass(frozen=True)
class FittedTransform(Expr):
    """A learned transformation: ``kind`` estimator with ``config`` applied
    to ``arg``.  ``state`` is None until fitted (then a FittedState)."""

    kind: str
    config: tuple[tuple[str, Any], ...]
    state: Any
    arg: Expr


@dataclass(frozen=True)
class SvdPart(Expr):
    """One component (0=U, 1=S, 2=Vh) of a shared SVD computation."""

    shared: Any
    index: int


@dataclass(frozen=True)
class EighPart(Expr):
    """One component (0=eigenvalues, 1=eigenvectors) of a shared eigh."""

    shared: Any
    index: int


@dataclass(frozen=True)
class Solve(Expr):
    """Solve the square linear system ``a x = b``."""

    a: Expr
    b: Expr


@dataclass(frozen=True)
class Norm(Expr):
    """Matrix/vector norm (NumPy ``ord``/``axis`` conventions)."""

    obj: Expr
    ord: Any = None
    axis: Any = None


@dataclass(frozen=True)
class Cov(Expr):
    """Covariance matrix (columns-as-variables for frames)."""

    obj: Expr
    rowvar: bool = True


@dataclass(frozen=True)
class Corr(Expr):
    """Correlation matrix (columns-as-variables for frames)."""

    obj: Expr
    rowvar: bool = True


@dataclass(frozen=True)
class Quantile(Expr):
    """Quantiles of data at probabilities ``q``."""

    obj: Expr
    q: tuple[float, ...] = (0.5,)


@dataclass(frozen=True)
class Einsum(Expr):
    """Einstein summation over ``parts`` per subscript ``spec``."""

    spec: str
    parts: tuple[Expr, ...]


@dataclass(frozen=True)
class Tensordot(Expr):
    """Tensor dot product along ``axes`` (NumPy conventions)."""

    a: Expr
    b: Expr
    axes: Any = 2


@dataclass(frozen=True)
class MoveAxis(Expr):
    """Move array axes (NumPy conventions)."""

    obj: Expr
    source: tuple[int, ...] = ()
    destination: tuple[int, ...] = ()


@dataclass(frozen=True)
class OneHot(Expr):
    """One-hot encoding; ``categories`` pins coverage/order (else inferred)."""

    obj: Expr
    categories: Optional[tuple] = None


@dataclass(frozen=True)
class Polynomial(Expr):
    """Polynomial features up to ``degree`` (bias column if requested)."""

    obj: Expr
    degree: int = 2
    include_bias: bool = True

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("polynomial degree must be >= 1")


# -- v0.4 automatic differentiation nodes ---------------------------------------


@dataclass(frozen=True)
class DiffVar(Expr):
    """A differentiable input leaf: parameter ``(owner, name)`` bound to a
    snapshotted ``value`` at operator-application time."""

    owner: str
    name: str
    value: Expr


@dataclass(frozen=True)
class AdjointOf(Expr):
    """Reverse-mode gradient(s) of a closed graph held by ``shared``."""

    shared: Any
    indices: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class ForwardOf(Expr):
    """Forward value of a closed graph held by ``shared``."""

    shared: Any


def iter_params(node: Expr) -> set[tuple[str, str]]:
    """Return the set of ``(owner, name)`` params referenced by ``node``."""
    found: set[tuple[str, str]] = set()

    def visit(n: Expr) -> None:
        if isinstance(n, Param):
            found.add((n.owner, n.name))
        elif isinstance(n, BinOp):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, UnOp):
            visit(n.operand)
        elif isinstance(n, FuncCall):
            for a in n.args:
                visit(a)
        elif isinstance(n, FillNull):
            visit(n.operand)
            visit(n.value)
        elif isinstance(n, FieldAccess):
            visit(n.obj)
        elif isinstance(n, GetItem):
            visit(n.obj)
            visit(n.index)
        elif isinstance(n, Transpose):
            visit(n.obj)
        elif isinstance(n, Reshape):
            visit(n.obj)
        elif isinstance(n, Flatten):
            visit(n.obj)
        elif isinstance(n, Concat):
            for p in n.parts:
                visit(p)
        elif isinstance(n, Stack):
            for p in n.parts:
                visit(p)
        elif isinstance(n, Reduce):
            visit(n.operand)
            if n.by is not None:
                visit(n.by)
        elif isinstance(n, ApplyOp):
            for a in n.args:
                visit(a)
        elif isinstance(n, Scan):
            pass
        elif isinstance(n, Chunked):
            pass
        elif isinstance(n, WithField):
            visit(n.obj)
            visit(n.value)
        elif isinstance(n, Join):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, (SortBy, Unique)):
            visit(n.obj)
        elif isinstance(n, Where):
            visit(n.cond)
            visit(n.a)
            visit(n.b)
        elif isinstance(n, Clip):
            visit(n.obj)
            visit(n.lo)
            visit(n.hi)
        elif isinstance(n, Over):
            visit(n.operand)
            if n.by is not None:
                visit(n.by)
        elif isinstance(n, Rolling):
            visit(n.operand)
        elif isinstance(n, FittedTransform):
            visit(n.arg)
        elif isinstance(n, (SvdPart, EighPart)):
            pass  # shared holders bind no parameters
        elif isinstance(n, Solve):
            visit(n.a)
            visit(n.b)
        elif isinstance(n, (Norm, Cov, Corr)):
            visit(n.obj)
        elif isinstance(n, Quantile):
            visit(n.obj)
        elif isinstance(n, Einsum):
            for p in n.parts:
                visit(p)
        elif isinstance(n, Tensordot):
            visit(n.a)
            visit(n.b)
        elif isinstance(n, (MoveAxis, OneHot, Polynomial)):
            visit(n.obj)
        elif isinstance(n, DiffVar):
            found.add((n.owner, n.name))
        elif isinstance(n, (AdjointOf, ForwardOf)):
            pass  # holders always wrap closed, already-bound graphs
        elif isinstance(n, (Const, Literal)):
            pass
        else:  # pragma: no cover - defensive
            raise TypeError(f"unknown Expr node: {type(n).__name__}")

    visit(node)
    return found


def remap_params(node: Expr, owner: str) -> Expr:
    """Rebuild ``node`` with every ``Param`` rebound to ``owner`` (by name).

    Sharing-preserving (v0.6 M1): subtrees needing no rebinding are
    returned as-is (no copy), and repeated references within one call map
    to a single copy. Output is structurally identical to a full rebuild,
    so parameter-identity semantics are unchanged while downstream
    evaluation memoizes shared nodes instead of recomputing them.
    """
    memo: dict[int, Expr] = {}

    def visit(n: Expr) -> Expr:
        if isinstance(n, Param):
            return n if n.owner == owner else Param(owner, n.name)
        if isinstance(n, DiffVar):
            return n if n.owner == owner else DiffVar(owner, n.name, n.value)
        key = id(n)
        hit = memo.get(key)
        if hit is not None:
            return hit
        out = _remap_rebuild(n)
        memo[key] = out
        return out

    def _same(items: tuple) -> bool:
        return all(new is old for new, old in items)

    def _remap_rebuild(n: Expr) -> Expr:
        if isinstance(n, BinOp):
            left, right = visit(n.left), visit(n.right)
            return n if left is n.left and right is n.right else BinOp(
                n.op, left, right)
        if isinstance(n, UnOp):
            operand = visit(n.operand)
            return n if operand is n.operand else UnOp(n.op, operand)
        if isinstance(n, FuncCall):
            args = tuple(visit(a) for a in n.args)
            return n if _same(zip(args, n.args)) else FuncCall(n.func, args)
        if isinstance(n, FillNull):
            operand, value = visit(n.operand), visit(n.value)
            return (n if operand is n.operand and value is n.value
                    else FillNull(operand, value))
        if isinstance(n, FieldAccess):
            obj = visit(n.obj)
            return n if obj is n.obj else FieldAccess(obj, n.field)
        if isinstance(n, GetItem):
            obj, index = visit(n.obj), visit(n.index)
            return (n if obj is n.obj and index is n.index
                    else GetItem(obj, index))
        if isinstance(n, Transpose):
            obj = visit(n.obj)
            return n if obj is n.obj else Transpose(obj)
        if isinstance(n, Reshape):
            obj = visit(n.obj)
            return n if obj is n.obj else Reshape(obj, n.shape)
        if isinstance(n, Flatten):
            obj = visit(n.obj)
            return n if obj is n.obj else Flatten(obj)
        if isinstance(n, Concat):
            parts = tuple(visit(p) for p in n.parts)
            return n if _same(zip(parts, n.parts)) else Concat(parts, n.axis)
        if isinstance(n, Stack):
            parts = tuple(visit(p) for p in n.parts)
            return n if _same(zip(parts, n.parts)) else Stack(parts, n.axis)
        if isinstance(n, Reduce):
            operand = visit(n.operand)
            by = visit(n.by) if n.by is not None else None
            return (n if operand is n.operand and by is n.by
                    else Reduce(n.op, operand, by, n.axis, n.keepdims))
        if isinstance(n, ApplyOp):
            # Embedded operators keep their own independent scopes.
            args = tuple(visit(a) for a in n.args)
            return (n if _same(zip(args, n.args))
                    else ApplyOp(n.operator, args))
        if isinstance(n, Scan):
            return n
        if isinstance(n, Chunked):
            return n
        if isinstance(n, WithField):
            obj, value = visit(n.obj), visit(n.value)
            return (n if obj is n.obj and value is n.value
                    else WithField(obj, n.name, value))
        if isinstance(n, Join):
            left, right = visit(n.left), visit(n.right)
            return (n if left is n.left and right is n.right
                    else Join(left, right, n.on, n.how, n.suffix))
        if isinstance(n, SortBy):
            obj = visit(n.obj)
            return (n if obj is n.obj
                    else SortBy(obj, n.keys, n.descending))
        if isinstance(n, Unique):
            obj = visit(n.obj)
            return n if obj is n.obj else Unique(obj)
        if isinstance(n, Where):
            cond, a, b = visit(n.cond), visit(n.a), visit(n.b)
            return (n if cond is n.cond and a is n.a and b is n.b
                    else Where(cond, a, b))
        if isinstance(n, Clip):
            obj, lo, hi = visit(n.obj), visit(n.lo), visit(n.hi)
            return (n if obj is n.obj and lo is n.lo and hi is n.hi
                    else Clip(obj, lo, hi))
        if isinstance(n, Over):
            operand = visit(n.operand)
            by = visit(n.by) if n.by is not None else None
            return (n if operand is n.operand and by is n.by
                    else Over(n.op, operand, by))
        if isinstance(n, Rolling):
            operand = visit(n.operand)
            return (n if operand is n.operand
                    else Rolling(operand, n.window, n.op, n.min_periods))
        if isinstance(n, FittedTransform):
            arg = visit(n.arg)
            return (n if arg is n.arg
                    else FittedTransform(n.kind, n.config, n.state, arg))
        if isinstance(n, (SvdPart, EighPart)):
            return n
        if isinstance(n, Solve):
            a, b = visit(n.a), visit(n.b)
            return n if a is n.a and b is n.b else Solve(a, b)
        if isinstance(n, Norm):
            obj = visit(n.obj)
            return (n if obj is n.obj
                    else Norm(obj, n.ord, n.axis))
        if isinstance(n, Cov):
            obj = visit(n.obj)
            return n if obj is n.obj else Cov(obj, n.rowvar)
        if isinstance(n, Corr):
            obj = visit(n.obj)
            return n if obj is n.obj else Corr(obj, n.rowvar)
        if isinstance(n, Quantile):
            obj = visit(n.obj)
            return n if obj is n.obj else Quantile(obj, n.q)
        if isinstance(n, Einsum):
            parts = tuple(visit(p) for p in n.parts)
            return (n if _same(zip(parts, n.parts))
                    else Einsum(n.spec, parts))
        if isinstance(n, Tensordot):
            a, b = visit(n.a), visit(n.b)
            return (n if a is n.a and b is n.b
                    else Tensordot(a, b, n.axes))
        if isinstance(n, MoveAxis):
            obj = visit(n.obj)
            return (n if obj is n.obj
                    else MoveAxis(obj, n.source, n.destination))
        if isinstance(n, OneHot):
            obj = visit(n.obj)
            return n if obj is n.obj else OneHot(obj, n.categories)
        if isinstance(n, Polynomial):
            obj = visit(n.obj)
            return (n if obj is n.obj
                    else Polynomial(obj, n.degree, n.include_bias))
        return n

    return visit(node)


def substitute(node: Expr, env: dict[tuple[str, str], Expr]) -> Expr:
    """Replace ``Param`` leaves using ``env``; unbound params are kept."""

    def visit(n: Expr) -> Expr:
        if isinstance(n, Param):
            return env.get((n.owner, n.name), n)
        if isinstance(n, DiffVar):
            # Already-bound differentiable inputs pass through untouched.
            return n
        if isinstance(n, BinOp):
            return BinOp(n.op, visit(n.left), visit(n.right))
        if isinstance(n, UnOp):
            return UnOp(n.op, visit(n.operand))
        if isinstance(n, FuncCall):
            return FuncCall(n.func, tuple(visit(a) for a in n.args))
        if isinstance(n, FillNull):
            return FillNull(visit(n.operand), visit(n.value))
        if isinstance(n, FieldAccess):
            return FieldAccess(visit(n.obj), n.field)
        if isinstance(n, GetItem):
            return GetItem(visit(n.obj), visit(n.index))
        if isinstance(n, Transpose):
            return Transpose(visit(n.obj))
        if isinstance(n, Reshape):
            return Reshape(visit(n.obj), n.shape)
        if isinstance(n, Flatten):
            return Flatten(visit(n.obj))
        if isinstance(n, Concat):
            return Concat(tuple(visit(p) for p in n.parts), n.axis)
        if isinstance(n, Stack):
            return Stack(tuple(visit(p) for p in n.parts), n.axis)
        if isinstance(n, Reduce):
            return Reduce(n.op, visit(n.operand),
                          visit(n.by) if n.by is not None else None,
                          n.axis, n.keepdims)
        if isinstance(n, ApplyOp):
            return ApplyOp(n.operator, tuple(visit(a) for a in n.args))
        if isinstance(n, Scan):
            return n
        if isinstance(n, Chunked):
            return n
        if isinstance(n, WithField):
            return WithField(visit(n.obj), n.name, visit(n.value))
        if isinstance(n, Join):
            return Join(visit(n.left), visit(n.right), n.on, n.how, n.suffix)
        if isinstance(n, SortBy):
            return SortBy(visit(n.obj), n.keys, n.descending)
        if isinstance(n, Unique):
            return Unique(visit(n.obj))
        if isinstance(n, Where):
            return Where(visit(n.cond), visit(n.a), visit(n.b))
        if isinstance(n, Clip):
            return Clip(visit(n.obj), visit(n.lo), visit(n.hi))
        if isinstance(n, Over):
            return Over(n.op, visit(n.operand),
                        visit(n.by) if n.by is not None else None)
        if isinstance(n, Rolling):
            return Rolling(visit(n.operand), n.window, n.op, n.min_periods)
        if isinstance(n, FittedTransform):
            return FittedTransform(n.kind, n.config, n.state, visit(n.arg))
        if isinstance(n, (SvdPart, EighPart)):
            return n
        if isinstance(n, Solve):
            return Solve(visit(n.a), visit(n.b))
        if isinstance(n, Norm):
            return Norm(visit(n.obj), n.ord, n.axis)
        if isinstance(n, Cov):
            return Cov(visit(n.obj), n.rowvar)
        if isinstance(n, Corr):
            return Corr(visit(n.obj), n.rowvar)
        if isinstance(n, Quantile):
            return Quantile(visit(n.obj), n.q)
        if isinstance(n, Einsum):
            return Einsum(n.spec, tuple(visit(p) for p in n.parts))
        if isinstance(n, Tensordot):
            return Tensordot(visit(n.a), visit(n.b), n.axes)
        if isinstance(n, MoveAxis):
            return MoveAxis(visit(n.obj), n.source, n.destination)
        if isinstance(n, OneHot):
            return OneHot(visit(n.obj), n.categories)
        if isinstance(n, Polynomial):
            return Polynomial(visit(n.obj), n.degree, n.include_bias)
        if isinstance(n, DiffVar):
            return DiffVar(n.owner, n.name, n.value)
        if isinstance(n, (AdjointOf, ForwardOf)):
            return n
        return n

    return visit(node)
