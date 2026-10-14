"""Lower compatible structured subgraphs into one Polars LazyFrame plan.

Instead of dispatching operators one by one (each materializing an
intermediate), ``collect_lowered`` / ``try_sink`` recognize whole pipeline
regions — scan → filter → derive → sort → join → aggregate — and hand them
to Polars as a single lazy query.  Polars then applies its own predicate
pushdown, projection pushdown, and subplan elimination across the entire
computation.

Anything that cannot be lowered (dense arithmetic over whole frames,
Python-side values, genuinely eager constructs) raises ``_NotLowerable``
with a reason, and the executor falls back to node-by-node evaluation —
so optimization never changes an observable result.
"""
from __future__ import annotations

import polars as pl

from ..backends import sympy_backend as _sym
from ..core import expression as ir
from ..core.errors import TLTypeError
from . import io as _io


class _NotLowerable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_MATH_METHODS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "exp": "exp", "log": "log", "sqrt": "sqrt",
}

_BINOP_METHODS = {
    "add": "__add__", "sub": "__sub__", "mul": "__mul__",
    "div": "__truediv__", "pow": "__pow__", "mod": "__mod__",
    "eq": "__eq__", "ne": "__ne__", "gt": "__gt__", "lt": "__lt__",
    "ge": "__ge__", "le": "__le__", "and": "__and__", "or": "__or__",
}


class _LowerContext:
    def __init__(self) -> None:
        self.scans: dict[tuple, pl.LazyFrame] = {}
        self.used_scan = False

    def scan(self, node: ir.Scan) -> pl.LazyFrame:
        key = (node.spec.paths, node.spec.format, node.spec.options)
        if key not in self.scans:
            try:
                self.scans[key] = _io.lazy_scan(node.spec)
            except Exception as exc:
                raise _NotLowerable(f"cannot plan {node.spec.describe()}: {exc}")
        self.used_scan = True
        return self.scans[key]


def lower_frame(node: ir.Expr, ctx: _LowerContext) -> pl.LazyFrame:
    """Lower a frame-producing node to a single LazyFrame."""
    if isinstance(node, ir.Scan):
        return ctx.scan(node)
    if isinstance(node, ir.Literal):
        if node.kind != "frame":
            raise _NotLowerable(f"{node.kind} literals are not lazy frames")
        return node.value.lazy()
    if isinstance(node, ir.GetItem):
        return _lower_getitem(node, ctx)
    if isinstance(node, ir.WithField):
        base = lower_frame(node.obj, ctx)
        return base.with_columns(lower_expr(node.value, ctx).alias(node.name))
    if isinstance(node, ir.SortBy):
        if not node.keys:
            raise _NotLowerable("sort without keys is not a lazy plan")
        return lower_frame(node.obj, ctx).sort(list(node.keys),
                                               descending=node.descending)
    if isinstance(node, ir.Unique):
        return lower_frame(node.obj, ctx).unique(maintain_order=True)
    if isinstance(node, ir.Join):
        left = lower_frame(node.left, ctx)
        right = lower_frame(node.right, ctx)
        if node.how == "cross":
            return left.join(right, how="cross", suffix=node.suffix)
        return left.join(right, on=list(node.on), how=node.how,
                         suffix=node.suffix)
    if isinstance(node, ir.Reduce):
        return _lower_reduce(node, ctx)
    if isinstance(node, ir.FuncCall) and node.func == "drop_null":
        return lower_frame(node.args[0], ctx).drop_nulls()
    raise _NotLowerable(f"'{type(node).__name__}' has no lazy-frame lowering")


def _lower_getitem(node: ir.GetItem, ctx: _LowerContext) -> pl.LazyFrame:
    if isinstance(node.index, ir.Const):
        value = node.index.value
        if isinstance(value, str):
            return lower_frame(node.obj, ctx).select(value)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return lower_frame(node.obj, ctx).select(value)
        if isinstance(value, slice):
            start, stop, step = value.indices(2**63 - 1)
            if step != 1:
                raise _NotLowerable("strided row slices are not lazily plannable")
            length = None if stop == 2**63 - 1 else max(stop - start, 0)
            return lower_frame(node.obj, ctx).slice(start, length)
        if isinstance(value, int):
            i = value
            return lower_frame(node.obj, ctx).slice(i, 1)
    base = lower_frame(node.obj, ctx)
    return base.filter(lower_expr(node.index, ctx))


def _lower_reduce(node: ir.Reduce, ctx: _LowerContext) -> pl.LazyFrame:
    base, numeric = _reduce_source(node, ctx)
    agg = node.op
    if node.axis is not None:
        # Axis reductions over structured frames are ambiguous (rows vs
        # fields); decline lowering so the executor reports TL0511.
        raise _NotLowerable(
            f"reduction '{node.op}' with axis has no lazy-frame lowering")
    if node.by is None:
        return base.select([getattr(pl.col(c), agg)().alias(c) for c in numeric])
    key = _group_key(node.by)
    if key is None:
        raise _NotLowerable("group keys must be field names for lazy planning")
    try:
        schema = base.collect_schema()
    except Exception as exc:
        raise _NotLowerable(f"cannot plan reduction schema: {exc}")
    if key not in schema:
        raise TLTypeError(f"unknown group field '{key}'")
    cols = [c for c in numeric if c != key]
    if not cols:
        raise _NotLowerable(f"grouped '{node.op}' has no plannable numeric fields")
    return base.group_by(key).agg(
        [getattr(pl.col(c), agg)().alias(c) for c in cols]
    ).sort(key)


def _reduce_source(node: ir.Reduce, ctx: _LowerContext) -> tuple[pl.LazyFrame, list[str]]:
    """Lower the reduction operand; unwrap single-field projections."""
    operand = node.operand
    if isinstance(operand, ir.FieldAccess):
        base = lower_frame(operand.obj, ctx)
        try:
            schema = base.collect_schema()
        except Exception as exc:
            raise _NotLowerable(f"cannot plan reduction schema: {exc}")
        dtype = schema.get(operand.field)
        if dtype is None:
            raise TLTypeError(f"unknown field '{operand.field}'")
        if not dtype.is_numeric():
            raise _NotLowerable(
                f"reduction '{node.op}' of non-numeric field '{operand.field}'"
            )
        return base, [operand.field]
    base = lower_frame(operand, ctx)
    try:
        schema = base.collect_schema()
    except Exception as exc:
        raise _NotLowerable(f"cannot plan reduction schema: {exc}")
    numeric = [n for n, d in schema.items() if d.is_numeric()]
    if not numeric:
        raise _NotLowerable(f"reduction '{node.op}' has no plannable numeric fields")
    return base, numeric


def _group_key(by: ir.Expr) -> str | None:
    if isinstance(by, ir.Const) and isinstance(by.value, str):
        return by.value
    if isinstance(by, ir.FieldAccess):
        return by.field
    return None


def lower_expr(node: ir.Expr, ctx: _LowerContext) -> pl.Expr:
    """Lower a row-level expression to a Polars expression."""
    if isinstance(node, ir.FieldAccess):
        lower_frame(node.obj, ctx)  # validates plannability of the source
        return pl.col(node.field)
    if isinstance(node, ir.Const):
        return _lower_const(node.value)
    if isinstance(node, ir.BinOp):
        if node.op == "matmul":
            raise _NotLowerable("matrix multiplication is not a row expression")
        left = lower_expr(node.left, ctx)
        right = lower_expr(node.right, ctx)
        return getattr(left, _BINOP_METHODS[node.op])(right)
    if isinstance(node, ir.UnOp):
        operand = lower_expr(node.operand, ctx)
        if node.op == "neg":
            return -operand
        if node.op == "not":
            return ~operand
        if node.op == "abs":
            return operand.abs()
    if isinstance(node, ir.FuncCall):
        if node.func in ("is_null", "not_null"):
            operand = lower_expr(node.args[0], ctx)
            expr = operand.is_null()
            return expr if node.func == "is_null" else ~expr
        if node.func in _MATH_METHODS:
            operand = lower_expr(node.args[0], ctx)
            try:
                return getattr(operand, _MATH_METHODS[node.func])()
            except AttributeError:
                raise _NotLowerable(f"'{node.func}' has no lazy expression form")
    if isinstance(node, ir.FillNull):
        operand = lower_expr(node.operand, ctx)
        fill = lower_expr(node.value, ctx)
        return operand.fill_null(fill)
    if isinstance(node, ir.Where):
        return (
            pl.when(lower_expr(node.cond, ctx))
            .then(lower_expr(node.a, ctx))
            .otherwise(lower_expr(node.b, ctx))
        )
    if isinstance(node, ir.Clip):
        operand = lower_expr(node.obj, ctx)
        lo = _const_scalar(node.lo)
        hi = _const_scalar(node.hi)
        try:
            return operand.clip(lo, hi)
        except Exception as exc:
            raise _NotLowerable(f"clip is not lazily plannable: {exc}")
    if isinstance(node, ir.Over):
        operand = lower_expr(node.operand, ctx)
        key = _group_key(node.by) if node.by is not None else None
        if key is None:
            raise _NotLowerable("over() needs a field key for lazy planning")
        try:
            return getattr(operand, node.op)().over(key)
        except AttributeError:
            raise _NotLowerable(f"over('{node.op}') is not lazily plannable")
    if isinstance(node, ir.Rolling):
        operand = lower_expr(node.operand, ctx)
        fn = {"sum": "rolling_sum", "mean": "rolling_mean",
              "min": "rolling_min", "max": "rolling_max"}[node.op]
        try:
            return getattr(operand, fn)(
                window_size=node.window,
                min_samples=node.min_periods or node.window,
            )
        except AttributeError:
            raise _NotLowerable(f"rolling('{node.op}') is not lazily plannable")
    raise _NotLowerable(f"'{type(node).__name__}' is not a lazy row expression")


def _lower_const(value) -> pl.Expr:
    import sympy as _sp

    if value is None or isinstance(value, (bool, int, float, str)):
        return pl.lit(value)
    if isinstance(value, _sp.Basic):
        return pl.lit(_sym.to_numeric(value))
    raise _NotLowerable(f"constant {value!r} is not a lazy expression")


def _const_scalar(node: ir.Expr):
    if isinstance(node, ir.Const):
        value = node.value
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
    raise _NotLowerable("clip bounds must be scalar literals for lazy planning")


def collect_lowered(root: ir.Expr):
    """Try to execute ``root`` as one Polars lazy query.

    Returns ``(value, kind, plan_text)`` or raises ``_NotLowerable``.
    """
    ctx = _LowerContext()
    if isinstance(root, ir.Reduce):
        lf = lower_frame(root, ctx)
        plan_text = lf.explain()
        frame = lf.collect()
        if ctx.used_scan:
            _io._bump()
        if (
            root.by is None
            and isinstance(root.operand, ir.FieldAccess)
            and frame.width == 1
            and frame.height == 1
        ):
            return frame.item(), "scalar", plan_text
        return frame, "frame", plan_text
    if isinstance(root, ir.FieldAccess):
        lf = lower_frame(root.obj, ctx).select(root.field)
        plan_text = lf.explain()
        frame = lf.collect()
        if ctx.used_scan:
            _io._bump()
        return frame.to_series(), "series", plan_text
    lf = lower_frame(root, ctx)
    plan_text = lf.explain()
    frame = lf.collect()
    if ctx.used_scan:
        _io._bump()
    if (
        isinstance(root, ir.GetItem)
        and isinstance(root.index, ir.Const)
        and isinstance(root.index.value, str)
    ):
        return frame.to_series(), "series", plan_text
    return frame, "frame", plan_text


def try_sink(root: ir.Expr, path: str, format: str) -> tuple[bool, str]:
    """Stream a lowerable pipeline straight to ``path``.

    Returns ``(True, plan)`` when the whole pipeline streamed without
    Python-side materialization, else ``(False, reason)``.
    """
    ctx = _LowerContext()
    try:
        if isinstance(root, ir.FieldAccess):
            lf = lower_frame(root.obj, ctx).select(root.field)
        else:
            lf = lower_frame(root, ctx)
    except _NotLowerable as exc:
        return False, exc.reason
    except (TLTypeError, Exception) as exc:
        return False, str(exc)
    try:
        if format == "parquet":
            lf.sink_parquet(path)
        else:
            lf.sink_csv(path)
        if ctx.used_scan:
            _io._bump()
        return True, lf.explain()
    except Exception as exc:
        return False, f"sink failed: {exc}"


def explain_plan(root: ir.Expr) -> tuple[str, bool]:
    """Return ``(physical_plan_text, lowered)`` without executing."""
    ctx = _LowerContext()
    try:
        if isinstance(root, ir.FieldAccess):
            lf = lower_frame(root.obj, ctx).select(root.field)
        else:
            lf = lower_frame(root, ctx)
        text = (
            "physical plan: polars-lazy (one optimized query; predicate and "
            "projection pushdown delegated to Polars)\n" + lf.explain()
        )
        return text, True
    except _NotLowerable as exc:
        return f"physical plan: materialized fallback ({exc.reason})", False
    except (TLTypeError, Exception) as exc:
        return f"physical plan: materialized fallback ({exc})", False
