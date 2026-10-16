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
from ..core.errors import TLShapeError, TLTypeError
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
    if isinstance(node, ir.BinOp) and node.op in ("and", "or"):
        return _lower_logical(node, ctx)
    raise _NotLowerable(f"'{type(node).__name__}' has no lazy-frame lowering")


def _struct_eq(a: object, b: object) -> bool:
    """Safe structural equality for IR row-domain comparison.

    Never raises on opaque values (arrays, frames): those unify by
    identity only, so independently constructed snapshots are treated
    as distinct row domains (fail-closed).
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, ir.Const):
        va, vb = a.value, b.value  # type: ignore[union-attr]
        if va is vb:
            return True
        if va is None or isinstance(va, (bool, int, float, str)):
            return type(va) is type(vb) and va == vb
        return False
    if isinstance(a, ir.Literal):
        return a is b or (a.value is b.value  # type: ignore[union-attr]
                           and a.kind == b.kind)  # type: ignore[union-attr]
    if isinstance(a, ir.Expr):
        import dataclasses as _dc

        fa = _dc.fields(a)
        fb = _dc.fields(b)  # type: ignore[arg-type]
        if [f.name for f in fa] != [f.name for f in fb]:
            return False
        try:
            return all(_struct_eq(getattr(a, f.name), getattr(b, f.name))
                       for f in fa)
        except ValueError:
            return False
    if isinstance(a, tuple):
        b = b  # type: ignore[assignment]
        return len(a) == len(b) and all(_struct_eq(x, y) for x, y in zip(a, b))  # type: ignore[arg-type]
    try:
        return bool(a == b)  # type: ignore[comparison-overlap]
    except ValueError:
        return False


def _row_roots(node: ir.Expr) -> list[ir.Expr] | None:
    """Frame-root IRs a boolean expression reads, or None if unanalyzable.

    Roots identify the logical row domain: two masks may only combine
    when they resolve to exactly one structurally equal root.  Anything
    outside boolean row expressions over plannable frame shapes returns
    None (the caller decides between historical fallback and rejection).
    Scalar reductions contribute no domain.
    """
    if isinstance(node, ir.Const):
        return []
    if isinstance(node, ir.FieldAccess):
        return [node.obj]
    if isinstance(node, ir.GetItem):
        if (isinstance(node.index, ir.Const)
                and isinstance(node.index.value, str)):
            return [node.obj]
        return None
    if isinstance(node, ir.BinOp):
        left = _row_roots(node.left)
        right = _row_roots(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ir.UnOp):
        return _row_roots(node.operand)
    if isinstance(node, ir.FuncCall) and node.func in ("is_null", "not_null"):
        if len(node.args) != 1:
            return None
        return _row_roots(node.args[0])
    if isinstance(node, ir.Reduce):
        return [] if node.by is None else [node]
    if isinstance(node, (ir.Scan, ir.Join, ir.SortBy, ir.Unique,
                         ir.WithField)):
        return [node]
    if isinstance(node, ir.Literal):
        if node.kind in ("dense", "scalar"):
            return []
        return [node]
    return None


def _touches_chunked(node: object) -> bool:
    if isinstance(node, ir.Chunked):
        return True
    if isinstance(node, ir.Literal):
        return node.kind == "chunked"
    if isinstance(node, ir.Expr):
        import dataclasses as _dc

        try:
            fields = _dc.fields(node)
        except TypeError:
            return False
        return any(_touches_chunked(getattr(node, f.name)) for f in fields)
    if isinstance(node, tuple):
        return any(_touches_chunked(x) for x in node)
    return False


def _touches_structured(node: object) -> bool:
    """Whether an expression reads structured (row-domain) data."""
    if isinstance(node, ir.Const):
        return False
    if isinstance(node, ir.Literal):
        return node.kind in ("frame", "series", "chunked")
    if isinstance(node, (ir.Scan, ir.Join, ir.SortBy, ir.Unique,
                         ir.WithField, ir.Reduce, ir.Chunked,
                         ir.FieldAccess, ir.GetItem)):
        return True
    if isinstance(node, ir.Expr):
        import dataclasses as _dc

        try:
            fields = _dc.fields(node)
        except TypeError:
            return False
        return any(_touches_structured(getattr(node, f.name))
                   for f in fields)
    if isinstance(node, tuple):
        return any(_touches_structured(x) for x in node)
    return False


def _first_field(node: ir.Expr) -> str | None:
    """Leftmost field name in a boolean expression, if any."""
    if isinstance(node, ir.FieldAccess):
        return node.field
    if isinstance(node, ir.GetItem):
        if (isinstance(node.index, ir.Const)
                and isinstance(node.index.value, str)):
            return node.index.value
        return None
    if isinstance(node, ir.BinOp):
        return _first_field(node.left) or _first_field(node.right)
    if isinstance(node, ir.UnOp):
        return _first_field(node.operand)
    if isinstance(node, ir.FuncCall) and node.args:
        return _first_field(node.args[0])
    return None


def _lower_logical(node: ir.BinOp, ctx: _LowerContext) -> pl.LazyFrame:
    """Fuse `and`/`or` over one row domain into a single query.

    Both sides must resolve to exactly one structurally equal frame
    root; otherwise the masks have unproven positional correspondence
    and combining them is a hard `TL0512` error rather than a silent
    positional zip.  Chunked inputs keep the historical chunked path;
    pure dense inputs keep the historical positional path.
    """
    if _touches_chunked(node):
        raise _NotLowerable(f"'{node.op}' over chunked data keeps chunked lowering")
    left_roots = _row_roots(node.left)
    right_roots = _row_roots(node.right)
    if left_roots is None or right_roots is None:
        if _touches_structured(node):
            raise TLShapeError(
                f"cannot combine Boolean results with unproven row domains "
                f"using '{node.op}': equal length is not proof of equal row "
                f"identity; filter sequentially or derive both predicates "
                f"from one source",
                code="TL0512",
            )
        raise _NotLowerable(
            f"'{node.op}' has operands outside one plannable row domain")
    roots: list[ir.Expr] = []
    for root in left_roots + right_roots:
        if not any(_struct_eq(root, seen) for seen in roots):
            roots.append(root)
    if not roots:
        # Degenerate scalar combination: no row domain involved.
        raise _NotLowerable(f"'{node.op}' has no row domain to fuse")
    if len(roots) != 1:
        raise TLShapeError(
            f"cannot combine Boolean results from {len(roots)} different "
            f"row domains with '{node.op}': equal length is not proof of "
            f"equal row identity; filter sequentially or derive both "
            f"predicates from one source",
            code="TL0512",
        )
    base = lower_frame(roots[0], ctx)
    left = lower_expr(node.left, ctx)
    right = lower_expr(node.right, ctx)
    combined = left & right if node.op == "and" else left | right
    name = _first_field(node.left) or _first_field(node.right) or "mask"
    return base.select(combined.alias(name))


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
        if node.op == "count" and not isinstance(node.operand, ir.FieldAccess):
            # Whole-frame row count (all rows, nulls included).
            return base.select(pl.len().alias("count"))
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
    if node.op == "count" and not isinstance(node.operand, ir.FieldAccess):
        # Group sizes (all rows per group). Column collision guarded
        # like the eager path. Series operands keep per-field counts
        # through the generic path below.
        if key == "count":
            raise TLTypeError(
                "grouped count needs a group key not named 'count' "
                "(it collides with the count output column); rename "
                "the key first")
        return base.group_by(key).agg(pl.len().alias("count")).sort(key)
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
    if isinstance(node, ir.GetItem):
        if (isinstance(node.index, ir.Const)
                and isinstance(node.index.value, str)):
            lower_frame(node.obj, ctx)  # validates plannability of the source
            return pl.col(node.index.value)
        raise _NotLowerable("'GetItem' is not a lazy row expression")
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
        isinstance(root, ir.BinOp)
        and root.op in ("and", "or")
        and frame.width == 1
        and frame.dtypes[0] == pl.Boolean
    ):
        # Fused single-predicate lowering produces one Boolean column;
        # match the historical single-Series result contract.
        return frame.to_series(), "series", plan_text
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
