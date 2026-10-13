"""Execution: evaluate closed IR graphs to materialized values.

``evaluate(root)`` returns ``(value, kind)`` with kind in
``dense | frame | series | scalar``.  ``collect_value`` converts a
materialized value to the public ``collect()`` contract:

- dense ``ndarray`` (0-d arrays become Python scalars),
- structured :class:`polars.DataFrame` / :class:`polars.Series`,
- exact symbolic values (SymPy) and plain scalars as-is.
"""
from __future__ import annotations

import contextvars
from typing import Any

import numpy as np
import polars as pl

from ..core import expression as ir
from ..core.errors import TLTypeError
from ..operators import arithmetic as _arith
from ..operators import data_science as _ds
from ..operators import elementary as _math
from ..operators import indexing as _indexing
from ..operators import linear_algebra as _linalg
from ..operators import relational as _rel
from ..operators import statistics as _stats
from ..operators import transform as _tf
from . import polars_plan as _lower

# Implicit per-evaluation memo. Off by default (existing paths are
# untouched); the autodiff tracer enables it so shared subgraphs evaluate
# exactly once. Entries keep (node, result) pairs so live nodes pin their
# ids and recycled ids can never alias a dead entry.
_eval_cache: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_eval_cache", default=None)


def evaluate(node: ir.Expr) -> tuple[Any, str]:
    """Evaluate a closed expression node (no unbound ``Param`` leaves)."""
    cache = _eval_cache.get()
    if cache is None:
        return _evaluate(node)
    key = id(node)
    hit = cache.get(key)
    if hit is not None:
        return hit[1]
    out = _evaluate(node)
    cache[key] = (node, out)
    return out


def trace_evaluate(root: ir.Expr):
    """Evaluate with memoization; return (result, {id: (node, result)})."""
    cache: dict = {}
    token = _eval_cache.set(cache)
    try:
        result = evaluate(root)
    finally:
        _eval_cache.reset(token)
    return result, cache


def _try_lowered(node: ir.Expr):
    """Best-effort single-query execution; ``None`` when not lowerable."""
    try:
        value, kind, _ = _lower.collect_lowered(node)
        return value, kind
    except Exception:
        # Graceful degradation: any lowering failure (unsupported shape or
        # internal error) falls back to the node-by-node reference path,
        # which re-validates with descriptive errors.
        return None


def _evaluate(node: ir.Expr) -> tuple[Any, str]:
    """Evaluate a closed expression node (no unbound ``Param`` leaves)."""
    if isinstance(node, ir.Literal):
        return node.value, node.kind
    if isinstance(node, ir.Const):
        return _const_value(node.value)
    if isinstance(node, ir.Param):
        raise TLTypeError(
            f"unbound operator parameter '{node.name}': apply the operator "
            f"to data before execution"
        )
    if isinstance(node, ir.BinOp):
        left, _ = evaluate(node.left)
        right, _ = evaluate(node.right)
        return _arith.apply(node.op, left, right)
    if isinstance(node, ir.UnOp):
        operand, _ = evaluate(node.operand)
        return _arith.apply_unary(node.op, operand)
    if isinstance(node, ir.FuncCall):
        args = [evaluate(a)[0] for a in node.args]
        if node.func in ("sin", "cos", "tan", "exp", "log", "sqrt"):
            return _math.apply(node.func, args[0])
        return _ds.apply(node.func, args[0], _kind(args[0]))
    if isinstance(node, ir.FillNull):
        operand, kind = evaluate(node.operand)
        fill, _ = evaluate(node.value)
        return _ds.apply_fill(operand, kind, fill)
    if isinstance(node, ir.FieldAccess):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        obj, kind = evaluate(node.obj)
        if kind != "frame":
            if kind == "series":
                raise TLTypeError(
                    f"cannot access field '{node.field}' on a single field series"
                )
            raise TLTypeError(
                f"cannot access field '{node.field}' on a {kind} value; "
                f"named fields exist only on structured tensors"
            )
        from ..backends import polars_backend as _pl_be

        return _pl_be.field_access(obj, node.field), "series"
    if isinstance(node, ir.GetItem):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        obj, kind = evaluate(node.obj)
        return _indexing.apply_index(obj, kind, node.index, evaluate)
    if isinstance(node, ir.Transpose):
        obj, kind = evaluate(node.obj)
        return _linalg.apply_transpose(obj, kind)
    if isinstance(node, ir.Reshape):
        obj, kind = evaluate(node.obj)
        return _linalg.apply_reshape(obj, kind, node.shape)
    if isinstance(node, ir.Flatten):
        obj, kind = evaluate(node.obj)
        return _linalg.apply_flatten(obj, kind)
    if isinstance(node, ir.Concat):
        parts = [evaluate(p) for p in node.parts]
        return _linalg.apply_concat(parts, node.axis)
    if isinstance(node, ir.Stack):
        parts = [evaluate(p) for p in node.parts]
        return _linalg.apply_stack(parts, node.axis)
    if isinstance(node, ir.Reduce):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        operand, kind = evaluate(node.operand)
        return _stats.apply(node.op, operand, kind, node.by, evaluate)
    if isinstance(node, ir.WithField):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        obj, kind = evaluate(node.obj)
        value, _ = evaluate(node.value)
        return _with_field_fallback(obj, kind, node.name, value)
    if isinstance(node, ir.Join):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        left, lkind = evaluate(node.left)
        right, rkind = evaluate(node.right)
        return _rel.evaluate_join(left, lkind, right, rkind,
                                  node.on, node.how, node.suffix)
    if isinstance(node, ir.SortBy):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        obj, kind = evaluate(node.obj)
        return _rel.evaluate_sort(obj, kind, node.keys, node.descending)
    if isinstance(node, ir.Unique):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        obj, kind = evaluate(node.obj)
        return _rel.evaluate_unique(obj, kind)
    if isinstance(node, ir.Where):
        cond, ckind = evaluate(node.cond)
        a, akind = evaluate(node.a)
        b, bkind = evaluate(node.b)
        return _tf.evaluate_where(cond, ckind, a, akind, b, bkind)
    if isinstance(node, ir.Clip):
        obj, kind = evaluate(node.obj)
        lo, _ = evaluate(node.lo)
        hi, _ = evaluate(node.hi)
        return _tf.evaluate_clip(obj, kind, lo, hi)
    if isinstance(node, ir.Over):
        return _tf.evaluate_over(node, evaluate)
    if isinstance(node, ir.Rolling):
        obj, kind = evaluate(node.operand)
        return _tf.evaluate_rolling(obj, kind, node.window, node.op,
                                    node.min_periods)
    if isinstance(node, ir.Scan):
        from . import io as _io

        return _io.materialize_scan(node.spec), "frame"
    if isinstance(node, ir.Chunked):
        # Never silently materialize persistent storage inside a larger
        # graph: chunked roots execute via chunk_plan (dask) from explicit
        # boundaries (collect/write/profile), or not at all.
        from ..core.errors import TLUnsupportedError

        raise TLUnsupportedError(
            "chunked data reached node-by-node evaluation, which would "
            "materialize persistent storage; only chunk-lowerable pipelines "
            "(elementwise/shape/reduction ops) execute out-of-core in v0.5 "
            "M1 — collect() explicitly to materialize, or restructure",
            code="TL1001",
            details={"path": node.path},
        )
    if isinstance(node, ir.FittedTransform):
        from ..operators import ml as _ml

        return _ml.apply_fitted(node)
    if isinstance(node, ir.SvdPart):
        from ..operators import linalg as _la

        return _la.evaluate_svd_part(node.shared, node.index)
    if isinstance(node, ir.EighPart):
        from ..operators import linalg as _la

        return _la.evaluate_eigh_part(node.shared, node.index)
    if isinstance(node, ir.Solve):
        from ..operators import linalg as _la

        a, _ = evaluate(node.a)
        b, _ = evaluate(node.b)
        return _la.evaluate_solve(a, b)
    if isinstance(node, ir.Norm):
        from ..operators import linalg as _la

        value, kind = evaluate(node.obj)
        return _la.evaluate_norm(value, kind, node.ord, node.axis)
    if isinstance(node, ir.Cov):
        value, kind = evaluate(node.obj)
        return _stats.evaluate_cov(value, kind, node.rowvar)
    if isinstance(node, ir.Corr):
        value, kind = evaluate(node.obj)
        return _stats.evaluate_corr(value, kind, node.rowvar)
    if isinstance(node, ir.Quantile):
        value, kind = evaluate(node.obj)
        return _stats.evaluate_quantile(value, kind, node.q)
    if isinstance(node, ir.Einsum):
        from ..operators import tensor_algebra as _ta

        return _ta.evaluate_einsum(node.spec, [evaluate(p) for p in node.parts])
    if isinstance(node, ir.Tensordot):
        from ..operators import tensor_algebra as _ta

        a, akind = evaluate(node.a)
        b, bkind = evaluate(node.b)
        return _ta.evaluate_tensordot(a, akind, b, bkind, node.axes)
    if isinstance(node, ir.MoveAxis):
        from ..operators import tensor_algebra as _ta

        value, kind = evaluate(node.obj)
        return _ta.evaluate_moveaxis(value, kind, node.source, node.destination)
    if isinstance(node, ir.OneHot):
        from ..operators import features as _fx

        value, kind = evaluate(node.obj)
        return _fx.evaluate_onehot(value, kind, node.categories)
    if isinstance(node, ir.Polynomial):
        from ..operators import features as _fx

        value, kind = evaluate(node.obj)
        return _fx.evaluate_polynomial(value, kind, node.degree,
                                       node.include_bias)
    if isinstance(node, ir.DiffVar):
        from ..operators import autodiff as _ad

        return _ad.evaluate_diffvar(node)
    if isinstance(node, ir.AdjointOf):
        from ..operators import autodiff as _ad

        return _ad.evaluate_adjoint(node)
    if isinstance(node, ir.ForwardOf):
        from ..operators import autodiff as _ad

        return _ad.evaluate_forward(node)
    if isinstance(node, ir.ApplyOp):
        return _evaluate_apply(node)
    raise TLTypeError(  # pragma: no cover - defensive
        f"cannot evaluate '{type(node).__name__}'"
    )


def _evaluate_apply(node: ir.ApplyOp):
    op = node.operator
    arg_values = [evaluate(a) for a in node.args]
    if len(arg_values) != len(op.params):
        raise TLTypeError(
            f"operator expects {len(op.params)} argument(s), "
            f"got {len(arg_values)}"
        )
    env: dict[tuple[str, str], ir.Expr] = {}
    for name, (value, kind) in zip(op.params, arg_values):
        env[(op.uid, name)] = _literal_of(value, kind)
    return evaluate(ir.substitute(op.root, env))


def _with_field_fallback(obj, kind: str, name: str, value):
    from ..core.tensor import Tensor

    if isinstance(value, Tensor):
        value = value._material()
    if kind == "frame":
        from ..backends import polars_backend as _pl_be

        return _pl_be.set_field(obj, name, value), "frame"
    if kind == "series":
        raise TLTypeError(
            f"cannot assign field '{name}' on a single field series"
        )
    raise TLTypeError(
        f"cannot assign field '{name}' on {kind} data; named fields exist "
        f"only on structured tensors"
    )


def _literal_of(value: Any, kind: str) -> ir.Literal:
    if kind == "dense":
        from ..core.tensor import _readonly_view

        return ir.Literal(_readonly_view(np.asarray(value)), "dense")
    if kind == "frame":
        return ir.Literal(value.clone(), "frame")
    if kind == "series":
        return ir.Literal(value.clone(), "series")
    return ir.Literal(value, "scalar")


def _const_value(value):
    if isinstance(value, slice):
        return value, "scalar"
    if isinstance(value, np.ndarray):
        return value, "dense"
    import sympy as _sp

    if isinstance(value, _sp.Basic):
        from ..backends import sympy_backend as _sym

        return _sym.to_numeric(value), "scalar"
    return value, "scalar"


def _kind(value) -> str:
    if isinstance(value, np.ndarray):
        return "dense"
    if isinstance(value, pl.DataFrame):
        return "frame"
    if isinstance(value, pl.Series):
        return "series"
    return "scalar"


def collect_value(value, kind: str):
    """Convert a materialized value to the public ``collect()`` result."""
    if kind == "dense":
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr
    if kind == "scalar":
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            return arr.item() if arr.ndim == 0 else arr
        return value
    return value
