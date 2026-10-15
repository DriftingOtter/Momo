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

# Remaining-use counts for liveness eviction (v0.6 M1). When present,
# a cache entry is dropped as soon as its last counted consumer reads
# it, so shared intermediates are computed once but not retained past
# use. Miscounts are safe by construction: eviction only ever causes
# recomputation, never a wrong value (evaluation is pure for memoized
# node types). Nodes absent from the table default to single use.
_eval_uses: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_eval_uses", default=None)

# Structural keys per node id (v0.6 M1): canonical content hashes that
# identify duplicate subtrees even when they are distinct objects (e.g.
# copies produced by per-composition rebinding). Post-substitution,
# duplicates share identical leaf objects, so id-keyed leaves unify
# copies soundly. Keys must be EXACT: equal keys imply identical
# evaluation. Unkeyable nodes map to None (id-only legacy path).
_eval_keys: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_eval_keys", default=None)

# Value cache by structural key, with remaining-use counts. Bounded by
# the number of distinct structures; freed at trace reset.
_eval_keycache: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_eval_keycache", default=None)
_eval_keyuses: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_eval_keyuses", default=None)
# Canonical node per structural key (legacy aliasing): the first node
# evaluated for a key; later identical subtrees alias its values.
_eval_keyowner: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_eval_keyowner", default=None)


def _alias_subtree(skipped: ir.Expr, canon: ir.Expr,
                   cache: dict) -> tuple | None:
    """Record canonical values under skipped-subtree ids (lockstep).

    Precondition: both roots share one structural key (hence identical
    shape). Returns the root value, or None to fall back to normal
    evaluation (missing canonical entry or structural surprise — never
    a wrong value).
    """
    from .planner import _children

    def _expr_kids(n: ir.Expr) -> list:
        return [c for c in _children(n) if isinstance(c, ir.Expr)]

    if cache.get(id(skipped)) is not None:
        return cache[id(skipped)][1]
    stack = [(skipped, canon)]
    seen: set[int] = set()
    while stack:
        s, c = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        if cache.get(id(s)) is not None:
            continue
        if type(s) is not type(c):
            return None
        centry = cache.get(id(c))
        if centry is None:
            return None
        cache[id(s)] = (s, centry[1])
        s_kids, c_kids = _expr_kids(s), _expr_kids(c)
        if len(s_kids) != len(c_kids):
            return None
        stack.extend(zip(s_kids, c_kids))
    done = cache.get(id(skipped))
    return done[1] if done is not None else None

# Node types that bypass memoization: nested AD control nodes run forward
# / backward passes with tape state (re-evaluation must not be skipped),
# and scans re-materialize on every visit (preserves existing
# scan-execution counts and profile honesty). DiffVar leaves stay
# memoized: forward tapes read them back by id. Miscounts remain safe
# (eviction only ever causes recomputation).
_NO_MEMO = (ir.AdjointOf, ir.ForwardOf, ir.Scan)


def _freeze(value):
    """Hashable structural freeze; unfreezable payloads fall back to
    identity (same object unifies, distinct objects never do)."""
    if value is None:
        return ("none",)
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, slice):
        return ("slice", _freeze(value.start), _freeze(value.stop),
                _freeze(value.step))
    if isinstance(value, (tuple, list)):
        return ("seq",) + tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return ("dict",) + tuple(sorted(
            ((_freeze(k), _freeze(v)) for k, v in value.items()),
            key=repr))
    if isinstance(value, np.ndarray):
        return ("array", id(value))
    if isinstance(value, np.generic):
        return ("scalar", str(value.dtype), value.item())
    return ("id", id(value))


def _structure_keys(root: ir.Expr) -> dict[int, object]:
    """Canonical content key per reachable node id (None = not unifiable).

    Keys are exact: equal keys guarantee identical evaluation. Leaves
    key by object identity, which unifies post-substitution duplicates
    (they share leaf objects) without ever merging distinct content.
    """
    from .planner import _children

    memo: dict[int, object] = {}

    def key_of(n: ir.Expr):
        # Taint rule: an unkeyable child (AdjointOf/Scan/unknown, keyed
        # None) poisons the whole parent key to None. Without this,
        # siblings differing only under an opaque child (e.g. ``0.5 *
        # Adj_k`` across loop steps) share one key and the first value
        # silently wins for all of them. Legitimate None *attributes*
        # (Norm ord, one-sided bounds) are frozen as ("none",) by
        # _freeze, so a bare None inside a key always means tainted.
        key = id(n)
        if key in memo:
            return memo[key]
        out = _node_key(n)
        if _tainted(out):
            out = None
        memo[key] = out
        return out

    def _tainted(key) -> bool:
        if key is None:
            return True
        if isinstance(key, tuple):
            return any(_tainted(k) for k in key)
        return False

    def kids(n: ir.Expr):
        # None poisons: an unkeyable child (AdjointOf/Scan/unknown) makes
        # the parent unkeyable too. Without this, siblings differing only
        # under an opaque child (e.g. ``0.5 * Adj_k`` across loop steps)
        # share one key and the first value wins — silent wrong results.
        out = tuple(key_of(c) for c in _children(n)
                    if isinstance(c, ir.Expr))
        return None if any(k is None for k in out) else out

    def _node_key(n: ir.Expr):
        if isinstance(n, ir.BinOp):
            return ("BinOp", n.op, kids(n))
        if isinstance(n, ir.UnOp):
            return ("UnOp", n.op, kids(n))
        if isinstance(n, ir.FuncCall):
            return ("FuncCall", n.func, kids(n))
        if isinstance(n, ir.FillNull):
            return ("FillNull", kids(n))
        if isinstance(n, ir.FieldAccess):
            return ("FieldAccess", n.field, kids(n))
        if isinstance(n, ir.GetItem):
            index = n.index
            index_key = (key_of(index) if isinstance(index, ir.Expr)
                         else _freeze(index))
            obj_keys = tuple(key_of(c) for c in _children(n)
                             if isinstance(c, ir.Expr) and c is not index)
            return ("GetItem", index_key, obj_keys)
        if isinstance(n, ir.Transpose):
            return ("Transpose", kids(n))
        if isinstance(n, ir.Reshape):
            return ("Reshape", tuple(n.shape), kids(n))
        if isinstance(n, ir.Flatten):
            return ("Flatten", kids(n))
        if isinstance(n, ir.Concat):
            return ("Concat", n.axis, kids(n))
        if isinstance(n, ir.Stack):
            return ("Stack", n.axis, kids(n))
        if isinstance(n, ir.Reduce):
            return ("Reduce", n.op, kids(n))
        if isinstance(n, ir.WithField):
            return ("WithField", n.name, kids(n))
        if isinstance(n, ir.ToDense):
            return ("ToDense", tuple(n.columns), n.nulls, kids(n))
        if isinstance(n, ir.Join):
            return ("Join", _freeze(n.on), n.how, n.suffix, kids(n))
        if isinstance(n, ir.SortBy):
            return ("SortBy", _freeze(n.keys), n.descending, kids(n))
        if isinstance(n, ir.Unique):
            return ("Unique", kids(n))
        if isinstance(n, ir.Where):
            return ("Where", kids(n))
        if isinstance(n, ir.Clip):
            return ("Clip", kids(n))
        if isinstance(n, ir.Over):
            return ("Over", n.op, kids(n))
        if isinstance(n, ir.Rolling):
            return ("Rolling", n.window, n.op, n.min_periods, kids(n))
        if isinstance(n, ir.Scan):
            return None
        if isinstance(n, ir.Chunked):
            return ("Chunked", n.path)
        if isinstance(n, ir.FittedTransform):
            return ("FittedTransform", n.kind, _freeze(n.config),
                     id(n.state), kids(n))
        if isinstance(n, (ir.SvdPart, ir.EighPart)):
            return (type(n).__name__, n.index, id(n.shared))
        if isinstance(n, ir.Solve):
            return ("Solve", kids(n))
        if isinstance(n, ir.Norm):
            return ("Norm", _freeze(n.ord), _freeze(n.axis), kids(n))
        if isinstance(n, (ir.Cov, ir.Corr)):
            return (type(n).__name__, n.rowvar, kids(n))
        if isinstance(n, ir.Quantile):
            return ("Quantile", _freeze(n.q), kids(n))
        if isinstance(n, ir.Einsum):
            return ("Einsum", _freeze(n.spec), kids(n))
        if isinstance(n, ir.Tensordot):
            return ("Tensordot", _freeze(n.axes), kids(n))
        if isinstance(n, ir.MoveAxis):
            return ("MoveAxis", _freeze(n.source),
                     _freeze(n.destination), kids(n))
        if isinstance(n, ir.OneHot):
            return ("OneHot", _freeze(n.categories), kids(n))
        if isinstance(n, ir.Polynomial):
            return ("Polynomial", n.degree, n.include_bias, kids(n))
        if isinstance(n, ir.Literal):
            return ("Literal", id(n.value), n.kind)
        if isinstance(n, ir.Const):
            return ("Const", _freeze(n.value))
        if isinstance(n, ir.Param):
            return ("Param", n.owner, n.name)
        if isinstance(n, ir.DiffVar):
            return ("DiffVar", n.owner, n.name, kids(n))
        if isinstance(n, ir.ApplyOp):
            return ("ApplyOp", id(n.operator), kids(n))
        if isinstance(n, (ir.AdjointOf, ir.ForwardOf)):
            return None
        return None

    key_of(root)
    return memo


def _count_uses(root: ir.Expr) -> dict[int, int]:
    """Parent-edge counts per node id over the reachable graph."""
    from .planner import _children

    counts = {id(root): 1}
    seen = {id(root)}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in _children(node):
            if not isinstance(child, ir.Expr):
                continue
            counts[id(child)] = counts.get(id(child), 0) + 1
            if id(child) not in seen:
                seen.add(id(child))
                stack.append(child)
    return counts


def evaluate(node: ir.Expr) -> tuple[Any, str]:
    """Evaluate a closed expression node (no unbound ``Param`` leaves).

    With no active trace, a scoped memo is established for the call, so
    shared subgraphs evaluate once and entries are released at last use.
    Nested calls share the enclosing trace. :class:`AdjointOf`,
    :class:`ForwardOf`, and :class:`Scan` nodes always evaluate directly
    (tape state and scan counts preserved).
    """
    if isinstance(node, _NO_MEMO) or _eval_cache.get() is not None:
        return _evaluate_memoized(node)
    cache: dict = {}
    uses = _count_uses(node)
    keys = _structure_keys(node)
    keyuses: dict = {}
    for nid, key in keys.items():
        if key is not None:
            keyuses[key] = keyuses.get(key, 0) + uses.get(nid, 0)
    token_cache = _eval_cache.set(cache)
    token_uses = _eval_uses.set(uses)
    token_keys = _eval_keys.set(keys)
    token_keycache = _eval_keycache.set({})
    token_keyuses = _eval_keyuses.set(keyuses)
    token_keyowner = _eval_keyowner.set(None)
    try:
        return _evaluate(node)
    finally:
        _eval_keyowner.reset(token_keyowner)
        _eval_keyuses.reset(token_keyuses)
        _eval_keycache.reset(token_keycache)
        _eval_keys.reset(token_keys)
        _eval_uses.reset(token_uses)
        _eval_cache.reset(token_cache)


def _evaluate_memoized(node: ir.Expr) -> tuple[Any, str]:
    """Cache-aware evaluation within an active trace (or plain fallback).

    Two-level lookup: object identity first, then the structural key
    (deduplicates copied-but-identical subtrees). Liveness eviction
    applies to both levels; every path degrades to recomputation, so
    key or count imperfections cost performance, never correctness.
    """
    cache = _eval_cache.get()
    if cache is None or isinstance(node, _NO_MEMO):
        return _evaluate(node)
    key = id(node)
    uses = _eval_uses.get()
    keys = _eval_keys.get()
    skey = keys.get(key) if keys is not None else None
    if uses is None:
        # Legacy keep-forever mode (AD forward tape): identity dedupe
        # plus structural aliasing. A structural hit skips evaluation
        # but must still record per-node entries: reverse() reads EVERY
        # node in its postorder, so the skipped subtree is walked in
        # lockstep with the canonical one and each id aliases the
        # canonical value (arrays shared by reference; reverse rules
        # never mutate taped values). Any mismatch aborts to normal
        # evaluation — aliasing only ever skips recomputation.
        hit = cache.get(key)
        if hit is not None:
            return hit[1]
        if skey is not None:
            owners = _eval_keyowner.get()
            canon = owners.get(skey) if owners is not None else None
            if canon is not None:
                aliased = _alias_subtree(node, canon, cache)
                if aliased is not None:
                    return aliased
        out = _evaluate(node)
        cache[key] = (node, out)
        if skey is not None:
            owners = _eval_keyowner.get()
            if owners is not None and skey not in owners:
                owners[skey] = node
        return out
    remaining = uses.get(key, 1)
    if remaining <= 0:
        return _evaluate(node)  # defensive: evicted or uncounted reuse
    uses[key] = remaining - 1
    hit = cache.get(key)
    if hit is not None:
        if remaining <= 1:
            del cache[key]  # last counted consumer served: release now
        _consume_key(skey)
        return hit[1]
    if skey is not None:
        keycache = _eval_keycache.get()
        khit = keycache.get(skey) if keycache is not None else None
        if khit is not None:
            _consume_key(skey)
            if remaining > 1:
                cache[key] = (node, khit)
            return khit
    out = _evaluate(node)
    _consume_key(skey)
    if remaining > 1:
        cache[key] = (node, out)
    if skey is not None:
        _store_key(skey, out)
    return out


def _consume_key(skey) -> None:
    """Record one structural consumption; evict at zero remaining uses."""
    if skey is None:
        return
    keycache = _eval_keycache.get()
    keyuses = _eval_keyuses.get()
    if keycache is None or keyuses is None:
        return
    left = keyuses.get(skey, 1) - 1
    keyuses[skey] = left
    if left <= 0:
        keycache.pop(skey, None)


def _store_key(skey, out) -> None:
    """Retain a freshly computed value under its structural key when
    further identical subtrees remain unconsumed."""
    if skey is None:
        return
    keycache = _eval_keycache.get()
    keyuses = _eval_keyuses.get()
    if keycache is None or keyuses is None:
        return
    if keyuses.get(skey, 0) > 0 and skey not in keycache:
        keycache[skey] = out


def trace_evaluate(root: ir.Expr, release: bool = True):
    """Evaluate with memoization; return (result, {id: (node, result)}).

    ``release=True`` (default) drops entries at last use, bounding peak
    retention. ``release=False`` keeps the legacy keep-forever tape for
    consumers (notably AD ``forward()``) that re-read values after the
    pass completes.
    """
    cache: dict = {}
    token_cache = _eval_cache.set(cache)
    tokens = [token_cache]
    if release:
        keys = _structure_keys(root)
        uses = _count_uses(root)
        keyuses: dict = {}
        for nid, key in keys.items():
            if key is not None:
                keyuses[key] = keyuses.get(key, 0) + uses.get(nid, 0)
        tokens.append(_eval_uses.set(uses))
        tokens.append(_eval_keys.set(keys))
        tokens.append(_eval_keycache.set({}))
        tokens.append(_eval_keyuses.set(keyuses))
        tokens.append(_eval_keyowner.set(None))
    else:
        # AD forward tape: structural keys serve aliasing only (the tape
        # stays keep-forever); uses stay None so nothing is ever evicted.
        tokens.append(_eval_uses.set(None))
        tokens.append(_eval_keys.set(_structure_keys(root)))
        tokens.append(_eval_keycache.set(None))
        tokens.append(_eval_keyuses.set(None))
        tokens.append(_eval_keyowner.set({}))
    try:
        result = evaluate(root)
    finally:
        for var, token in reversed(list(zip(
                (_eval_cache, _eval_uses, _eval_keys, _eval_keycache,
                 _eval_keyuses, _eval_keyowner), tokens))):
            var.reset(token)
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
            hint = ""
            if node.field in ("T", "transpose", "reshape", "flatten",
                              "where", "clip", "sort", "moveaxis", "stack",
                              "concat"):
                func = "transpose" if node.field == "T" else node.field
                hint = (f"; did you mean the {func} operator "
                        f"(e.g. {func}(X))?")
            raise TLTypeError(
                f"cannot access field '{node.field}' on a {kind} value; "
                f"named fields exist only on structured tensors{hint}"
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
        return _stats.apply(node.op, operand, kind, node.by, evaluate,
                            node.axis, node.keepdims)
    if isinstance(node, ir.WithField):
        lowered = _try_lowered(node)
        if lowered is not None:
            return lowered
        obj, kind = evaluate(node.obj)
        value, _ = evaluate(node.value)
        return _with_field_fallback(obj, kind, node.name, value)
    if isinstance(node, ir.ToDense):
        from ..operators import bridge as _bridge

        return _bridge.evaluate(node, evaluate)
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
