"""Chunk-aware execution planning: lower dense subgraphs into one dask
graph while reasoning about physical cost.

Logical computation and physical execution stay separate: every operator
keeps its mathematical meaning while the planner checks chunk
compatibility, inserts explicit union-subdivision rechunks, estimates
movement and graph growth, and rejects what cannot run boundedly — all
inspectable through the recorded plan. No silent full materialization.

Capability classes: chunk-local, tree-reducible, alignment-dependent,
rechunk-dependent, neighborhood, global. Anything outside the rule set
raises ``_NotChunkable`` (surfaced as TL1001/TL1010-1013 with reasons).
"""
from __future__ import annotations

import asyncio
import contextlib
import math
from dataclasses import dataclass, field

import numpy as np

from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError, TLUnsupportedError

# -- planning policy (conservative heuristics, not guarantees) -------------------
# These bound what the planner admits. Chunk sizes bound typical working
# sets, but scheduler concurrency means several blocks can be live at once.
# MEMORY_BUDGET_BYTES=None disables budget enforcement (legacy behavior);
# MAX_WORKERS=None leaves Dask's default thread pool. Unknown resource
# requirements are never treated as zero: unmeasurable graphs are rejected
# (TL1021) when a budget is configured instead of being admitted for free.
# Union subdivision only ever splits chunks (never merges), so alignment
# itself is always structurally safe; the gates below bound the resulting
# graph size and block bytes instead.

MAX_TASKS = 200_000       # admitted dask graph size
MAX_BLOCK_BYTES = 256 * 1024**2  # admitted max chunk bytes
MEMORY_BUDGET_BYTES = None  # working-memory budget; None = unenforced
MAX_WORKERS = None        # worker cap; None = Dask default


def set_chunk_policy(*, max_tasks=None, max_block_bytes=None,
                     memory_budget_bytes=None, max_workers=None,
                     union_growth=None) -> dict:
    """Override planning limits (tests/power users); returns the old ones.

    ``memory_budget_bytes`` gates admission on modeled working memory
    (see :func:`decide`); ``max_workers`` caps dask worker threads used by
    chunked ``collect()``/``write_zarr()``. ``union_growth`` is accepted
    and ignored (retired M2 heuristic; union subdivision is always safe).
    """
    global MAX_TASKS, MAX_BLOCK_BYTES, MEMORY_BUDGET_BYTES, MAX_WORKERS
    old = {"max_tasks": MAX_TASKS, "max_block_bytes": MAX_BLOCK_BYTES,
           "memory_budget_bytes": MEMORY_BUDGET_BYTES,
           "max_workers": MAX_WORKERS}
    if max_tasks is not None:
        globals()["MAX_TASKS"] = max_tasks
    if max_block_bytes is not None:
        globals()["MAX_BLOCK_BYTES"] = max_block_bytes
    if memory_budget_bytes is not None:
        globals()["MEMORY_BUDGET_BYTES"] = memory_budget_bytes
    if max_workers is not None:
        if (not isinstance(max_workers, int) or isinstance(max_workers, bool)
                or max_workers < 1):
            from ..core.errors import TLUnsupportedError as _TLU

            raise _TLU(
                f"max_workers must be a positive integer, got {max_workers!r}",
                code="TL1023",
                details={"max_workers": max_workers})
        globals()["MAX_WORKERS"] = max_workers
    return old


def clear_chunk_policy() -> None:
    """Restore unenforced defaults (tests and REPL reset)."""
    set_chunk_policy(max_tasks=200_000, max_block_bytes=256 * 1024**2)
    global MEMORY_BUDGET_BYTES, MAX_WORKERS
    MEMORY_BUDGET_BYTES, MAX_WORKERS = None, None


def current_policy() -> dict:
    """Snapshot of the active chunk policy (no reads; safe for abstract())."""
    import os as _os

    return {"memory_budget_bytes": MEMORY_BUDGET_BYTES,
            "max_workers": MAX_WORKERS,
            "cpu_count": _os.cpu_count() or 4,
            "max_tasks": MAX_TASKS,
            "max_block_bytes": MAX_BLOCK_BYTES}


class _NotChunkable(Exception):
    def __init__(self, reason: str, code: str = "TL1001",
                 details: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.details = details or {}


def _unsupported(op: str, requirement: str = "a later milestone",
                 details: dict | None = None) -> _NotChunkable:
    return _NotChunkable(
        f"'{op}' is not chunk-lowerable in v0.5 "
        f"(needs {requirement}); supported: elementwise math, comparisons, "
        f"where/clip, shape ops, concat/stack, slices, whole-array "
        f"reductions, bounded matmul", code="TL1001",
        details={"operation": op, **(details or {})})


def _global(op: str, requirement: str) -> _NotChunkable:
    return _NotChunkable(
        f"'{op}' needs {requirement} across partitions: deferred; "
        f"restructure outside chunked graphs", code="TL1013",
        details={"operation": op, "requirement": requirement})


_MATH_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "exp": "exp", "log": "log", "sqrt": "sqrt",
}

_BINOPS = {
    "add": "__add__", "sub": "__sub__", "mul": "__mul__",
    "div": "__truediv__", "pow": "__pow__", "mod": "__mod__",
    "eq": "__eq__", "ne": "__ne__", "gt": "__gt__", "lt": "__lt__",
    "ge": "__ge__", "le": "__le__", "and": "__and__", "or": "__or__",
}

_REDUCTIONS = {"sum": "sum", "mean": "mean", "min": "min", "max": "max",
               "std": "std", "var": "var"}


# -- plan record -------------------------------------------------------------------


@dataclass
class ChunkPlan:
    steps: list = field(default_factory=list)
    rechunks: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    estimates: dict = field(default_factory=dict)
    backend: str = "dask-chunked"

    def step(self, capability: str, op: str, detail: str) -> None:
        self.steps.append({"op": op, "class": capability, "detail": detail})

    def rechunk(self, reason: str, before, after, movement_bytes: int) -> None:
        self.rechunks.append({"reason": reason, "before": before,
                              "after": after, "movement_bytes": movement_bytes})

    def render(self, graph) -> str:
        from ..backends import dask_backend as _be

        try:
            shape = list(graph.shape)
            chunks = [list(c) for c in graph.chunks]
            parts = int(getattr(graph, "npartitions", -1))
            tasks = _be.task_count(graph)
        except Exception:  # pragma: no cover - defensive
            return "dask plan (details unavailable)"
        lines = ["physical plan: dask-chunked (one lazy graph; chunk-local "
                 "execution keeps peak memory near chunk size)"]
        if self.inputs:
            lines.append("  inputs:")
            for spec in self.inputs:
                lines.append(
                    f"    scan_zarr({spec['path']}): shape {spec['shape']}, "
                    f"chunks {spec['chunks']}, {spec['partitions']} partitions")
        for step in self.steps:
            lines.append(f"  [{step['class']}] {step['op']}: {step['detail']}")
        for rechunk in self.rechunks:
            lines.append(
                f"  [rechunk] {rechunk['reason']}: {rechunk['before']} -> "
                f"{rechunk['after']} (~{rechunk['movement_bytes'] / 1e6:.1f}MB "
                f"re-read upper bound)")
        lines.append(f"  output: shape {shape}, chunks {chunks}, "
                     f"{parts} partitions, {tasks} tasks")
        lines.append("  estimates (planning heuristics, not guarantees): "
                     f"max block {self.estimates.get('max_block_mb', '?')}MB, "
                     f"tasks {tasks}")
        outcome = self.estimates.get("outcome")
        if outcome is not None:
            budget = self.estimates.get("budget_mb")
            budget_text = f"{budget}MB" if budget is not None else "unenforced"
            est = self.estimates.get("estimated_peak_mb")
            est_text = f"{est}MB" if est is not None else "n/a (unenforced)"
            lines.append(
                f"  resources [{self.estimates.get('purpose', 'collect')}]: "
                f"{outcome}; peak task "
                f"{self.estimates.get('peak_task_mb', '?')}MB "
                f"({self.estimates.get('peak_task_detail', '?')}); "
                f"workers {self.estimates.get('workers_used')} "
                f"(requested {self.estimates.get('workers_requested')}); "
                f"estimated peak {est_text} vs budget {budget_text}")
            if outcome == "rejected" and self.estimates.get("reason"):
                lines.append(f"  admission: [{self.estimates.get('code')}] "
                             f"{self.estimates.get('reason')}")
            lines.append("  (estimates cover modeled chunk buffers + output + "
                         "graph overhead only; total process RSS is not "
                         "guaranteed — see LANGUAGE_SPEC §16)")
        return "\n".join(lines)


# Cross-run source baselines (v0.5 M4): path -> fingerprint from the most
# recent planning touch of that path (bounded; no content, metadata only).
# verify_sources() compares fresh reality against the planned spec
# (catches stale metadata caches) and against the previous baseline
# (catches content-only changes between runs). A TL1030 resynchronizes
# both, so retrying after a reported change establishes the new baseline.
_SOURCE_BASELINES: dict[str, dict] = {}
_SOURCE_BASELINE_LIMIT = 256


class _ChunkContext:
    def __init__(self) -> None:
        self.sources: dict[str, object] = {}
        self.used_source = False
        self.plan = ChunkPlan()
        # Modeled per-task working bytes (max over steps; see §3).
        self.peak_task_bytes = 0
        self.peak_detail = ""
        # Source fingerprints captured at plan time (M4 consistency).
        self.fingerprints: dict[str, dict] = {}
        # Previous-run baselines for the same paths (may be absent).
        self.prior_baselines: dict[str, dict | None] = {}

    def source(self, path: str):
        from ..backends import dask_backend as _be

        if path not in self.sources:
            spec = _be.describe(path)
            self.sources[path] = _be.from_spec(spec)
            self.plan.inputs.append({
                "path": spec.path, "shape": list(spec.shape),
                "chunks": [list(c) for c in _chunks_of(self.sources[path])],
                "partitions": spec.npartitions})
            # Planned baseline: the (possibly cached) spec the graph is
            # built on, plus a fresh stamp. verify_sources() compares
            # fresh reality against THIS — cache staleness across runs and
            # mid-run modification both surface as TL1030.
            try:
                stamp = _be.fingerprint(path)
            except Exception:
                stamp = {"path": path, "unavailable": True}
            self.fingerprints[path] = {"spec": spec, "stamp": stamp}
            self.prior_baselines[path] = _SOURCE_BASELINES.get(path)
            if stamp.get("unavailable") is not True:
                if (len(_SOURCE_BASELINES) >= _SOURCE_BASELINE_LIMIT
                        and path not in _SOURCE_BASELINES):
                    _SOURCE_BASELINES.pop(next(iter(_SOURCE_BASELINES)))
                _SOURCE_BASELINES[path] = stamp
        self.used_source = True
        return self.sources[path]


# Test-only failure injection (v0.5 M4). Never set outside tests: keys are
# read on the write path only. The controlled interface is
# :func:`injected_failures` (context manager); direct mutation is
# reserved for the test-suite fixture. Production execution never sets
# these, and no public API accepts hook arguments (see
# test_hooks_inaccessible_through_public_api).
_TEST_HOOKS = {"fail_before_write": False,
               "fail_after_partitions": None,
               "fail_validation": False,
               "fail_publication": False,
               "raise_interrupt_at": None,
               "slow_partition_secs": 0.0}


@contextlib.contextmanager
def injected_failures(**overrides):
    """Arm deterministic write-path faults within a ``with`` block.

    Internal test interface (not exported at package top level). Unknown
    keys raise immediately so typos cannot silently arm nothing.
    """
    unknown = set(overrides) - set(_TEST_HOOKS)
    if unknown:
        raise KeyError(f"unknown test hooks: {sorted(unknown)}")
    saved = {key: _TEST_HOOKS[key] for key in overrides}
    _TEST_HOOKS.update(overrides)
    try:
        yield
    finally:
        _TEST_HOOKS.update(saved)


def verify_sources(ctx: _ChunkContext) -> None:
    """Re-check sources against the planned baseline (metadata only).

    Compares fresh store reality with the spec the graph was lowered on.
    Shape/dtype/chunk-count changes are always caught (including via a
    stale metadata cache from an earlier run); content-only changes are
    caught via the filesystem stamp on a best-effort basis. Raises TL1030.
    """
    from ..backends import dask_backend as _be

    for path, baseline in ctx.fingerprints.items():
        spec, stamp = baseline.get("spec"), baseline.get("stamp", {})
        if stamp.get("unavailable"):
            continue
        try:
            fresh = _be.fingerprint(path)
        except Exception as exc:
            raise _NotChunkable(
                f"source '{path}' became unreadable after planning: {exc}",
                code="TL1030", details={"path": path})
        planned = {"shape": tuple(spec.shape), "dtype": str(spec.dtype),
                   "chunks": tuple(spec.chunks),
                   "nchunks": int(spec.npartitions)}
        if any(int(d) == 0 for d in planned["shape"]):
            # Degenerate census: zarr reports 0 chunks for empty dims while
            # the spec clamps to 1; both describe the same empty store.
            planned["nchunks"] = 0
            if any(int(d) == 0 for d in fresh.get("shape", ())):
                fresh["nchunks"] = 0
        for key, old_value in planned.items():
            new_value = fresh.get(key)
            if key == "chunks":
                new_value = tuple(new_value) if new_value else None
                old_value = tuple(old_value) if old_value else None
            if _norm_nchunks(key, new_value, fresh) != _norm_nchunks(
                    key, old_value, planned):
                _resync_source(path, fresh)
                raise _NotChunkable(
                    f"source '{path}' changed after planning ({key}: "
                    f"{old_value} -> {new_value}); the admitted plan no "
                    f"longer describes it",
                    code="TL1030",
                    details={"path": path, "field": key,
                             "planned": str(old_value),
                             "observed": str(new_value)})
        old_stamp, new_stamp = stamp.get("mtime_ns"), fresh.get("mtime_ns")
        if (old_stamp != new_stamp
                and -1 not in (old_stamp, new_stamp)
                and old_stamp is not None and new_stamp is not None):
            _resync_source(path, fresh)
            raise _NotChunkable(
                f"source '{path}' was modified after planning "
                f"(filesystem stamp changed); content is not guaranteed",
                code="TL1030",
                details={"path": path, "field": "mtime"})
        prior = ctx.prior_baselines.get(path)
        if prior and not prior.get("unavailable"):
            for key in ("shape", "dtype", "chunks", "nchunks"):
                old_value = prior.get(key)
                new_value = fresh.get(key)
                if key == "chunks":
                    new_value = tuple(new_value) if new_value else None
                    old_value = tuple(old_value) if old_value else None
                if _norm_nchunks(key, new_value, fresh) != _norm_nchunks(
                        key, old_value, prior):
                    _resync_source(path, fresh)
                    raise _NotChunkable(
                        f"source '{path}' changed since its previous "
                        f"baseline ({key}: {old_value} -> {new_value}); "
                        f"re-run to establish the new baseline",
                        code="TL1030",
                        details={"path": path, "field": key,
                                 "planned": str(old_value),
                                 "observed": str(new_value)})
            old_stamp, new_stamp = prior.get("mtime_ns"), fresh.get("mtime_ns")
            if (old_stamp != new_stamp
                    and -1 not in (old_stamp, new_stamp)
                    and old_stamp is not None and new_stamp is not None):
                _resync_source(path, fresh)
                raise _NotChunkable(
                    f"source '{path}' was modified since its previous "
                    f"baseline (filesystem stamp changed); content is not "
                    f"guaranteed — re-run to establish the new baseline",
                    code="TL1030",
                    details={"path": path, "field": "mtime"})


def _norm_nchunks(key: str, value, scope: dict):
    if key == "nchunks" and any(int(d) == 0 for d in scope.get("shape", ())):
        return 0
    return value


def _resync_source(path: str, fresh: dict) -> None:
    """Report-once semantics: a detected change resynchronizes the baseline
    and metadata cache, so retrying proceeds on the new reality."""
    _SOURCE_BASELINES[path] = fresh
    from ..backends import dask_backend as _be

    try:
        _be.describe.cache_clear()
    except Exception:  # pragma: no cover - defensive
        pass


def _chunks_of(graph) -> tuple:
    try:
        return tuple(tuple(int(c) for c in axis) for axis in graph.chunks)
    except Exception:  # pragma: no cover - defensive
        return ()


def _block_bytes(graph) -> int:
    import numpy as _np

    try:
        per_axis = [max(c) for c in graph.chunks]
        total = 1
        for dim in per_axis:
            total *= dim
        return total * _np.dtype(graph.dtype).itemsize
    except Exception:  # pragma: no cover - defensive
        return -1


def _operand_bytes(graph) -> int:
    import numpy as _np

    try:
        total = 1
        for dim in graph.shape:
            total *= dim
        return total * _np.dtype(graph.dtype).itemsize
    except Exception:  # pragma: no cover - defensive
        return -1


def _itemsize(graph) -> int:
    import numpy as _np

    try:
        return int(_np.dtype(graph.dtype).itemsize)
    except Exception:  # pragma: no cover - defensive
        return -1


def _track(ctx: _ChunkContext, working: int, detail: str) -> None:
    """Record a per-task working-set estimate (max wins; never zeroed)."""
    if working is None or working < 0:
        return
    if working > ctx.peak_task_bytes:
        ctx.peak_task_bytes = working
        ctx.peak_detail = detail


def contains_chunked(root: ir.Expr) -> bool:
    """Conservative walk: True if chunked storage appears anywhere,
    including inside embedded operator applications."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ir.Chunked):
            return True
        if isinstance(node, ir.ApplyOp):
            stack.extend(node.args)
            stack.append(node.operator.root)
        else:
            from ..runtime.planner import _children

            stack.extend(c for c in _children(node)
                         if isinstance(c, ir.Expr))
    return False


# -- lowering ------------------------------------------------------------------------


def lower(node: ir.Expr, ctx: _ChunkContext, env: dict) -> object:
    """Lower a node to a dask collection; ``env`` binds (uid, name) params."""
    import dask.array as da

    from ..backends import dask_backend as _be

    if isinstance(node, ir.Chunked):
        return ctx.source(node.path)
    if isinstance(node, ir.Param):
        if (node.owner, node.name) in env:
            return env[(node.owner, node.name)]
        raise _NotChunkable(f"unbound parameter '{node.name}'")
    if isinstance(node, ir.DiffVar):
        raise _NotChunkable("differentiation over chunked data is deferred",
                            code="TL1001",
                            details={"operation": "grad"})
    if isinstance(node, ir.Literal):
        if node.kind == "dense":
            return _be.from_array(np.asarray(node.value), chunks="auto")
        if node.kind == "series":
            return _be.from_array(np.asarray(node.value.to_list()),
                                  chunks="auto")
        raise _unsupported(f"{node.kind} literal")
    if isinstance(node, ir.Const):
        value = node.value
        if isinstance(value, slice):
            raise _unsupported("slice as value")
        if isinstance(value, (str, bytes)):
            raise TLTypeError(
                f"chunked arithmetic needs numeric constants, got "
                f"{value!r}", details={"constant": str(value)[:32]})
        if value is None:
            raise _unsupported("null constant")
        try:
            return _be.from_array(np.asarray(value), chunks=-1)
        except (TypeError, ValueError):
            raise _unsupported("constant")
    if isinstance(node, ir.BinOp):
        if node.op == "matmul":
            return _lower_matmul(node, ctx, env)
        left = lower(node.left, ctx, env)
        right = lower(node.right, ctx, env)
        (left, right), moved = _align_all(f"binary '{node.op}'",
                                          [left, right], ctx)
        ctx.plan.step("alignment", f"binary '{node.op}'",
                      "rechunked to union subdivision" if moved else
                      _layout_note(left, right))
        _track(ctx, _block_bytes(left) + _block_bytes(right)
               + max(_block_bytes(left), _block_bytes(right)),
               f"binary '{node.op}' (two inputs + output per task)")
        try:
            return getattr(left, _BINOPS[node.op])(right)
        except (KeyError, TypeError, ValueError) as exc:
            raise _NotChunkable(f"binary '{node.op}' failed: {exc}")
    if isinstance(node, ir.UnOp):
        operand = lower(node.operand, ctx, env)
        if node.op == "neg":
            ctx.plan.step("chunk-local", "neg", "sign flip, chunks preserved")
            _track(ctx, 2 * _block_bytes(operand), "neg (input + output)")
            return -operand
        if node.op == "abs":
            ctx.plan.step("chunk-local", "abs", "chunks preserved")
            _track(ctx, 2 * _block_bytes(operand), "abs (input + output)")
            return abs(operand)
        raise _unsupported("logical not")
    if isinstance(node, ir.FuncCall):
        if node.func not in _MATH_FUNCS and node.func not in ("is_null", "not_null"):
            raise _unsupported(node.func)
        operand = lower(node.args[0], ctx, env)
        ctx.plan.step("chunk-local", node.func, "chunks preserved")
        _track(ctx, 2 * _block_bytes(operand),
               f"'{node.func}' (input + output per task)")
        if node.func == "is_null":
            return da.isnan(operand)
        if node.func == "not_null":
            return ~da.isnan(operand)
        try:
            return getattr(da, _MATH_FUNCS[node.func])(operand)
        except (TypeError, ValueError) as exc:
            raise _NotChunkable(f"'{node.func}' failed: {exc}")
    if isinstance(node, ir.FillNull):
        operand = lower(node.operand, ctx, env)
        fill = lower(node.value, ctx, env)
        operand, fill = _align_all("fill_null", [operand, fill], ctx)[0]
        ctx.plan.step("chunk-local", "fill_null", "masked replace")
        _track(ctx, _block_bytes(operand) + _block_bytes(fill)
               + max(_block_bytes(operand), _block_bytes(fill)),
               "fill_null (operand + fill + output per task)")
        return da.where(da.isnan(operand), fill, operand)
    if isinstance(node, ir.Where):
        cond = lower(node.cond, ctx, env)
        a = lower(node.a, ctx, env)
        b = lower(node.b, ctx, env)
        (cond, a, b), moved = _align_all("where", [cond, a, b], ctx)
        ctx.plan.step("chunk-local", "where", "masked select")
        _track(ctx, _block_bytes(cond) + _block_bytes(a) + _block_bytes(b)
               + max(_block_bytes(a), _block_bytes(b)),
               "where (cond + two branches + output per task)")
        return da.where(cond, a, b)
    if isinstance(node, ir.Clip):
        operand = lower(node.obj, ctx, env)
        lo = _bound(node.lo, ctx, env)
        hi = _bound(node.hi, ctx, env)
        ctx.plan.step("chunk-local", "clip", "chunks preserved")
        _track(ctx, 2 * _block_bytes(operand), "clip (input + output)")
        try:
            return da.clip(operand, lo, hi)
        except (TypeError, ValueError) as exc:
            raise _NotChunkable(f"clip failed: {exc}")
    if isinstance(node, ir.GetItem):
        return _lower_getitem(node, ctx, env)
    if isinstance(node, ir.Transpose):
        base = lower(node.obj, ctx, env)
        ctx.plan.step("chunk-local", "transpose",
                      f"metadata-only: {_chunks_of(base)} stays partitioned")
        _track(ctx, 2 * _block_bytes(base), "transpose (input + output)")
        return base.transpose()
    if isinstance(node, ir.MoveAxis):
        base = lower(node.obj, ctx, env)
        _check_axis(node.source, base, "moveaxis source")
        _check_axis(node.destination, base, "moveaxis destination")
        ctx.plan.step("chunk-local", "moveaxis",
                      f"metadata-only: {node.source} -> {node.destination}")
        _track(ctx, 2 * _block_bytes(base), "moveaxis (input + output)")
        return da.moveaxis(base, node.source, node.destination)
    if isinstance(node, ir.Reshape):
        return _lower_reshape(node, ctx, env)
    if isinstance(node, ir.Flatten):
        base = lower(node.obj, ctx, env)
        ctx.plan.step("rechunk", "flatten", "may merge chunks along one axis")
        _track(ctx, 2 * _block_bytes(base), "flatten (input + output)")
        return _admit(base.ravel(), ctx, "flatten")
    if isinstance(node, ir.Concat):
        return _lower_concat(node, ctx, env)
    if isinstance(node, ir.Stack):
        parts = [lower(p, ctx, env) for p in node.parts]
        _check_shapes_match(parts, "stack")
        parts, _ = _align_all("stack", parts, ctx)
        ctx.plan.step("alignment", "stack", f"new axis {node.axis}")
        _track(ctx, 2 * sum(_block_bytes(p) for p in parts),
               "stack (inputs + stacked output per task)")
        try:
            return _admit(da.stack(parts, axis=node.axis), ctx, "stack")
        except ValueError as exc:
            raise _NotChunkable(f"stack failed: {exc}")
    if isinstance(node, ir.Reduce):
        if node.by is not None:
            if node.axis is not None:
                raise _NotChunkable(
                    f"reduction '{node.op}' cannot combine grouped 'by' "
                    f"with 'axis'",
                    code="TL1001",
                    details={"operation": node.op})
            raise _global("grouped reduction", "grouped aggregation")
        from ..core.expression import resolve_axes as _resolve_axes

        operand = lower(node.operand, ctx, env)
        shape = tuple(operand.shape)
        axes = _resolve_axes(node.axis, len(shape), node.op, shape)
        if node.op == "count":
            # Non-null census via self-equality (NaN != NaN excludes only
            # NaN; uniform across dtypes, tree-reducible, narrow bool
            # temporaries per block).
            counted = (operand == operand)
            if axes:
                out = counted.sum(axis=list(axes),
                                  keepdims=bool(node.keepdims))
            else:
                out = counted.astype("int64")
            ctx.plan.step("tree-reducible", "count",
                          f"non-null census over axes {list(axes)} "
                          f"keepdims={bool(node.keepdims)}: "
                          f"{_nparts(operand)} partitions")
            _track(ctx, _block_bytes(operand) + _block_bytes(out),
                   "reduction 'count' (input + census blocks)")
            return _admit(out, ctx, "count")
        try:
            out = getattr(operand, _REDUCTIONS[node.op])(
                axis=list(axes), keepdims=bool(node.keepdims))
        except (AttributeError, ValueError) as exc:
            raise _NotChunkable(f"reduction '{node.op}' failed: {exc}")
        if axes:
            detail = (f"axes {list(axes)} keepdims={bool(node.keepdims)}: "
                      f"{_nparts(operand)} partitions -> "
                      f"{_chunks_of(out)} (tree combine)")
        else:
            detail = (f"{_nparts(operand)} partitions -> scalar "
                      f"(tree combine)")
        ctx.plan.step("tree-reducible", node.op, detail)
        if axes:
            _track(ctx, _block_bytes(operand) + _block_bytes(out)
                   + _reduced_partial_bytes(operand, axes, out),
                   f"reduction '{node.op}' (input + output blocks + "
                   f"axis partials)")
        else:
            _track(ctx, _block_bytes(operand)
                   + max(_nparts(operand), 0) * max(_itemsize(operand), 0) * 4,
                   f"reduction '{node.op}' (input block + partial buffers)")
        return out
    if isinstance(node, ir.ApplyOp):
        return _lower_apply(node, ctx, env)
    if isinstance(node, ir.FittedTransform):
        raise _unsupported(f"estimator '{node.kind}'",
                           "fitting on chunked data is deferred")
    if isinstance(node, ir.SortBy):
        raise _global("sort", "a global ordering")
    if isinstance(node, ir.Unique):
        raise _global("unique", "global deduplication")
    if isinstance(node, ir.Join):
        raise _global("join", "a global shuffle")
    if isinstance(node, ir.Over):
        raise _global("grouped transform", "grouped aggregation")
    if isinstance(node, ir.Rolling):
        raise _global("rolling", "overlap exchange between chunks")
    raise _unsupported(type(node).__name__)


def _nparts(graph) -> int:
    try:
        return int(graph.npartitions)
    except Exception:  # pragma: no cover - defensive
        return -1


def _layout_note(left, right) -> str:
    return (f"layouts {_chunks_of(left)} vs {_chunks_of(right)}: "
            f"compatible, no movement")


# -- shapes (mathematical validity, separate from chunk feasibility) ------------------


def _check_shapes_match(parts: list, op: str) -> None:
    if not parts:
        raise TLUnsupportedError(f"{op} needs inputs", code="TL1001",
                                 details={"operation": op})
    ref = list(parts[0].shape)
    for part in parts[1:]:
        if list(part.shape) != ref:
            raise TLShapeError(
                f"{op} needs matching shapes, got {ref} and {list(part.shape)}",
                code="TL0599",
                details={"shapes": [ref, list(part.shape)]},
            )


def _check_axis(axes, graph, desc: str) -> None:
    ndim = len(graph.shape)
    flat = list(axes) if isinstance(axes, tuple) else [axes]
    for axis in flat:
        if not isinstance(axis, int) or not -ndim <= axis < ndim:
            raise TLShapeError(
                f"{desc} axis {axis!r} out of range for rank {ndim}",
                code="TL0599",
                details={"axis": axis, "rank": ndim},
            )


# -- alignment: explicit union subdivision ----------------------------------------------
# Union boundaries are a superset of both inputs' boundaries, so every
# union chunk sits inside one chunk of each input: sides only ever SPLIT
# (local, no cross-partition shuffle). Growth beyond the policy limit, or
# unknown (nan) dimensions, reject instead.


def _boundaries(chunks_1d: tuple) -> tuple[set, int]:
    total = 0
    bounds = {0}
    for size in chunks_1d:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise _NotChunkable("unknown chunk dimensions",
                                code="TL1010",
                                details={"chunks": str(tuple(chunks_1d))})
        if size < 0:
            raise _NotChunkable("negative chunk dimension",
                                code="TL1010",
                                details={"chunks": str(tuple(chunks_1d))})
        total += int(size)
        bounds.add(total)
    return bounds, total


def _union_1d(a: tuple, b: tuple) -> tuple:
    bounds_a, total_a = _boundaries(a)
    bounds_b, total_b = _boundaries(b)
    if total_a != total_b:
        raise _NotChunkable("chunk totals differ along an axis",
                            code="TL1010",
                            details={"chunks": [str(a), str(b)]})
    merged = sorted(bounds_a | bounds_b)
    return tuple(merged[i + 1] - merged[i] for i in range(len(merged) - 1))


def _align_all(desc: str, parts: list, ctx: _ChunkContext,
               skip_axes: set = frozenset()) -> tuple[list, bool]:
    """Align N collections to their finest common subdivision.

    Trailing axes follow NumPy broadcasting; size-1 and skipped axes need
    no movement. Returns ``(aligned, moved)``.
    """
    if len(parts) < 2:
        return parts, False
    shapes = [tuple(p.shape) for p in parts]
    masked = []
    for shape in shapes:
        masked.append(tuple(1 if ax in skip_axes and ax < len(shape) else dim
                            for ax, dim in enumerate(shape)))
    _check_broadcastable_shapes(masked, desc)
    n = max(len(s) for s in shapes)
    padded_shapes = [[1] * (n - len(s)) + list(s) for s in shapes]
    padded_chunks = [[(1,)] * (n - len(_chunks_of(p))) + [tuple(c) for c in _chunks_of(p)]
                     for p in parts]
    targets: list[tuple | None] = []
    for axis in range(n):
        dims = [shape[axis] for shape in padded_shapes]
        top = max(dims)
        if axis in skip_axes or top <= 1 or any(d not in (1, top) for d in dims):
            targets.append(None)  # broadcast/skipped axis: no movement
            continue
        cols = [padded_chunks[i][axis] for i in range(len(parts))
                if padded_shapes[i][axis] == top]
        union = cols[0]
        for other in cols[1:]:
            union = _union_1d(union, other)
        targets.append(union)
    out, moved = [], False
    for part in parts:
        real = _chunks_of(part)
        off = n - len(real)
        want = tuple(targets[off + axis] if targets[off + axis] is not None
                     else real[axis]
                     for axis in range(len(real)))
        if tuple(tuple(c) for c in want) == tuple(tuple(c) for c in real):
            out.append(part)
        else:
            out.append(_rechunk_to(part, want, ctx, f"{desc}"))
            moved = True
    return out, moved


def _check_broadcastable_shapes(shapes: list, desc: str) -> None:
    try:
        np.broadcast_shapes(*shapes)
    except ValueError:
        raise TLShapeError(
            f"cannot broadcast shapes {shapes} for operator '{desc}'",
            code="TL0501",
            details={"shapes": shapes, "op": desc},
        ) from None


def _rechunk_to(graph, chunks, ctx: _ChunkContext, reason: str):
    before = _chunks_of(graph)
    if tuple(tuple(c) for c in before) == tuple(tuple(c) for c in chunks):
        return graph
    movement = _operand_bytes(graph)
    ctx.plan.rechunk(reason, [list(c) for c in before],
                     [list(c) for c in chunks], movement)
    _track(ctx, _block_bytes(graph)
           + _block_bytes_of_chunks(chunks, graph),
           f"rechunk {reason} (input + output blocks live)")
    try:
        return graph.rechunk(chunks)
    except ValueError as exc:
        raise _NotChunkable(f"rechunk failed: {exc}", code="TL1011",
                            details={"reason": reason})


def _block_bytes_of_chunks(chunks, graph) -> int:
    import numpy as _np

    try:
        per_axis = [max(c) for c in chunks]
        total = 1
        for dim in per_axis:
            total *= dim
        return total * _np.dtype(graph.dtype).itemsize
    except Exception:  # pragma: no cover - defensive
        return -1


# -- shape-changing and contraction ops --------------------------------------------------


def _lower_getitem(node: ir.GetItem, ctx, env):
    base = lower(node.obj, ctx, env)
    index = node.index
    if isinstance(index, ir.Const):
        value = index.value
        if isinstance(value, (int, slice)) or (
                isinstance(value, tuple) and value
                and all(isinstance(v, (int, slice)) for v in value)):
            try:
                out = base[value]
            except (ValueError, TypeError, IndexError) as exc:
                raise _NotChunkable(f"indexing failed: {exc}")
            ctx.plan.step("chunk-local", "index",
                          f"subselect -> chunks {_chunks_of(out)}")
            _track(ctx, _block_bytes(base) + _block_bytes(out),
                   "index (input block + selected output)")
            return out
    raise _unsupported("mask/computed indexing",
                       "full materialization (repeated positions can explode "
                       "chunk sizes)")


def _lower_reshape(node: ir.Reshape, ctx, env):
    base = lower(node.obj, ctx, env)
    requested = list(node.shape)
    if requested.count(-1) > 1:
        raise TLShapeError(f"cannot reshape to {requested}: "
                           f"only one unknown dimension allowed",
                           code="TL0599", details={"to": requested})
    if -1 not in requested:
        try:
            old, new = int(np.prod(base.shape)), int(np.prod(requested))
        except (TypeError, ValueError):
            old = new = -1
        if old != -1 and old != new:
            raise TLShapeError(
                f"cannot reshape {list(base.shape)} to {requested}",
                code="TL0599",
                details={"from": list(base.shape), "to": requested},
            )
    try:
        out = base.reshape(tuple(requested))
    except (NotImplementedError, ValueError) as exc:
        raise _NotChunkable(
            f"reshape {list(base.shape)} -> {requested} would merge "
            f"chunks unsafely: {exc}",
            code="TL1010",
            details={"from": [list(base.shape), _chunks_of(base)],
                     "to": requested})
    ctx.plan.step("rechunk", "reshape",
                  f"{_chunks_of(base)} -> {_chunks_of(out)}")
    _track(ctx, _block_bytes(base) + _block_bytes(out),
           "reshape (input block + output block)")
    return _admit(out, ctx, "reshape")


def _lower_concat(node: ir.Concat, ctx, env):
    import dask.array as da

    parts = [lower(p, ctx, env) for p in node.parts]
    norm = node.axis % len(parts[0].shape) if parts[0].shape else 0
    _check_concat_shapes(parts, node.axis, norm)
    parts, moved = _align_all(f"concat axis {node.axis}", parts, ctx,
                                skip_axes={norm})
    if moved:
        ctx.plan.step("alignment", "concat",
                      "non-axis chunks rechunked to union subdivision")
    ctx.plan.step("alignment", "concat",
                  f"axis {node.axis}, {len(parts)} parts")
    _track(ctx, sum(_block_bytes(p) for p in parts)
           + max((_block_bytes(p) for p in parts), default=0),
           "concat (input blocks + output block per task)")
    try:
        return _admit(da.concatenate(parts, axis=node.axis), ctx, "concat")
    except ValueError as exc:
        raise _NotChunkable(f"concat failed: {exc}")


def _check_concat_shapes(parts: list, axis: int, norm: int) -> None:
    if not parts:
        raise TLUnsupportedError("concat needs inputs", code="TL1001",
                                 details={"operation": "concat"})
    ref = list(parts[0].shape)
    for part in parts[1:]:
        shape = list(part.shape)
        if len(shape) != len(ref) or any(
                d != r for i, (d, r) in enumerate(zip(shape, ref)) if i != norm):
            raise TLShapeError(
                f"concat needs matching shapes except on axis {axis}",
                code="TL0599",
                details={"shapes": [ref, shape]},
            )


def _lower_matmul(node: ir.BinOp, ctx, env):
    import dask.array as da

    left = lower(node.left, ctx, env)
    right = lower(node.right, ctx, env)
    a_shape, b_shape = tuple(left.shape), tuple(right.shape)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise _unsupported("N-D matmul", "rank-2 products only in v0.5")
    if a_shape[1] != b_shape[0]:
        raise TLShapeError(
            f"matmul requires (…, n, k) @ (…, k, m), got {a_shape} and {b_shape}",
            code="TL0502",
            details={"shapes": [list(a_shape), list(b_shape)]},
        )
    a_chunks = _chunks_of(left)
    b_chunks = _chunks_of(right)
    if len(a_chunks[1]) != 1 or len(b_chunks[0]) != 1:
        raise _NotChunkable(
            f"matmul contract dimension is partitioned "
            f"({list(a_chunks[1])} vs {list(b_chunks[0])}): bounded block "
            f"products need an unchunked contract axis",
            code="TL1010",
            details={"left_chunks": [list(c) for c in a_chunks],
                     "right_chunks": [list(c) for c in b_chunks],
                     "contract": [a_shape[1], b_shape[0]]})
    out = da.matmul(left, right)
    ctx.plan.step("alignment", "matmul",
                  f"contract {a_shape[1]} unchunked; "
                  f"output chunks {_chunks_of(out)}")
    _track(ctx, _matmul_working(left, right, a_shape, b_shape),
           "matmul (row panel + column panel + output block per task)")
    return _admit(out, ctx, "matmul")


def _reduced_partial_bytes(graph, axes: tuple, out) -> int:
    """Tree-combine buffers for an axis reduction: each output element
    combines one partial per input chunk along the reduced axes."""
    try:
        chunks = _chunks_of(graph)
        factor = 1
        for ax in axes:
            factor *= max(len(chunks[ax]), 1)
        return factor * _block_bytes(out)
    except Exception:  # pragma: no cover - defensive
        return -1


def _matmul_working(left, right, a_shape, b_shape) -> int:
    """Per-task bytes for one output block: its row panel of A, column
    panel of B, and the output block itself (contract axis unchunked)."""
    import numpy as _np

    try:
        item = int(_np.dtype(left.dtype).itemsize)
        rows = max(int(c) for c in _chunks_of(left)[0])
        cols = max(int(c) for c in _chunks_of(right)[1])
        k = int(a_shape[1])
        return (rows * k + k * cols + rows * cols) * item
    except Exception:  # pragma: no cover - defensive
        return -1


def _lower_apply(node: ir.ApplyOp, ctx, env):
    op = node.operator
    args = [lower(a, ctx, env) for a in node.args]
    if len(args) != len(op.params):
        raise _NotChunkable("nested operator arity mismatch")
    sub_env = dict(env)
    for name, graph in zip(op.params, args):
        sub_env[(op.uid, name)] = graph
    return lower(op.root, ctx, sub_env)


# -- admission gate ------------------------------------------------------------------------


def _admit(graph, ctx: _ChunkContext, op: str):
    """Measure a lowered graph against policy limits (no data read)."""
    from ..backends import dask_backend as _be

    if _has_unknown_chunks(graph):
        raise _NotChunkable(f"{op} produced unknown chunk dimensions",
                            code="TL1010",
                            details={"operation": op})
    tasks = _be.task_count(graph)
    if tasks < 0:
        raise _NotChunkable(f"{op}: cannot estimate graph size",
                            code="TL1021", details={"operation": op})
    if tasks > MAX_TASKS:
        raise _NotChunkable(
            f"{op} would build {tasks} tasks (limit {MAX_TASKS})",
            code="TL1012",
            details={"operation": op, "tasks": tasks, "limit": MAX_TASKS})
    block = _block_bytes(graph)
    if block > MAX_BLOCK_BYTES:
        raise _NotChunkable(
            f"{op} needs ~{block / 1e6:.1f}MB blocks (limit "
            f"{MAX_BLOCK_BYTES / 1e6:.0f}MB)",
            code="TL1011",
            details={"operation": op, "block_bytes": block,
                     "limit": MAX_BLOCK_BYTES})
    return graph


def _has_unknown_chunks(graph) -> bool:
    try:
        for axis in graph.chunks:
            for dim in axis:
                if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
                    return True
                if int(dim) < 0:
                    return True
    except Exception:  # pragma: no cover - defensive
        return True
    return False


def _bound(bound: ir.Expr, ctx, env):
    """A clip bound: None stays None (one-sided), else a scalar float."""
    if isinstance(bound, ir.Const) and bound.value is None:
        return None
    graph = lower(bound, ctx, env)
    try:
        value = graph.compute() if hasattr(graph, "compute") else graph
    except Exception as exc:
        raise _NotChunkable(f"clip bound failed: {exc}")
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1:
        raise _NotChunkable("clip bounds must be scalar")
    return float(arr.flat[0])

# -- resource model, admission, and monitored execution (v0.5 M3) -------------------
# Memory-budget contract (see LANGUAGE_SPEC §16):
#
# - The budget (MEMORY_BUDGET_BYTES) covers *estimated incremental working
#   memory*: per-task working sets x active workers + output materialization
#   (collect) or one output block (streaming writes) + graph overhead.
# - Included: input/output/temporary chunk buffers implied by chunk layouts
#   and dtypes, broadcast/rechunk/reduction/matmul intermediates as modeled
#   per step, and a linear task-graph overhead term.
# - NOT guaranteed: total process RSS (baseline interpreter/BLAS/thread-pool
#   allocations, allocator fragmentation, or concurrent native ops already
#   running — Dask's own worker limits are best-effort and its pause
#   mechanism cannot stop running tasks). Estimates are planning heuristics;
#   only the operating system can enforce hard process containment.
# - Unknown requirements are never zero: unmeasurable graphs are rejected
#   (TL1021) whenever a budget is configured.
#
# Outcomes: admitted | admitted-reduced-concurrency | rejected (TL1020 /
# TL1021 / TL1023 / TL1024). Concurrency reduction (workers -> 1) is the
# only automatic adaptation; the planner never silently rechunks data to
# fit a budget — an unplannable layout stays rejected.

GRAPH_OVERHEAD_PER_TASK = 256  # bytes: scheduler/graph metadata estimate
PRESSURE_FLOOR_BYTES = 64 * 1024**2  # RSS noise floor for pressure reports

# Last chunked execution record (estimates + observations for profile()).
LAST_DECISION: dict = {}
LAST_OBSERVED: dict = {}


def _effective_workers() -> int:
    import os as _os

    if MAX_WORKERS is not None:
        return int(MAX_WORKERS)
    try:
        return max(int(_os.cpu_count() or 4), 1)
    except Exception:  # pragma: no cover - defensive
        return 4


def _rss_peak_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import platform

        if platform.system() == "Darwin":  # bytes on macOS
            return rss / 1e6
        return rss / 1e3  # kilobytes on Linux
    except Exception:  # pragma: no cover - platform dependent
        return float("nan")


def _fmt_mb(nbytes) -> str:
    try:
        return f"{float(nbytes) / 1e6:.2f}MB"
    except Exception:  # pragma: no cover - defensive
        return "?"


def decide(graph, ctx: _ChunkContext, purpose: str = "collect") -> dict:
    """Pure resource decision (no data read). Raises _NotChunkable on
    rejection; otherwise returns the decision dict and records it."""
    from ..backends import dask_backend as _be

    tasks = _be.task_count(graph)
    if tasks < 0:
        raise _NotChunkable("cannot estimate graph size: resource "
                            "requirements unbounded",
                            code="TL1021",
                            details={"operation": purpose})
    block = _block_bytes(graph)
    peak_task = ctx.peak_task_bytes or block
    if peak_task is None or peak_task < 0 or block < 0:
        if MEMORY_BUDGET_BYTES is not None:
            raise _NotChunkable(
                "resource requirements cannot be bounded reliably",
                code="TL1021",
                details={"operation": purpose,
                         "peak_task": ctx.peak_detail or "unknown"})
        peak_task = max(peak_task or 0, 0)
    output_bytes = _operand_bytes(graph)
    overhead = max(tasks, 0) * GRAPH_OVERHEAD_PER_TASK
    workers_req = _effective_workers()
    budget = MEMORY_BUDGET_BYTES

    if purpose == "stream":
        sink_block = block
        est = peak_task * workers_req + sink_block + overhead
        est_1 = peak_task * 1 + sink_block + overhead
    else:
        out = max(output_bytes, 0)
        est = peak_task * workers_req + out + overhead
        est_1 = peak_task * 1 + out + overhead

    decision = {"purpose": purpose, "tasks": tasks,
                "peak_task_bytes": peak_task,
                "peak_task_detail": ctx.peak_detail or "final output block",
                "output_bytes": max(output_bytes, 0),
                "overhead_bytes": overhead,
                "workers_requested": workers_req, "workers_used": workers_req,
                "budget_bytes": budget,
                "estimated_peak_bytes": est if budget is not None else None,
                "outcome": "admitted", "uncertainty": "modeled"}
    if budget is None:
        return decision
    if purpose == "collect" and max(output_bytes, 0) > budget:
        decision.update(
            outcome="rejected", estimated_peak_bytes=est_1,
            workers_used=1)
        raise _NotChunkable(
            f"collect() would materialize {_fmt_mb(output_bytes)} "
            f"(budget {_fmt_mb(budget)}): stream with write_zarr() instead",
            code="TL1024",
            details={"operation": "collect", "output_bytes": output_bytes,
                     "budget_bytes": budget,
                     "suggestion": "write_zarr() streams outputs larger "
                                   "than the working-memory budget",
                     **{k: v for k, v in decision.items()}})
    if est <= budget:
        return decision
    if est_1 <= budget:
        decision.update(outcome="admitted-reduced-concurrency",
                        workers_used=1, estimated_peak_bytes=est_1)
        return decision
    if peak_task > budget:
        decision.update(outcome="rejected", estimated_peak_bytes=est_1,
                        workers_used=1)
        raise _NotChunkable(
            f"single-task working set {_fmt_mb(peak_task)} "
            f"({_fmt_mb(block)} blocks; {ctx.peak_detail}) exceeds budget "
            f"{_fmt_mb(budget)}: concurrency cannot help; rechunk the "
            f"store to smaller chunks",
            code="TL1023",
            details={"peak_task_bytes": peak_task, "block_bytes": block,
                     "budget_bytes": budget, "workers": workers_req,
                     "suggested_max_block_bytes": max(budget // 3, 1),
                     **{k: v for k, v in decision.items()}})
    decision.update(outcome="rejected", estimated_peak_bytes=est_1,
                    workers_used=1)
    raise _NotChunkable(
        f"modeled peak {_fmt_mb(est_1)} (workers=1) exceeds budget "
        f"{_fmt_mb(budget)}: raise the budget to ~{_fmt_mb(est_1)} or "
        f"use smaller chunks",
        code="TL1020",
        details={"estimated_peak_bytes": est_1,
                 "estimated_default_workers_bytes": est,
                 "budget_bytes": budget, "workers_requested": workers_req,
                 "suggested_budget_bytes": est_1,
                 "peak_task_detail": ctx.peak_detail,
                 **{k: v for k, v in decision.items()}})


def assess_pressure(observed_delta_bytes: float, budget_bytes,
                    estimated_bytes) -> None:
    """Raise TL1022 when observed execution pressure is significant.

    Below the noise floor, within budget, or within 2x of the model, this
    is a no-op: monitoring is observational and never fails healthy runs.
    Far above both budget and model, it reports execution-time pressure.
    """
    if budget_bytes is None:
        return
    try:
        observed = float(observed_delta_bytes)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return
    if observed <= budget_bytes or observed < PRESSURE_FLOOR_BYTES:
        return
    try:
        estimated = float(estimated_bytes)
    except (TypeError, ValueError):
        estimated = 0.0
    if observed <= 2 * max(estimated, 0.0):
        return
    from ..core.errors import TLUnsupportedError as _TLU

    raise _TLU(
        f"execution-time resource pressure: observed +{_fmt_mb(observed)} "
        f"above baseline (budget {_fmt_mb(budget_bytes)}, modeled "
        f"{_fmt_mb(estimated)}); already-running native operations can "
        f"temporarily exceed modeled limits",
        code="TL1022",
        details={"observed_delta_bytes": observed,
                 "budget_bytes": budget_bytes,
                 "estimated_bytes": estimated})


def _record(decision: dict, observed: dict) -> None:
    global LAST_DECISION, LAST_OBSERVED
    LAST_DECISION = dict(decision)
    LAST_OBSERVED = dict(observed)


def _compute_monitored(graph, workers: int):
    """Run one dask graph with per-call worker cap (never global config)
    while tracking baseline/peak high-water RSS. Returns (array, observed)."""
    baseline = _rss_peak_mb()
    try:
        if workers and hasattr(graph, "compute"):
            value = graph.compute(num_workers=int(workers))
        else:
            value = graph.compute()
    except TypeError:
        value = graph.compute()  # backend ignores the cap: still honest
    peak = _rss_peak_mb()
    try:
        delta = max(0.0, peak - baseline)
    except Exception:  # pragma: no cover - defensive
        delta = float("nan")
    return np.asarray(value), {"baseline_rss_mb": baseline,
                               "peak_rss_mb": peak, "delta_mb": delta}


# -- boundaries ------------------------------------------------------------------------------


def _finalize(ctx: _ChunkContext, graph, enforce: bool = True,
              purpose: str = "collect"):
    if _has_unknown_chunks(graph):
        raise _NotChunkable("plan produced unknown chunk dimensions",
                            code="TL1010", details={})
    from ..backends import dask_backend as _be

    tasks = _be.task_count(graph)
    if tasks < 0:
        raise _NotChunkable("plan resource requirements cannot be bounded",
                            code="TL1021", details={"operation": purpose})
    if enforce:
        if tasks > MAX_TASKS:
            raise _NotChunkable(f"plan would build {tasks} tasks "
                                f"(limit {MAX_TASKS})", code="TL1012",
                                details={"tasks": tasks, "limit": MAX_TASKS})
        block = _block_bytes(graph)
        if block > MAX_BLOCK_BYTES:
            raise _NotChunkable(
                f"plan needs ~{block / 1e6:.1f}MB blocks (limit "
                f"{MAX_BLOCK_BYTES / 1e6:.0f}MB)", code="TL1011",
                details={"block_bytes": block, "limit": MAX_BLOCK_BYTES})
    else:
        block = _block_bytes(graph)
    try:
        decision = decide(graph, ctx, purpose=purpose)
        rejected = None
    except _NotChunkable as exc:
        if not enforce:
            decision = {**exc.details, "purpose": purpose,
                        "outcome": "rejected",
                        "reason": exc.reason, "code": exc.code,
                        "budget_bytes": MEMORY_BUDGET_BYTES,
                        "workers_requested": _effective_workers(),
                        "workers_used": 1,
                        "peak_task_bytes": ctx.peak_task_bytes or None,
                        "peak_task_detail": ctx.peak_detail or "?"}
            rejected = exc
        else:
            raise
    ctx.plan.estimates = {"tasks": tasks,
                          "max_block_mb": round(block / 1e6, 2)
                          if block >= 0 else "?",
                          "partitions": int(getattr(graph, "npartitions", -1)),
                          "peak_task_bytes": decision.get("peak_task_bytes"),
                          "peak_task_mb": round(decision.get(
                              "peak_task_bytes", 0) / 1e6, 2)
                          if isinstance(decision.get("peak_task_bytes"), int)
                          else "?",
                          "peak_task_detail": decision.get(
                              "peak_task_detail", "?"),
                          "estimated_peak_mb": round(decision.get(
                              "estimated_peak_bytes", 0) / 1e6, 2)
                          if isinstance(decision.get("estimated_peak_bytes"),
                                        (int, float)) else None,
                          "budget_mb": (round(MEMORY_BUDGET_BYTES / 1e6, 2)
                                        if MEMORY_BUDGET_BYTES is not None
                                        else None),
                          "workers_requested": decision.get(
                              "workers_requested"),
                          "workers_used": decision.get("workers_used"),
                          "outcome": decision.get("outcome"),
                          "purpose": purpose,
                          "reason": decision.get("reason"),
                          "code": decision.get("code"),
                          "uncertainty": decision.get("uncertainty",
                                                      "modeled")}
    global LAST_DECISION
    LAST_DECISION = dict(decision)
    return ctx.plan.render(graph)


def collect_chunked(root: ir.Expr):
    """Execute a chunked root as one dask plan. Returns (value, kind, plan).

    Lifecycle: planned -> admitted -> running -> completed/failed. Sources
    are consistency-checked after admission (TL1030); interruption reports
    TL1032. Observed RSS is recorded for profile() but never rebrands a
    completed computation as a failure.
    """
    from ..runtime import execution as _ex
    from ..runtime import io as _io

    rec = _ex.begin("collect")
    try:
        ctx = _ChunkContext()
        graph = lower(root, ctx, {})
        plan_text = _finalize(ctx, graph, purpose="collect")
        decision = dict(LAST_DECISION)
        rec.decision = decision
        rec.sources = list(ctx.plan.inputs)
        rec.transition("admitted")
        try:
            verify_sources(ctx)
        except _NotChunkable as exc:
            raise TLUnsupportedError(
                f"source consistency check failed: {exc.reason}",
                code=exc.code, details=exc.details)
        workers = int(decision.get("workers_used") or _effective_workers())
        rec.transition("running")
        import time as _time
        t_compute = _time.perf_counter()
        try:
            value, observed = _compute_monitored(graph, workers)
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            raise _cancelled("collect", exc)
        except Exception as exc:
            raise _execution_failed("collect", None, exc)
        observed = {**observed,
                    "estimated_peak_mb": (decision.get("estimated_peak_bytes",
                                                       0) or 0) / 1e6
                    if isinstance(decision.get("estimated_peak_bytes"),
                                  (int, float)) else None,
                    "budget_mb": (MEMORY_BUDGET_BYTES / 1e6
                                  if MEMORY_BUDGET_BYTES is not None else None),
                    "workers_used": workers}
        _record(decision, observed)
        import time as _time2
        rec.timings_ms["compute_ms"] = (_time2.perf_counter()
                                        - t_compute) * 1000.0
        _ex.finish(rec, "completed")
        if ctx.used_source:
            _io.CHUNK_EXECUTIONS += 1
        return value, "dense", plan_text
    except (TLTypeError, TLUnsupportedError) as exc:
        code = getattr(exc, "code", "TL1001")
        _ex.finish(rec, "cancelled" if code == "TL1032" else "failed",
                   {"code": code,
                    "message": str(exc),
                    "details": getattr(exc, "details", {})})
        raise
    except _NotChunkable as exc:
        _ex.finish(rec, "failed",
                   {"code": exc.code, "message": exc.reason,
                    "details": exc.details})
        raise TLUnsupportedError(
            f"operation is not chunk-lowerable in v0.5: {exc.reason}",
            code=exc.code,
            details=exc.details,
        )


def _cancelled(op: str, exc: BaseException) -> TLUnsupportedError:
    from ..runtime import execution as _ex

    rec = _ex.current()
    if rec is not None:
        rec.transition("cancelled")
    return TLUnsupportedError(
        f"{op} interrupted: no new work is scheduled and completed "
        f"partitions are released; already-running native tasks may have "
        f"run to completion ({type(exc).__name__})",
        code="TL1032",
        details={"operation": op, "interruption": type(exc).__name__})


def _execution_failed(op: str, dest: str | None,
                      exc: Exception) -> TLUnsupportedError:
    return TLUnsupportedError(
        f"{op} failed"
        f"{f' for destination {dest!r}' if dest else ''}: {exc} "
        f"({type(exc).__name__}); no partial output was published",
        code="TL1031",
        details={"operation": op,
                 "destination": dest,
                 "backend_error": f"{type(exc).__name__}: {exc}"})


def collect_or_raise(root: ir.Expr):
    """Shared chunk boundary: single-query execution or a coded error."""
    try:
        return collect_chunked(root)
    except _NotChunkable as exc:
        raise TLUnsupportedError(
            f"operation is not chunk-lowerable in v0.5: {exc.reason}",
            code=exc.code,
            details=exc.details,
        )


# -- persistent publication (v0.5 M4) ------------------------------------------------------
# Two modes, one surface (``staged`` kwarg on write_zarr, default auto):
# direct streaming (lower storage overhead; partial outputs possible after
# failure) and staged publication (complete + validate before an atomic
# same-filesystem rename; preexisting destinations survive failures).


def publication_strategy(path: str, staged: bool | None) -> tuple[str, str]:
    """Decide staged vs direct publication without executing.

    Returns (strategy, reason). Explicit ``staged=True`` on a non-local
    store is rejected (TL1031): stronger guarantees are refused rather
    than silently weakened.
    """
    from ..backends import dask_backend as _be

    if staged is True:
        if not _be.is_local_store(path):
            raise TLUnsupportedError(
                f"staged publication is unsupported for non-local store "
                f"'{path}'; use direct streaming or a local destination",
                code="TL1031", details={"path": str(path)})
        return "staged", "local filesystem supports atomic same-directory rename"
    if staged is False:
        return "direct", "explicitly requested direct streaming"
    if _be.is_local_store(path):
        return "staged", "local filesystem default"
    return ("direct",
            "non-local store: staged publication unsupported; partial "
            "outputs are possible after failure")


def _precheck_destination(path: str, overwrite: bool, output_bytes: int,
                          strategy: str) -> None:
    """Fail fast before any compute: existence policy + staging disk space
    (storage bytes, reported separately from the RAM budget)."""
    import os as _os

    from ..backends import dask_backend as _be

    if _os.path.exists(path) and not overwrite:
        raise TLTypeError(
            f"chunked write target '{path}' already exists; pass "
            f"overwrite=True to replace it or choose a fresh path",
            code="TL1002", details={"path": str(path)})
    if strategy == "staged" and output_bytes > 0:
        free = _be.disk_free_bytes(path)
        if free >= 0 and free < output_bytes:
            raise TLUnsupportedError(
                f"insufficient disk space to stage '{path}': need "
                f"~{_fmt_mb(output_bytes)} storage (not RAM), "
                f"{_fmt_mb(free)} free",
                code="TL1031",
                details={"path": str(path), "needed_bytes": output_bytes,
                         "free_bytes": free})


def _reject_self_overwrite(dest: str, ctx: _ChunkContext) -> None:
    """A run must never publish over one of its own live sources."""
    import os as _os

    own = {_os.path.realpath(p) for p in ctx.sources}
    if _os.path.realpath(dest) in own:
        raise TLUnsupportedError(
            f"refusing to publish '{dest}' over its own live source; "
            f"write to a fresh path instead",
            code="TL1031",
            details={"destination": dest, "conflict": "source==destination"})


def _block_slices(chunks) -> list:
    """Per-block index ranges matching ``to_delayed`` order (C order)."""
    import itertools as _it

    offsets = []
    for axis in chunks:
        start, spans = 0, []
        for size in axis:
            spans.append((start, start + int(size)))
            start += int(size)
        offsets.append(spans)
    return [tuple(slice(a, b) for a, b in combo)
            for combo in _it.product(*offsets)]


def _write_partition_loop(graph, staging: str, workers: int) -> int:
    """Deterministic test-path writer: one partition per step (dask still
    computes each block). Honors fail_after_partitions / slow hooks. Only
    used when hooks are armed; production uses ``da.to_zarr``."""
    import time as _time

    import zarr

    shape = tuple(int(d) for d in graph.shape)
    first_chunks = tuple(max(1, int(c[0])) for c in _chunks_of(graph))
    store = zarr.open_array(staging, mode="w", shape=shape,
                            chunks=first_chunks, dtype=graph.dtype)
    slices = _block_slices(_chunks_of(graph))
    delayed = list(graph.to_delayed().flat)
    assert len(delayed) == len(slices), (len(delayed), len(slices))
    written = 0
    for index, (part, region) in enumerate(zip(delayed, slices)):
        interrupt_at = _TEST_HOOKS["raise_interrupt_at"]
        if interrupt_at is not None and index >= interrupt_at:
            raise KeyboardInterrupt(
                f"injected interruption at partition {index} "
                f"({written} staged; destination untouched)")
        limit = _TEST_HOOKS["fail_after_partitions"]
        if limit is not None and index >= limit:
            raise _NotChunkable(
                f"injected failure after {written} partitions "
                f"(staging holds a partial output; destination untouched)",
                code="TL1031",
                details={"written_partitions": written,
                         "total_partitions": len(delayed)})
        if _TEST_HOOKS["slow_partition_secs"]:
            _time.sleep(float(_TEST_HOOKS["slow_partition_secs"]))
        try:
            block = part.compute(num_workers=workers)
        except TypeError:
            block = part.compute()
        store[region] = np.asarray(block)
        written += 1
    return written


def _execute_write(graph, target: str, workers: int, overwrite: bool) -> dict:
    """Bulk write to ``target``; returns timing info. Test hooks select the
    deterministic partition loop, otherwise parallel ``da.to_zarr``."""
    import time as _time

    from ..backends import dask_backend as _be

    start = _time.perf_counter()
    if (_TEST_HOOKS["fail_after_partitions"] is not None
            or _TEST_HOOKS["raise_interrupt_at"] is not None
            or _TEST_HOOKS["slow_partition_secs"]):
        _write_partition_loop(graph, target, workers)
    else:
        _be.write_array(graph, target, overwrite=overwrite,
                        num_workers=workers)
    return {"write_ms": (_time.perf_counter() - start) * 1000.0}


def _cleanup_staging(staging: str | None) -> None:
    """Remove staging after workers are done; TL1035 with a recovery path
    when removal itself fails.

    Removal is verified: an interrupt landing inside a native call
    (e.g. ``zarr.open_array``) leaves that call's event-loop thread
    completing store creation after the first removal, so a reappeared
    staging directory is removed again within a small bounded window.
    Anything still present afterwards is left for
    ``recover_stale_staging()`` rather than masking the original
    error. A directory that vanishes on its own counts as cleaned.
    """
    if not staging:
        return
    import os as _os
    import time as _time

    if not _os.path.exists(staging):
        return
    from ..backends import dask_backend as _be

    try:
        _be.remove_path(staging)
    except FileNotFoundError:
        return
    except Exception as exc:
        raise TLUnsupportedError(
            f"cleanup of staging '{staging}' failed: {exc}; remove it "
            f"manually or call recover_stale_staging()",
            code="TL1035",
            details={"staging": staging,
                     "recovery": f"remove {staging!r} manually "
                                 f"after confirming no run is active"}) from exc
    deadline = _time.monotonic() + 2.0
    delay = 0.02
    while True:
        try:
            still_there = _os.path.exists(staging)
        except OSError:
            return
        if not still_there:
            return
        if _time.monotonic() >= deadline:
            # Best effort exhausted: the original error stands and the
            # leftover is recoverable via recover_stale_staging().
            return
        _time.sleep(delay)
        delay = min(delay * 2, 0.2)
        try:
            _be.remove_path(staging)
        except FileNotFoundError:
            return
        except Exception:
            # A genuine removal failure here must not mask the
            # interruption/failure being handled; leave the staging
            # for recover_stale_staging().
            return


def write_or_raise(root: ir.Expr, path: str, overwrite: bool,
                   staged: bool | None = None):
    """Write a lowerable pipeline to zarr with a lifecycle and integrity.

    Local destinations default to staged publication (validate, then an
    atomic rename); ``staged=False`` keeps direct streaming. The RAM
    budget gates the working set; outputs may exceed it (the documented
    streaming alternative). Raises coded errors; never publishes partial
    output as success.
    """
    import time as _time

    from ..backends import dask_backend as _be
    from ..runtime import execution as _ex

    dest = str(path)
    rec = _ex.begin("write", destination=dest)
    staging: str | None = None
    validated = False
    try:
        ctx = _ChunkContext()
        graph = lower(root, ctx, {})
        strategy, strategy_reason = publication_strategy(dest, staged)
        rec.strategy = strategy
        try:
            plan_text = _finalize(ctx, graph, purpose="stream")
        except _NotChunkable as exc:
            raise TLUnsupportedError(
                f"operation is not chunk-lowerable in v0.5: {exc.reason}",
                code=exc.code,
                details=exc.details,
            )
        decision = dict(LAST_DECISION)
        rec.decision = decision
        rec.sources = list(ctx.plan.inputs)
        rec.transition("admitted")
        try:
            verify_sources(ctx)
        except _NotChunkable as exc:
            raise TLUnsupportedError(
                f"source consistency check failed: {exc.reason}",
                code=exc.code, details=exc.details)
        output_bytes = int(decision.get("output_bytes") or 0)
        _precheck_destination(dest, overwrite, output_bytes, strategy)
        _reject_self_overwrite(dest, ctx)
        workers = int(decision.get("workers_used") or _effective_workers())
        shape = list(graph.shape)
        dtype_str = str(graph.dtype)
        expected_nchunks = _expected_nchunks(graph)
        if _TEST_HOOKS["fail_before_write"]:
            raise _NotChunkable("injected failure before first chunk write",
                                code="TL1031",
                                details={"stage": "running"})
        rec.transition("running")
        target = _be.staging_dir_for(dest) if strategy == "staged" else dest
        if strategy == "staged":
            staging = target
        write_info = {}
        try:
            write_info = _execute_write(graph, target, workers,
                                        overwrite=(strategy == "direct"
                                                   and overwrite))
        except _NotChunkable:
            raise
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            raise _cancelled("write_zarr", exc)
        except Exception as exc:
            raise _execution_failed("write_zarr", dest, exc)
        rec.timings_ms.update(write_info)
        baseline = _rss_peak_mb()
        if strategy == "direct":
            _baseline_own_output(dest)
            _record(decision, _observed_for(decision, baseline,
                                            _rss_peak_mb(), workers))
            plan_text = (plan_text + "\n  publication: direct streaming "
                         f"({strategy_reason}); partial outputs are "
                         f"possible after failure")
            _ex.finish(rec, "completed")
            _note_write(ctx)
            return True, plan_text
        rec.transition("committing")
        if _TEST_HOOKS["fail_validation"]:
            raise _NotChunkable("injected failure during output validation",
                                code="TL1034",
                                details={"stage": "committing"})
        t_validate = _time.perf_counter()
        report = _be.validate_store(target, shape, dtype_str,
                                    expected_nchunks)
        rec.timings_ms["validate_ms"] = (_time.perf_counter()
                                         - t_validate) * 1000.0
        if report["problems"]:
            raise TLUnsupportedError(
                f"output validation failed for '{dest}': "
                f"{'; '.join(report['problems'])}",
                code="TL1034", details={**report, "staging": target})
        validated = True
        if _TEST_HOOKS["fail_publication"]:
            raise _NotChunkable("injected failure during publication "
                                f"(validated staging retained at '{target}')",
                                code="TL1031",
                                details={"stage": "committing",
                                         "staging": target})
        t_publish = _time.perf_counter()
        try:
            _be.publish_staged(target, dest, overwrite)
        except TLTypeError as exc:
            if getattr(exc, "code", None) == "TL1002":
                staging = target  # validated output retained for recovery
                raise TLUnsupportedError(
                    f"concurrent publication lost: '{dest}' appeared "
                    f"while this run staged its output; validated staging "
                    f"retained at '{target}' (see recover_stale_staging)",
                    code="TL1002",
                    details={"path": dest, "staging": target})
            raise
        staging = None  # published: nothing to clean
        rec.timings_ms["publish_ms"] = (_time.perf_counter()
                                        - t_publish) * 1000.0
        _baseline_own_output(dest)
        _record(decision, _observed_for(decision, baseline,
                                        _rss_peak_mb(), workers))
        plan_text = (plan_text + "\n  publication: staged "
                     f"({strategy_reason}); validated "
                     f"{report['initialized']}/{report['actual_nchunks']} "
                     f"chunks then atomically renamed")
        _ex.finish(rec, "completed")
        _note_write(ctx)
        return True, plan_text
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        _cleanup_best_effort(staging)
        _ex.finish(rec, "cancelled",
                   {"code": "TL1032", "message": str(exc)})
        raise _cancelled("write_zarr", exc)
    except (TLTypeError, TLUnsupportedError) as exc:
        code = getattr(exc, "code", "TL1001")
        terminal = "cancelled" if code == "TL1032" else "failed"
        try:
            # Validated staging is retained (recoverable), never deleted.
            _cleanup_best_effort(None if validated else staging)
        except TLUnsupportedError as cleanup_exc:
            _ex.finish(rec, "failed",
                       {"code": "TL1035",
                        "message": str(cleanup_exc),
                        "details": dict(getattr(cleanup_exc, "details", {})),
                        "original": {"code": code, "message": str(exc)}})
            raise cleanup_exc from exc
        _ex.finish(rec, terminal,
                   {"code": code, "message": str(exc),
                    "details": getattr(exc, "details", {})})
        raise
    except _NotChunkable as exc:
        try:
            # Only validated-but-unpublished staging is retained; anything
            # unvalidated is removed (its validation report, if any, is
            # already in the diagnostic details).
            _cleanup_best_effort(None if validated else staging)
        except TLUnsupportedError as cleanup_exc:
            _ex.finish(rec, "failed", {"code": "TL1035"})
            raise cleanup_exc
        _ex.finish(rec, "failed",
                   {"code": exc.code, "message": exc.reason,
                    "details": exc.details})
        raise TLUnsupportedError(
            f"operation is not chunk-lowerable in v0.5: {exc.reason}",
            code=exc.code, details=exc.details)


def _baseline_own_output(dest: str) -> None:
    """A successful publish establishes the source baseline for ``dest``.

    Without this, re-reading own output later in a long-lived process
    trips TL1030 on the mtime change from our own write (false positive:
    report-once would demand a pointless retry). External modifications
    after our publish are still detected against this baseline.
    Best-effort: fingerprint failures leave prior state untouched.
    """
    from ..backends import dask_backend as _be

    try:
        fresh = _be.fingerprint(dest)
    except Exception:
        return
    if len(_SOURCE_BASELINES) >= _SOURCE_BASELINE_LIMIT:
        _SOURCE_BASELINES.pop(next(iter(_SOURCE_BASELINES)))
    _SOURCE_BASELINES[dest] = fresh


def _cleanup_best_effort(staging: str | None) -> None:
    """Staging cleanup after workers stopped (compute returned/raised)."""
    if staging:
        _cleanup_staging(staging)


def _note_write(ctx: _ChunkContext) -> None:
    from ..runtime import io as _io

    if ctx.used_source:
        _io.CHUNK_EXECUTIONS += 1


def _observed_for(decision: dict, baseline: float, peak: float,
                  workers: int) -> dict:
    try:
        delta = max(0.0, peak - baseline)
    except Exception:  # pragma: no cover - defensive
        delta = float("nan")
    return {"baseline_rss_mb": baseline, "peak_rss_mb": peak,
            "delta_mb": delta,
            "estimated_peak_mb": (decision.get("estimated_peak_bytes", 0)
                                  or 0) / 1e6
            if isinstance(decision.get("estimated_peak_bytes"),
                          (int, float)) else None,
            "budget_mb": (MEMORY_BUDGET_BYTES / 1e6
                          if MEMORY_BUDGET_BYTES is not None else None),
            "workers_used": workers}


def _expected_nchunks(graph) -> int:
    try:
        if any(int(d) == 0 for d in graph.shape):
            return 0
        total = 1
        for axis in graph.chunks:
            total *= len(axis)
        return total
    except Exception:  # pragma: no cover - defensive
        return -1


def try_write_zarr(root: ir.Expr, path: str, overwrite: bool,
                   staged: bool | None = None):
    """Stream a lowerable pipeline to zarr. Returns (True, plan) else
    (False, reason) — never materializes. Coded failures surface through
    write_or_raise; this wrapper preserves the legacy boolean protocol."""
    try:
        return write_or_raise(root, path, overwrite, staged=staged)
    except _NotChunkable as exc:  # pragma: no cover - defensive
        return False, exc.reason
    except (TLTypeError, TLUnsupportedError) as exc:
        if getattr(exc, "code", "TL1001") in ("TL1002", "TL1003"):
            raise
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def recover_stale_staging(dest: str, remove: bool = True) -> dict:
    """Report (and by default remove) stale staging artifacts for ``dest``.

    Legacy explicit-cleanup entry point: returns {"stale": [...],
    "removed": [...]}. Unlike :func:`recover`, removal here is explicit
    and unconditional. Prefer :func:`recover` for state-aware recovery.
    """
    from ..backends import dask_backend as _be

    stale = _be.list_stale_staging(str(dest))
    report: dict = {"stale": stale, "removed": []}
    if not remove:
        return report
    for path in stale:
        try:
            _be.remove_path(path)
            report["removed"].append(path)
        except Exception as exc:
            raise TLUnsupportedError(
                f"cleanup of stale staging '{path}' failed: {exc}; "
                f"remove it manually after confirming no run is active",
                code="TL1035",
                details={"staging": path, "remaining": stale,
                         "recovery": f"remove {path!r} manually"})
    return report


def list_stale_staging(dest: str) -> list[str]:
    """Identify staging/backup artifacts for ``dest`` left by interrupted
    runs (no reads, never executes)."""
    from ..backends import dask_backend as _be

    return _be.list_stale_staging(str(dest))


def recover(dest: str, *, publish_validated: bool = False,
            drop_validated: bool = False, overwrite: bool = False) -> dict:
    """Execute the recovery state table for ``dest`` (idempotent).

    States and safest actions (see LANGUAGE_SPEC §16):

    1. no staging/backup → report clean, no action.
    2. incomplete staging → remove it (destination untouched).
    3. validated staging, destination absent → retain + report by
       default; publish it iff ``publish_validated``; remove it iff
       ``drop_validated`` AND another verified copy exists.
    4. backup exists, destination absent → restore the newest backup
       (regain the previous valid dataset), then handle leftovers.
    5./6. backup + destination, destination census-complete → drop the
       superseded backup; destination invalid → touch nothing (ambiguous).
    7. ambiguous (multiples, unreadable) → per-item safe actions only.

    Never automatically deletes the only verified valid dataset. Never
    publishes over an existing destination without ``overwrite=True``.
    """
    import os as _os

    from ..backends import dask_backend as _be

    dest = str(dest)
    report: dict = {"destination": dest, "actions": [],
                    "retained": [], "removed": [], "published": None,
                    "destination_valid": None}
    try:
        return _recover_states(dest, report, publish_validated,
                               drop_validated, overwrite)
    except (TLTypeError, TLUnsupportedError):
        raise
    except Exception as exc:
        raise TLUnsupportedError(
            f"recovery for '{dest}' failed: {exc}",
            code="TL1035",
            details={"destination": dest,
                     "recovery": "inspect staging/backup paths manually"}) \
            from exc


def _recover_states(dest: str, report: dict, publish_validated: bool,
                    drop_validated: bool, overwrite: bool) -> dict:
    import os as _os

    from ..backends import dask_backend as _be

    def refresh():
        stale = _be.list_stale_staging(dest)
        return ([p for p in stale if ".tlstage-" in p and _os.path.exists(p)],
                [p for p in stale if ".tlbackup-" in p and _os.path.exists(p)])

    stagings, backups = refresh()
    if not stagings and not backups and not _os.path.exists(dest):
        report["actions"].append("clean: no artifacts, no destination")
        return report

    # State 4: backup but no destination -> restore the newest backup.
    if backups and not _os.path.exists(dest):
        backups.sort(key=lambda p: _os.stat(p).st_mtime_ns)
        primary, leftover = backups[-1], backups[:-1]
        _os.rename(primary, dest)  # atomic; destination was absent
        report["actions"].append(f"restored backup {primary} -> destination")
        stagings, backups = refresh()
        # Leftover older backups are superseded by the restored copy.
        for old in leftover:
            if _os.path.exists(old):
                _remove_recovery_path(old, report,
                                      "removed superseded older backup")

    # States 5/6/7: destination present — check its census first.
    dest_valid: bool | None = None
    if _os.path.exists(dest):
        dest_valid, _ = _be.census_complete(dest)
        report["destination_valid"] = dest_valid
    for backup in backups:
        if not _os.path.exists(backup):
            continue
        if dest_valid:
            _remove_recovery_path(backup, report,
                                  "removed backup superseded by valid "
                                  "destination")
        else:
            report["retained"].append(backup)
            report["actions"].append(
                f"retained {backup}: destination missing or invalid; "
                f"not the only verified copy to delete automatically")

    # States 2/3/7: staging artifacts, per-item safe actions.
    for staging in stagings:
        if not _os.path.exists(staging):
            continue
        ok, census = _be.census_complete(staging)
        if not ok:
            _remove_recovery_path(staging, report,
                                  f"removed incomplete staging "
                                  f"({census.get('initialized', '?')}/"
                                  f"{census.get('nchunks', '?')} chunks)")
            continue
        if publish_validated and (not _os.path.exists(dest) or overwrite):
            _be.publish_staged(staging, dest, overwrite=overwrite)
            report["published"] = staging
            report["actions"].append(f"published validated staging {staging}")
            dest_valid = True
            report["destination_valid"] = True
            continue
        if publish_validated and _os.path.exists(dest) and not overwrite:
            report["retained"].append(staging)
            report["actions"].append(
                f"retained validated {staging}: destination exists "
                f"(pass overwrite=True to publish)")
            continue
        others_valid = bool(dest_valid) or any(
            _os.path.exists(other) and other != staging
            and _be.census_complete(other)[0] for other in stagings)
        if drop_validated and others_valid:
            _remove_recovery_path(staging, report,
                                  "removed validated staging "
                                  "(another verified copy exists)")
        else:
            report["retained"].append(staging)
            report["actions"].append(
                f"retained validated {staging}: only verified copy; "
                f"publish or drop explicitly")
    if _os.path.exists(dest) and report["destination_valid"] is None:
        valid, _ = _be.census_complete(dest)
        report["destination_valid"] = valid
    if not report["actions"]:
        report["actions"].append("clean: nothing to recover")
    return report


def _remove_recovery_path(path: str, report: dict, action: str) -> None:
    from ..backends import dask_backend as _be

    try:
        _be.remove_path(path)
    except Exception as exc:
        raise TLUnsupportedError(
            f"recovery removal of '{path}' failed: {exc}; remove it "
            f"manually after confirming no run is active",
            code="TL1035",
            details={"staging": path, "recovery": f"remove {path!r}"})
    report["removed"].append(path)
    report["actions"].append(f"{action}: {path}")


def describe_write(root: ir.Expr, path: str, overwrite: bool,
                   staged: bool | None = None) -> str:
    """Read-only write plan: strategy, destination status, resource
    decision (never executes, never writes, never creates stores)."""
    from ..backends import dask_backend as _be

    lines = [f"write plan for {path}:"]
    try:
        strategy, reason = publication_strategy(str(path), staged)
    except (TLTypeError, TLUnsupportedError) as exc:
        return "\n".join(lines + [f"  strategy: unavailable ({exc})"])
    lines.append(f"  strategy: {strategy} ({reason})")
    import os as _os

    exists = _os.path.exists(str(path))
    lines.append(f"  destination: {'exists' if exists else 'absent'}"
                 + (", overwrite requested" if exists and overwrite else ""))
    if exists and not overwrite:
        lines.append("  admission: would fail [TL1002] (destination exists; "
                     "pass overwrite=True)")
    stale = _be.list_stale_staging(str(path))
    if stale:
        lines.append(f"  stale staging present: {stale} "
                     f"(see recover_stale_staging)")
    ctx = _ChunkContext()
    try:
        graph = lower(root, ctx, {})
        text = _finalize(ctx, graph, enforce=False, purpose="stream")
    except _NotChunkable as exc:
        return "\n".join(lines + [f"  admission: rejected ({exc.reason})"])
    except (TLTypeError, TLUnsupportedError, Exception) as exc:
        return "\n".join(lines + [f"  admission: no chunk lowering ({exc})"])
    est = ctx.plan.estimates
    est_text = (f"{est.get('estimated_peak_mb')}MB"
                if est.get('estimated_peak_mb') is not None
                else "n/a (budget unenforced)")
    budget_text = (f"{est.get('budget_mb')}MB"
                   if est.get('budget_mb') is not None else "unenforced")
    lines.append(f"  admission: {est.get('outcome')} "
                 f"(est {est_text} vs budget {budget_text})")
    lines.append(f"  output: shape {list(graph.shape)}, "
                 f"{_expected_nchunks(graph)} chunks")
    if strategy == "staged":
        lines.append("  publication: validate chunk census + metadata, "
                     "then atomic rename; preexisting destination "
                     "survives failures")
    return "\n".join(lines)


def explain_chunked(root: ir.Expr) -> tuple[str, bool]:
    """Physical chunk plan text without executing (metadata reads only)."""
    ctx = _ChunkContext()
    try:
        graph = lower(root, ctx, {})
        return _finalize(ctx, graph, enforce=False, purpose="collect"), True
    except _NotChunkable as exc:
        return (f"physical plan: no chunk lowering ({exc.reason})", False)
    except (TLTypeError, TLUnsupportedError, Exception) as exc:
        return f"physical plan: no chunk lowering ({exc})", False
