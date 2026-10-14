"""Reverse-mode automatic differentiation over operator graphs.

``grad(loss, wrt)`` builds a first-class gradient operator; application
binds :class:`DiffVar` leaves and evaluation runs a memoized forward trace
followed by vector-Jacobian propagation (never full Jacobians). Rules are
shape-polymorphic: shapes resolve at trace time, so operators stay
data-independent.

Subgradient conventions (explicit): ``abs`` is 0 at 0; ``min``/``max``
split 1 evenly among ties; ``where``/``clip``/``fill_null`` follow the
active branch; ``pow``/``log``/``sqrt``/``div`` propagate inf/nan at
singularities instead of inventing values. Integer intermediates compute
in float64; differentiable *inputs* must be floating-point (strict).
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from ..core import expression as ir
from ..core.errors import TLDiffError, TLShapeError, TLTypeError
from ..core.operator import Operator, _bind_arg, fresh_uid
from ..core.tensor import Lazy, Tensor, literal_of

# Node families with reverse rules (shown by explain(); everything else
# raises an explicit nondifferentiable/deferred diagnostic).
SUPPORTED = (
    "elementwise arithmetic (add/sub/mul/div/pow/mod), negation, abs, "
    "sin/cos/tan/exp/log/sqrt, sum/mean/min/max/std/var reductions, "
    "matmul, transpose/reshape/flatten, concat/stack/moveaxis, indexing "
    "with scatter, where/clip/fill_null masks, solve, vector 2-norm, "
    "fitted affine transforms (standardize/minmax/normalize-l2/impute/pca)"
)


def _diff_error(node_desc: str, reason: str) -> TLDiffError:
    return TLDiffError(f"cannot differentiate through {node_desc}: {reason}",
                       code="TL0806", details={"operation": node_desc})


def _f64(value, desc: str) -> np.ndarray:
    """Forward values into float64 math; honest errors for non-numerics."""
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _diff_error(desc, f"non-numeric data ({exc})")


# -- cotangent plumbing (kind-aware) ---------------------------------------------


def _check_all_float_frame(frame: pl.DataFrame, desc: str) -> None:
    bad = [c for c, d in frame.schema.items() if d not in (pl.Float32, pl.Float64)]
    if bad:
        raise _diff_error(
            desc, f"non-floating fields {bad} have no derivatives; select "
            f"numeric floating-point fields explicitly")


def _zeros_like(value, kind: str):
    if kind == "dense":
        return np.zeros(np.shape(value), dtype=np.float64), "dense"
    if kind == "series":
        if value.dtype not in (pl.Float32, pl.Float64):
            raise _diff_error("series", "non-floating series have no derivatives")
        return pl.Series(value.name, [0.0] * len(value)), "series"
    if kind == "frame":
        _check_all_float_frame(value, "structured cotangent")
        return pl.DataFrame(
            {c: [0.0] * value.height for c in value.columns}), "frame"
    raise _diff_error("scalar", "scalars carry no cotangent")


def _unbroadcast(cot: np.ndarray, shape: tuple) -> np.ndarray:
    cot = np.asarray(cot, dtype=np.float64)
    while cot.ndim > len(shape):
        cot = cot.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and cot.shape[axis] != 1:
            cot = cot.sum(axis=axis, keepdims=True)
    try:
        return np.reshape(cot, shape)
    except ValueError as exc:
        raise TLDiffError(f"cotangent shape {cot.shape} incompatible with "
                          f"target shape {shape}",
                          code="TL0807",
                          details={"cotangent": str(cot.shape),
                                   "target": str(shape)}) from exc


def _wrap_frame(matrix: np.ndarray, frame: pl.DataFrame) -> pl.DataFrame:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (frame.height, len(frame.columns)):
        raise TLDiffError("frame cotangent shape mismatch")
    return pl.DataFrame(
        {c: matrix[:, j].tolist() for j, c in enumerate(frame.columns)})


def _series_cot(cot_value, cot_kind: str, target) -> np.ndarray:
    n = len(target) if not isinstance(target, int) else target
    if cot_kind == "scalar":
        return np.full(n, float(cot_value))
    if cot_kind == "series":
        arr = np.asarray(cot_value.to_list(), dtype=np.float64)
        if arr.size != n:
            raise TLDiffError("series cotangent length mismatch")
        return arr
    if cot_kind == "dense":
        arr = np.asarray(cot_value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        if arr.size != n:
            raise TLDiffError("series cotangent length mismatch")
        return arr
    raise _diff_error("structured value", "cannot mix frame cotangents here")


def _frame_cot(cot_value, cot_kind: str, frame: pl.DataFrame) -> np.ndarray:
    if cot_kind == "scalar":
        return np.full((frame.height, len(frame.columns)), float(cot_value))
    if cot_kind == "frame":
        if cot_value.columns != frame.columns or \
                cot_value.height != frame.height:
            if cot_value.height == 1 and cot_value.columns == frame.columns:
                row = np.asarray(cot_value.row(0), dtype=np.float64)
                return np.broadcast_to(
                    row, (frame.height, len(frame.columns))).copy()
            raise TLDiffError("frame cotangent schema mismatch")
        return cot_value.to_numpy().astype(np.float64)
    if cot_kind == "dense":
        arr = np.asarray(cot_value, dtype=np.float64)
        if arr.size == 1:
            return np.full((frame.height, len(frame.columns)), float(arr.flat[0]))
        if arr.shape == (frame.height, len(frame.columns)):
            return arr
        raise TLDiffError("frame cotangent shape mismatch")
    raise _diff_error("structured value", "cannot mix series cotangents here")


def _scalar_cot(cot_value, cot_kind: str, desc: str) -> float:
    if cot_kind == "scalar":
        return float(cot_value)
    arr = (np.asarray(cot_value.to_list(), dtype=np.float64)
           if cot_kind == "series"
           else np.asarray(cot_value, dtype=np.float64)).reshape(-1)
    if arr.size != 1:
        raise TLDiffError(f"{desc} output must be scalar")
    return float(arr[0])


def _masked_back(cot_value, cot_kind: str, mask, branch_value, branch_kind: str,
                 desc: str):
    """Masked cotangent for where/clip/fill_null/impute.

    Scalar branches normalize to 0-d dense (constants or scalar graphs
    simply receive shaped cotangents); series/frames keep labels.
    """
    mask = np.asarray(mask, dtype=bool)
    if branch_kind == "scalar":
        branch_value = _f64(branch_value, desc)
        branch_kind = "dense"
    if branch_kind == "dense":
        C = _f64(cot_value, desc) if cot_kind == "dense" \
            else _cot_scalar_to(cot_value, cot_kind, desc)
        try:
            C, M = np.broadcast_arrays(C, mask)
        except ValueError as exc:
            raise TLDiffError(f"mask shape mismatch in {desc}") from exc
        return _unbroadcast(C * M, tuple(np.shape(branch_value))), "dense"
    if branch_kind == "series":
        n = len(branch_value)
        C = _cot_to_vector(cot_value, cot_kind, n, mask, desc)
        M = np.broadcast_to(mask, (n,)).reshape(n)
        return pl.Series(branch_value.name, (C * M).tolist()), "series"
    _check_all_float_frame(branch_value, desc)
    h, w = branch_value.height, len(branch_value.columns)
    C = _cot_to_matrix(cot_value, cot_kind, (h, w), mask.shape, desc)
    M = np.broadcast_to(mask, (h, w))
    return _wrap_frame(C * M, branch_value), "frame"


def _cot_scalar_to(cot_value, cot_kind: str, desc: str) -> np.ndarray:
    if cot_kind == "scalar":
        return np.asarray(float(cot_value))
    if cot_kind == "series":
        return np.asarray(cot_value.to_list(), dtype=np.float64)
    if cot_kind == "dense":
        return np.asarray(cot_value, dtype=np.float64)
    raise _diff_error(desc, "cannot mix frame cotangents here")


def _cot_to_vector(cot_value, cot_kind: str, n: int, mask, desc: str):
    C = _cot_scalar_to(cot_value, cot_kind, desc).reshape(-1)
    if C.size == 1:
        return np.full(n, float(C[0]))
    if C.size != n:
        raise TLDiffError(f"cotangent length mismatch in {desc}")
    return C


def _cot_to_matrix(cot_value, cot_kind: str, shape: tuple, mask_shape: tuple,
                   desc: str):
    if cot_kind == "scalar":
        return np.full(shape, float(cot_value))
    arr = (np.asarray(cot_value.to_list(), dtype=np.float64)
           if cot_kind == "series"
           else np.asarray(cot_value, dtype=np.float64))
    if arr.shape == shape:
        return arr
    if arr.shape == mask_shape:
        return np.broadcast_to(arr, shape).copy()
    if arr.size == 1:
        return np.full(shape, float(arr.flat[0]))
    raise TLDiffError(f"cotangent shape {arr.shape} incompatible in {desc}")


# -- forward trace -----------------------------------------------------------------


def _ad_children(node: ir.Expr) -> list[ir.Expr]:
    if isinstance(node, ir.ApplyOp):
        return list(node.args)
    if isinstance(node, ir.DiffVar):
        return [node.value]
    if isinstance(node, (ir.AdjointOf, ir.ForwardOf, ir.SvdPart, ir.EighPart)):
        return []
    from ..runtime.planner import _children

    return _children(node)


def _postorder(root: ir.Expr) -> list[ir.Expr]:
    order: list[ir.Expr] = []
    seen: set[int] = set()
    stack: list[tuple[ir.Expr, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in seen:
            continue
        if expanded:
            seen.add(id(node))
            order.append(node)
        else:
            stack.append((node, True))
            for child in _ad_children(node):
                if id(child) not in seen:
                    stack.append((child, False))
    return order


class _GradShared:
    """One memoized forward trace shared by a value and its gradient(s)."""

    def __init__(self, root: ir.Expr, wrt_keys: list[tuple[str, str]],
                 seed=None) -> None:
        self.root = root
        self.wrt_keys = list(wrt_keys)
        self.seed = seed  # None → ones (grad); else an Expr evaluated at reverse
        self.forward_runs = 0
        self.output = None
        self.values: dict[int, tuple] = {}
        self.order: list[ir.Expr] = []
        self.diffvars: dict[tuple[str, str], ir.Expr] = {}
        self.last_forward_ms: float | None = None
        self.last_backward_ms: float | None = None
        self.last_released_cotangents: int = 0

    def forward(self):
        if self.output is None:
            from ..runtime import chunk_plan as _chunk
            from ..runtime.executor import trace_evaluate

            if _chunk.contains_chunked(self.root):
                raise TLDiffError(
                    "differentiation over chunked data is deferred in v0.5 "
                    "M1; collect() explicitly to materialize first",
                    code="TL0809",
                )
            t0 = time.perf_counter()
            # Keep-forever tape: reverse() and later applies re-read
            # forward values after the pass (forward_runs == 1).
            result, cache = trace_evaluate(self.root, release=False)
            self.values = {key: entry[1] for key, entry in cache.items()}
            self.order = _postorder(self.root)
            for node in self.order:
                if isinstance(node, ir.DiffVar):
                    self.diffvars[(node.owner, node.name)] = node
                    _validate_diffvar(node, self.values[id(node)])
            self.output = result
            self.forward_runs += 1
            self.last_forward_ms = (time.perf_counter() - t0) * 1000.0
        return self.output

    def reverse(self):
        from ..runtime.executor import evaluate

        t0 = time.perf_counter()
        out_value, out_kind = self.forward()
        if self.seed is None:
            seed_value, seed_kind = _ones_seed(out_value, out_kind)
        else:
            seed_value, seed_kind = evaluate(self.seed)
            seed_value, seed_kind = _check_seed(
                seed_value, seed_kind, out_value, out_kind)
        cot: dict[int, tuple] = {id(self.root): (seed_value, seed_kind)}
        released = 0
        for node in reversed(self.order):
            key = id(node)
            if isinstance(node, ir.DiffVar):
                # DiffVar cotangents are read out after the loop (grads
                # below); they survive the pass. Everything else is
                # released once propagated: reverse-topological order
                # guarantees all consumers already accumulated into it,
                # so deletion is observably identical to retention.
                entry = cot.get(key)
                if entry is None:
                    continue
                _propagate(node, entry[0], entry[1], self, cot)
                continue
            entry = cot.pop(key, None)
            if entry is None:
                continue
            _propagate(node, entry[0], entry[1], self, cot)
            released += 1
        self.last_released_cotangents = released
        grads = []
        for key in self.wrt_keys:
            dnode = self.diffvars.get(key)
            if dnode is None:  # pragma: no cover - defensive
                raise TLDiffError(f"differentiable input '{key[1]}' not found")
            value, kind = self.values[id(dnode)]
            grads.append(_cast_to_input(cot.get(id(dnode)), value, kind))
        self.last_backward_ms = (time.perf_counter() - t0) * 1000.0
        return grads


def _validate_diffvar(node: ir.DiffVar, evaluated) -> None:
    value, kind = evaluated
    if kind == "dense":
        arr = np.asarray(value)
        if arr.dtype.kind != "f":
            raise TLDiffError(
                f"gradient inputs must be floating-point; parameter "
                f"'{node.name}' has dtype {arr.dtype} — pass float data",
                code="TL0804",
                details={"param": node.name, "dtype": str(arr.dtype)},
            )
    elif kind == "series":
        if value.dtype not in (pl.Float32, pl.Float64):
            raise TLDiffError(
                f"gradient inputs must be floating-point series; field "
                f"'{value.name}' has dtype {value.dtype}",
                code="TL0804",
                details={"param": node.name, "dtype": str(value.dtype)},
            )
    elif kind == "frame":
        _check_all_float_frame(value, f"differentiable input '{node.name}'")
    else:
        raise TLDiffError(
            f"gradient inputs must be tensors; parameter '{node.name}' "
            f"bound a scalar — pass tensor data")


def _ones_seed(out_value, out_kind):
    if out_kind == "dense" and tuple(np.shape(out_value)) == ():
        return np.ones((), dtype=np.float64), "dense"
    if out_kind == "scalar" and isinstance(out_value, (int, float, np.generic)) \
            and not isinstance(out_value, bool):
        return np.ones((), dtype=np.float64), "dense"
    shape = (tuple(np.shape(out_value)) if out_kind == "dense"
             else f"{out_kind}")
    raise TLDiffError(
        f"grad needs a scalar-valued operator, got {out_kind} output "
        f"with shape {shape}; use vjp() with an explicit cotangent for "
        f"non-scalar outputs",
        code="TL0803",
        details={"kind": out_kind, "shape": str(shape)},
    )


def _check_seed(seed_value, seed_kind: str, out_value, out_kind):
    if out_kind == "scalar":
        if seed_kind == "scalar":
            return float(seed_value), "scalar"
        arr = np.asarray(seed_value, dtype=np.float64)
        if arr.shape == ():
            return float(arr.flat[0]), "scalar"
        raise TLDiffError(
            f"vjp cotangent shape {arr.shape} does not match scalar output; "
            f"pass a scalar cotangent")
    if out_kind == "dense":
        arr = _f64(seed_value, "vjp cotangent")
        if arr.shape != tuple(np.shape(out_value)):
            raise TLDiffError(
                f"vjp cotangent shape {arr.shape} does not match output "
                f"shape {tuple(np.shape(out_value))}")
        return arr, "dense"
    if seed_kind != out_kind:
        raise TLDiffError("vjp cotangent kind does not match output kind",
                          code="TL0808",
                          details={"seed_kind": seed_kind,
                                   "output_kind": out_kind})
    if out_kind == "series":
        if len(seed_value) != len(out_value):
            raise TLDiffError("vjp cotangent length does not match output")
        return seed_value, "series"
    if seed_value.columns != out_value.columns or \
            seed_value.height != out_value.height:
        raise TLDiffError("vjp cotangent schema does not match output")
    return seed_value, "frame"


def _cast_to_input(hit, value, kind: str):
    if hit is None:
        return _zeros_like(value, kind)
    gvalue, gkind = hit
    if kind == "dense":
        arr = np.asarray(value)
        g = _f64(gvalue, "gradient")
        if g.shape != arr.shape:
            g = _unbroadcast(g, arr.shape)
        return g.astype(arr.dtype), "dense"
    if kind == "series":
        arr = _series_cot(gvalue, gkind, value)
        return pl.Series(value.name, arr.tolist()).cast(value.dtype), "series"
    _check_all_float_frame(value, "differentiable input")
    if gkind == "scalar":
        return _frame_cot(float(gvalue), "scalar", value), "frame"
    if gkind == "frame":
        if gvalue.columns != value.columns or gvalue.height != value.height:
            raise TLDiffError("frame cotangent schema mismatch")
        out = value.clone()
        for c in value.columns:
            out = out.with_columns(gvalue[c].cast(value[c].dtype).alias(c))
        return out, "frame"
    raise TLDiffError("cotangent kind mismatch")


# -- reverse rules -------------------------------------------------------------------


def _propagate(node: ir.Expr, cot_value, cot_kind: str,
               shared: _GradShared, cot: dict) -> None:
    values = shared.values

    def val(child):
        return values[id(child)]

    def acc(target, cvalue, ckind: str = "dense"):
        key = id(target)
        if key in cot:
            old, old_kind = cot[key]
            cvalue, ckind = _add_cot(old, old_kind, cvalue, ckind)
        cot[key] = (cvalue, ckind)

    if isinstance(node, (ir.Literal, ir.Const, ir.Param, ir.Scan, ir.DiffVar,
                       ir.AdjointOf, ir.ForwardOf)):
        return  # source leaves carry no gradient; DiffVar cotangents are
        # deposited by consumers and read out in reverse(); already-computed
        # gradient values reused as data are constants (stop-gradient)
    if isinstance(node, ir.BinOp):
        _rev_binop(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.UnOp):
        _rev_unop(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.FuncCall):
        _rev_func(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.FillNull):
        operand, okind = val(node.operand)
        mask = _notnull_mask(operand, okind)
        acc(node.operand, *_masked_back(cot_value, cot_kind, mask, operand,
                                        okind, "fill_null"))
        return
    if isinstance(node, ir.FieldAccess):
        _rev_field(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.GetItem):
        _rev_getitem(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.Transpose):
        obj, okind = val(node.obj)
        _require_dense(obj, okind, "transpose")
        acc(node.obj, np.transpose(_f64(cot_value, "transpose")))
        return
    if isinstance(node, ir.Reshape):
        obj, okind = val(node.obj)
        _require_dense(obj, okind, "reshape")
        acc(node.obj, np.reshape(_f64(cot_value, "reshape"), np.shape(obj)))
        return
    if isinstance(node, ir.Flatten):
        obj, okind = val(node.obj)
        if okind == "series":
            C = _series_cot(cot_value, cot_kind, obj)
            acc(node.obj, pl.Series(obj.name, C.tolist()), "series")
        else:
            _require_dense(obj, okind, "flatten")
            acc(node.obj, np.reshape(_f64(cot_value, "flatten"), np.shape(obj)))
        return
    if isinstance(node, ir.Concat):
        _rev_concat(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.Stack):
        _rev_stack(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.MoveAxis):
        obj, okind = val(node.obj)
        _require_dense(obj, okind, "moveaxis")
        acc(node.obj, np.moveaxis(_f64(cot_value, "moveaxis"),
                                  node.destination, node.source))
        return
    if isinstance(node, ir.Reduce):
        _rev_reduce(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.Where):
        _rev_where(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.Clip):
        _rev_clip(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.Solve):
        _rev_solve(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.Norm):
        _rev_norm(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.FittedTransform):
        _rev_fitted(node, cot_value, cot_kind, val, acc)
        return
    if isinstance(node, ir.ApplyOp):
        _rev_apply(node, cot_value, cot_kind, shared, cot)
        return
    if isinstance(node, (ir.SvdPart, ir.EighPart)):
        raise _diff_error("SVD/eigendecomposition",
                          "deferred in v0.4 (see LANGUAGE_SPEC.md)")
    if isinstance(node, (ir.SortBy, ir.Unique)):
        raise _diff_error("sorting/unique",
                          "permutations have no useful gradient; sort explicitly "
                          "outside differentiable graphs")
    raise _diff_error(type(node).__name__,
                      "not differentiable in v0.4; see LANGUAGE_SPEC.md for "
                      "the supported rule set")


def _require_dense(value, kind: str, desc: str) -> np.ndarray:
    if kind != "dense":
        raise _diff_error(desc, "structured operands need explicit numeric "
                                "selection")
    return _f64(value, desc)


def _add_cot(old, old_kind: str, new, new_kind: str):
    if old_kind == "dense" and new_kind == "dense":
        return (np.asarray(old, dtype=np.float64)
                + np.asarray(new, dtype=np.float64)), "dense"
    if old_kind == "series" and new_kind == "series":
        if len(old) != len(new):
            raise TLDiffError("series cotangent length mismatch")
        left = old.rename(new.name) if old.name != new.name else old
        return (left + new.rename(left.name)), "series"
    if old_kind == "frame" and new_kind == "frame":
        if old.columns != new.columns or old.height != new.height:
            raise TLDiffError("frame cotangent schema mismatch")
        return old.with_columns(
            [(old[c] + new[c]).alias(c) for c in old.columns]), "frame"
    if new_kind == "scalar":
        return _add_cot(old, old_kind, *_broadcast_scalar(new, old, old_kind))
    raise TLDiffError("cotangent kind mismatch during accumulation")


def _broadcast_scalar(scalar, target_value, target_kind: str):
    if target_kind == "dense":
        return np.full(np.shape(target_value), float(scalar),
                       dtype=np.float64), "dense"
    if target_kind == "series":
        return pl.Series(target_value.name,
                         [float(scalar)] * len(target_value)), "series"
    _check_all_float_frame(target_value, "structured cotangent")
    h = target_value.height
    return pl.DataFrame(
        {c: [float(scalar)] * h for c in target_value.columns}), "frame"


def _notnull_mask(operand, okind: str):
    if okind == "dense":
        arr = np.asarray(operand)
        if arr.dtype.kind == "f":
            return ~np.isnan(arr)
        return np.ones(arr.shape, dtype=bool)
    if okind == "series":
        return np.asarray(operand.is_not_null().to_list(), dtype=bool)
    _check_all_float_frame(operand, "fill_null")
    return operand.select(pl.all().is_not_null()).to_numpy().astype(bool)


def _rev_binop(node: ir.BinOp, cot_value, cot_kind: str, val, acc) -> None:
    op = node.op
    if op in ("eq", "ne", "gt", "lt", "ge", "le", "and", "or"):
        raise _diff_error(f"operator '{op}'",
                          "comparisons and logic are not differentiable; "
                          "branch on data outside differentiable graphs")
    if op == "matmul":
        _rev_matmul(node, cot_value, cot_kind, val, acc)
        return
    left, lkind = val(node.left)
    right, rkind = val(node.right)
    if lkind == "scalar":
        left = _f64(left, op)
        lkind = "dense"
    if rkind == "scalar":
        right = _f64(right, op)
        rkind = "dense"
    if lkind != "dense" or rkind != "dense":
        _rev_structured_binop(node, cot_value, cot_kind, val, acc)
        return
    L, R = _f64(left, op), _f64(right, op)
    C = _f64(cot_value, op)
    lshape, rshape = tuple(L.shape), tuple(R.shape)
    if op == "add":
        acc(node.left, _unbroadcast(C, lshape))
        acc(node.right, _unbroadcast(C, rshape))
    elif op == "sub":
        acc(node.left, _unbroadcast(C, lshape))
        acc(node.right, _unbroadcast(-C, rshape))
    elif op == "mul":
        acc(node.left, _unbroadcast(C * R, lshape))
        acc(node.right, _unbroadcast(C * L, rshape))
    elif op == "div":
        with np.errstate(all="ignore"):
            acc(node.left, _unbroadcast(C / R, lshape))
            acc(node.right, _unbroadcast(-C * L / R**2, rshape))
    elif op == "pow":
        with np.errstate(all="ignore"):
            dL = C * R * np.power(L, R - 1)
            dR = C * np.power(L, R) * np.log(L)
        acc(node.left, _unbroadcast(dL, lshape))
        acc(node.right, _unbroadcast(dR, rshape))
    elif op == "mod":
        with np.errstate(all="ignore"):
            dR = -C * np.floor(L / R)
        acc(node.left, _unbroadcast(C, lshape))
        acc(node.right, _unbroadcast(dR, rshape))


def _rev_structured_binop(node, cot_value, cot_kind: str, val, acc) -> None:
    left, lkind = val(node.left)
    right, rkind = val(node.right)
    op = node.op
    if (lkind == "series" or rkind == "series") and \
            (lkind in ("series", "scalar") and rkind in ("series", "scalar")):
        series = left if lkind == "series" else right
        if series.dtype not in (pl.Float32, pl.Float64):
            raise _diff_error(f"operator '{op}'",
                              "non-floating series have no derivatives")
        base = np.asarray(series.to_list(), dtype=np.float64)
        mate = (np.asarray(right.to_list(), dtype=np.float64)
                if rkind == "series" else float(_f64(right, op).flat[0])
                if rkind == "scalar" else None)
        if mate is None:
            raise _diff_error(f"operator '{op}'",
                              "cannot mix series with frames/arrays in "
                              "differentiable graphs")
        C = _series_cot(cot_value, cot_kind, series)
        d_self, d_mate = _elem_rev(op, C, base, mate)
        me = node.left if lkind == "series" else node.right
        acc(me, pl.Series(series.name, np.asarray(d_self).reshape(-1).tolist()),
            "series")
        other_value = right if lkind == "series" else left
        if isinstance(other_value, pl.Series):
            other_node = node.right if lkind == "series" else node.left
            acc(other_node, pl.Series(
                other_value.name,
                np.asarray(d_mate).reshape(-1).tolist()), "series")
        return
    if lkind == "frame" or rkind == "frame":
        frame = left if lkind == "frame" else right
        _check_all_float_frame(frame, f"operator '{op}'")
        mate_value, mate_kind = (right, rkind) if lkind == "frame" else (left, lkind)
        if mate_kind == "scalar":
            M = float(_f64(mate_value, op).flat[0])
        elif mate_kind == "frame":
            if mate_value.columns != frame.columns or \
                    mate_value.height != frame.height:
                raise TLDiffError("frame operand schema mismatch")
            M = mate_value.to_numpy().astype(np.float64)
        else:
            raise _diff_error(f"operator '{op}'",
                              "cannot mix frames with other kinds in "
                              "differentiable graphs")
        F = frame.to_numpy().astype(np.float64)
        C = _frame_cot(cot_value, cot_kind, frame)
        d_self, d_mate = _elem_rev(op, C, F, M)
        acc(node.left if lkind == "frame" else node.right,
            _wrap_frame(np.asarray(d_self), frame), "frame")
        if mate_kind == "frame":
            acc(node.right if lkind == "frame" else node.left,
                _wrap_frame(np.asarray(d_mate), mate_value), "frame")
        return
    raise _diff_error(f"operator '{op}'", "unsupported operand kinds")


def _elem_rev(op: str, C, L, R):
    """Elementwise VJP pieces; C matches L's shape."""
    if op == "add":
        return C, C
    if op == "sub":
        return C, -C
    if op == "mul":
        return C * R, C * L
    if op == "div":
        with np.errstate(all="ignore"):
            return C / R, -C * L / R**2
    if op == "pow":
        with np.errstate(all="ignore"):
            return C * R * np.power(L, R - 1), C * np.power(L, R) * np.log(L)
    if op == "mod":
        with np.errstate(all="ignore"):
            return C, -C * np.floor(L / R)
    raise _diff_error(f"operator '{op}'", "not differentiable")  # pragma: no cover


def _rev_matmul(node: ir.BinOp, cot_value, cot_kind: str, val, acc) -> None:
    A, akind = val(node.left)
    B, bkind = val(node.right)
    if akind != "dense" or bkind != "dense":
        raise _diff_error("matmul", "structured operands are not differentiable")
    A, B = _f64(A, "matmul"), _f64(B, "matmul")
    C = _f64(cot_value, "matmul")
    if A.ndim == 1 and B.ndim == 1:
        acc(node.left, C * B)
        acc(node.right, C * A)
    elif A.ndim == 1:
        acc(node.left, np.matmul(C, B.T))
        acc(node.right, A[:, None] * C[None, :])
    elif B.ndim == 1:
        acc(node.left, np.outer(C, B))
        acc(node.right, np.matmul(A.T, C))
    elif A.ndim == 2 and B.ndim == 2:
        acc(node.left, np.matmul(C, B.T))
        acc(node.right, np.matmul(A.T, C))
    elif A.ndim >= 2 and B.ndim >= 2:
        acc(node.left, np.matmul(C, np.swapaxes(B, -1, -2)))
        acc(node.right, np.matmul(np.swapaxes(A, -1, -2), C))
    else:
        raise _diff_error("matmul", f"shapes {A.shape}/{B.shape} unsupported")


def _rev_unop(node: ir.UnOp, cot_value, cot_kind: str, val, acc) -> None:
    if node.op == "not":
        raise _diff_error("operator 'not'", "logic is not differentiable")
    operand, okind = val(node.operand)
    if okind == "scalar":
        operand = _f64(operand, node.op)
        okind = "dense"
    X = _require_dense(operand, okind, node.op)
    C = _f64(cot_value, node.op)
    if node.op == "neg":
        acc(node.operand, _unbroadcast(-C, X.shape))
    elif node.op == "abs":
        acc(node.operand, _unbroadcast(C * np.sign(X), X.shape))


def _rev_func(node: ir.FuncCall, cot_value, cot_kind: str, val, acc) -> None:
    table = {"sin", "cos", "tan", "exp", "log", "sqrt"}
    if node.func not in table:
        raise _diff_error(f"function '{node.func}'",
                          "not differentiable in v0.4 (null handling, dropping "
                          "and logic have no derivatives)")
    operand, okind = val(node.args[0])
    if okind == "scalar":
        operand = _f64(operand, node.func)
        okind = "dense"
    X = _require_dense(operand, okind, node.func)
    C = _f64(cot_value, node.func)
    with np.errstate(all="ignore"):
        if node.func == "sin":
            d = C * np.cos(X)
        elif node.func == "cos":
            d = -C * np.sin(X)
        elif node.func == "tan":
            d = C * (1.0 + np.tan(X) ** 2)
        elif node.func == "exp":
            d = C * np.exp(X)
        elif node.func == "log":
            d = C / X
        else:
            d = C / (2.0 * np.sqrt(X))
    acc(node.args[0], _unbroadcast(d, X.shape))


def _rev_field(node: ir.FieldAccess, cot_value, cot_kind: str, val, acc) -> None:
    obj, okind = val(node.obj)
    if okind != "frame":
        raise _diff_error(f".{node.field}", "only structured field projection "
                                            "is supported here")
    _check_all_float_frame(obj, f".{node.field}")
    col = _series_cot(cot_value, cot_kind, obj.height)
    full = np.zeros((obj.height, len(obj.columns)))
    full[:, obj.columns.index(node.field)] = col
    acc(node.obj, _wrap_frame(full, obj), "frame")


def _rev_getitem(node: ir.GetItem, cot_value, cot_kind: str, val, acc) -> None:
    obj, okind = val(node.obj)
    index = _index_value(node.index, val)
    if okind == "dense":
        base = _f64(obj, "indexing")
        C = _f64(cot_value, "indexing")
        scattered = np.zeros(base.shape, dtype=np.float64)
        try:
            np.add.at(scattered, index, C)
        except (TypeError, ValueError):
            scattered[index] += C
        acc(node.obj, scattered)
        return
    if okind == "series":
        if isinstance(index, str):
            raise _diff_error("series indexing", "field selection on a series "
                                                 "has no derivative")
        arr = _f64(obj.to_list(), "indexing")
        C = _series_cot(cot_value, cot_kind, obj)
        scattered = np.zeros(len(arr))
        idx = _series_index(index, len(arr))
        try:
            np.add.at(scattered, idx, C)
        except (TypeError, ValueError):
            scattered[idx] += C
        acc(node.obj, pl.Series(obj.name, scattered.tolist()), "series")
        return
    if isinstance(index, ir.Const) and isinstance(index.value, (str, list)):
        cols = [index.value] if isinstance(index.value, str) else index.value
        _check_all_float_frame(obj, "field selection")
        C = _frame_cot(cot_value, cot_kind, obj.select(cols))
        full = np.zeros((obj.height, len(obj.columns)))
        for j, c in enumerate(cols):
            full[:, obj.columns.index(c)] = C[:, j]
        acc(node.obj, _wrap_frame(full, obj), "frame")
        return
    raise _diff_error("row filtering",
                      "differentiation through structured row selection is not "
                      "supported; filter outside differentiable graphs")


def _index_value(index: ir.Expr, val):
    if isinstance(index, ir.Const):
        return index.value
    value, kind = val(index)
    if kind == "series":
        try:
            return np.asarray(value.to_list())
        except (TypeError, ValueError) as exc:
            raise _diff_error("indexing", f"bad mask dtype ({exc})")
    if kind == "dense":
        return np.asarray(value)
    if kind == "scalar" and isinstance(value, slice):
        return value
    raise _diff_error("indexing", "index must be a slice, integer, or mask array")


def _series_index(index, n: int):
    if isinstance(index, slice):
        return index
    arr = np.asarray(index)
    if arr.dtype == bool:
        if arr.size != n:
            raise TLDiffError("mask length mismatch in gradient scatter")
        return arr
    try:
        return arr.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise _diff_error("indexing", f"bad index dtype ({exc})")


def _rev_concat(node: ir.Concat, cot_value, cot_kind: str, val, acc) -> None:
    parts = [val(p) for p in node.parts]
    if all(k == "dense" for _, k in parts):
        C = _f64(cot_value, "concat")
        arrays = [_f64(v, "concat") for v, _ in parts]
        ndim = {a.ndim for a in arrays}
        axis = node.axis % arrays[0].ndim if arrays[0].ndim else 0
        if len(ndim) != 1 or node.axis >= arrays[0].ndim:
            raise _diff_error("concat", "part shapes incompatible")
        sizes = [a.shape[axis] for a in arrays]
        if sum(sizes) != C.shape[axis]:
            raise TLDiffError("concat cotangent shape mismatch")
        try:
            split = np.split(C, np.cumsum(sizes)[:-1], axis=axis)
        except ValueError as exc:
            raise TLDiffError(f"concat backward failed: {exc}") from exc
        for part, piece in zip(node.parts, split):
            acc(part, piece)
        return
    if all(k == "series" for _, k in parts):
        series = [v for v, _ in parts]
        C = _series_cot(cot_value, cot_kind, series[0])
        if C.size != sum(len(s) for s in series):
            raise TLDiffError("concat cotangent length mismatch")
        pos = 0
        for part, s in zip(node.parts, series):
            piece = C[pos:pos + len(s)]
            pos += len(s)
            acc(part, pl.Series(s.name, piece.tolist()), "series")
        return
    raise _diff_error("concat", "mixed part kinds are not differentiable")


def _rev_stack(node: ir.Stack, cot_value, cot_kind: str, val, acc) -> None:
    parts = [val(p) for p in node.parts]
    C = _f64(cot_value, "stack")
    for part, (value, kind) in zip(node.parts, parts):
        if kind not in ("dense", "series"):
            raise _diff_error("stack", "structured parts are not differentiable")
    try:
        split = np.split(C, len(parts), axis=node.axis)
    except ValueError as exc:
        raise TLDiffError(f"stack backward failed: {exc}") from exc
    for (part, (value, kind)), piece in zip(zip(node.parts, parts), split):
        piece = np.squeeze(np.asarray(piece), axis=node.axis)
        if kind == "series":
            acc(part, pl.Series(value.name, np.asarray(piece).reshape(-1).tolist()),
                "series")
        else:
            acc(part, piece)


def _rev_reduce(node: ir.Reduce, cot_value, cot_kind: str, val, acc) -> None:
    if node.by is not None:
        raise _diff_error("grouped reduction",
                          "grouping is discrete; aggregate outside "
                          "differentiable graphs")
    from ..core.expression import resolve_axes

    operand, okind = val(node.operand)
    op = node.op
    if okind == "scalar":
        operand = _f64(operand, op)
        okind = "dense"
    if okind == "dense":
        X = _f64(operand, op)
        axes = resolve_axes(node.axis, X.ndim, op, X.shape)
        c = _shaped_cot(cot_value, cot_kind, op, X.shape, axes,
                        node.keepdims)
        acc(node.operand, _reduce_rev_axis(op, X, c, axes))
        return
    if okind == "series":
        if node.keepdims:
            raise _diff_error(
                f"reduction '{op}'",
                "keepdims has no series result form; collect() to dense "
                "first for shaped reductions")
        if operand.dtype not in (pl.Float32, pl.Float64):
            raise _diff_error(f"reduction '{op}'",
                              "non-floating series have no derivatives")
        X = np.asarray(operand.to_list(), dtype=np.float64)
        c = _scalar_cot(cot_value, cot_kind, op)
        acc(node.operand,
            pl.Series(operand.name, _reduce_rev(op, X, c).tolist()), "series")
        return
    _check_all_float_frame(operand, f"reduction '{op}'")
    if node.axis is not None:
        raise _diff_error(
            f"reduction '{op}'",
            "axis reductions over frames are ambiguous (rows vs fields); "
            "reduce fields explicitly or collect() to dense first")
    C = _frame_cot(cot_value, cot_kind, operand)
    if C.shape[0] != 1:
        raise TLDiffError(f"reduction '{op}' output must be one row")
    F = operand.to_numpy().astype(np.float64)
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        out[:, j] = _reduce_rev(op, F[:, j], float(C[0, j]))
    acc(node.operand, _wrap_frame(out, operand), "frame")


def _shaped_cot(cot_value, cot_kind: str, op: str, shape: tuple,
                axes: tuple, keepdims: bool) -> np.ndarray:
    """Broadcast an axis-reduction cotangent back to the input shape."""
    c = np.asarray(_f64(cot_value, f"{op} cotangent"), dtype=np.float64)
    if keepdims:
        try:
            return np.broadcast_to(c, shape).copy()
        except ValueError:
            pass
    else:
        full = list(c.shape)
        for ax in axes:
            full.insert(ax, 1)
        try:
            c = np.asarray(c).reshape(full)
            return np.broadcast_to(c, shape).copy()
        except ValueError:
            pass
    raise TLDiffError(f"reduction '{op}' cotangent shape {c.shape} "
                      f"incompatible with input shape {list(shape)}",
                      code="TL0807",
                      details={"cotangent": str(c.shape),
                               "shape": [int(d) for d in shape]})


def _reduce_rev(op: str, X: np.ndarray, c: float) -> np.ndarray:
    """Legacy whole-array rule (series/frame paths); axis-aware callers
    use :func:`_reduce_rev_axis` directly."""
    axes = tuple(range(X.ndim))
    return _reduce_rev_axis(op, X, np.full(X.shape, c), axes)


def _reduce_rev_axis(op: str, X: np.ndarray, c: np.ndarray,
                     axes: tuple) -> np.ndarray:
    """Axis-aware reverse rule; cotangent ``c`` already matches X.shape."""
    n = 1
    for ax in axes:
        n *= X.shape[ax]
    if op == "sum":
        return c.copy()
    if op == "mean":
        return np.full(X.shape, 0.0) + c / n
    if op in ("min", "max"):
        m = X.min(axis=axes, keepdims=True) if op == "min" else X.max(
            axis=axes, keepdims=True)
        mask = X == m
        # Ties split the seed within each output element (established
        # whole-array convention, applied per reduced slice).
        counts = np.maximum(mask.sum(axis=axes, keepdims=True), 1)
        return np.where(mask, c / counts, 0.0)
    if op == "var":
        if n == 0:
            return np.full(X.shape, 0.0)
        mu = X.mean(axis=axes, keepdims=True)
        return c * 2.0 * (X - mu) / n
    if op == "std":
        if n == 0:
            return np.full(X.shape, 0.0)
        mu = X.mean(axis=axes, keepdims=True)
        sd = X.std(axis=axes, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = c * (X - mu) / (n * sd)
        # Constant slices: zero subgradient (established convention).
        return np.where(sd == 0.0, np.zeros(X.shape), out)
    raise _diff_error(f"reduction '{op}'", "not differentiable")  # pragma: no cover


def _rev_where(node: ir.Where, cot_value, cot_kind: str, val, acc) -> None:
    cond, ckind = val(node.cond)
    if ckind == "frame":
        mask = np.asarray(cond.select(
            [pl.col(c).fill_null(False) for c in cond.columns]
        ).to_numpy(), dtype=bool)
    elif ckind == "series":
        mask = np.asarray(cond.to_list(), dtype=bool)
    else:
        mask = np.asarray(cond, dtype=bool)
    for branch in (node.a, node.b):
        bvalue, bkind = val(branch)
        pick = mask if branch is node.a else ~mask
        acc(branch, *_masked_back(cot_value, cot_kind, pick, bvalue, bkind,
                                  "where"))


def _rev_clip(node: ir.Clip, cot_value, cot_kind: str, val, acc) -> None:
    obj, okind = val(node.obj)
    lo, _ = val(node.lo)
    hi, _ = val(node.hi)
    if okind == "scalar":
        obj = _f64(obj, "clip")
        okind = "dense"
    if okind == "dense":
        X = _f64(obj, "clip")
        keep = np.ones(X.shape, dtype=bool)
        if lo is not None:
            keep &= X > np.asarray(lo, dtype=np.float64)
        if hi is not None:
            keep &= X < np.asarray(hi, dtype=np.float64)
        acc(node.obj, *_masked_back(cot_value, cot_kind, keep, obj, "dense", "clip"))
    elif okind == "series":
        X = np.asarray(obj.to_list(), dtype=np.float64)
        keep = np.ones(len(X), dtype=bool)
        if lo is not None:
            keep &= X > float(np.asarray(lo).flat[0])
        if hi is not None:
            keep &= X < float(np.asarray(hi).flat[0])
        acc(node.obj, *_masked_back(cot_value, cot_kind, keep, obj, "series", "clip"))
    elif okind == "frame":
        _check_all_float_frame(obj, "clip")
        X = obj.to_numpy().astype(np.float64)
        keep = np.ones(X.shape, dtype=bool)
        if lo is not None:
            keep &= X > np.asarray(lo, dtype=np.float64)
        if hi is not None:
            keep &= X < np.asarray(hi, dtype=np.float64)
        acc(node.obj, *_masked_back(cot_value, cot_kind, keep, obj, "frame", "clip"))
    else:
        raise _diff_error("clip", "unsupported operand kind")


def _rev_solve(node: ir.Solve, cot_value, cot_kind: str, val, acc) -> None:
    import scipy.linalg as _la

    A, akind = val(node.a)
    B, bkind = val(node.b)
    if akind != "dense" or bkind != "dense":
        raise _diff_error("solve", "structured operands are not differentiable")
    A, B = _f64(A, "solve"), _f64(B, "solve")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise _diff_error("solve", f"coefficient shape {A.shape} unsupported")
    C = _f64(cot_value, "solve")
    if C.shape != B.shape:
        raise TLDiffError("solve cotangent shape mismatch")
    try:
        X = _la.solve(A, B)
        dB = _la.solve(A.T, C)
    except _la.LinAlgError as exc:
        raise TLDiffError(f"cannot differentiate solve: {exc}") from exc
    acc(node.b, dB)
    acc(node.a, -np.outer(dB, X) if X.ndim == 1 else -np.matmul(dB, X.T))


def _rev_norm(node: ir.Norm, cot_value, cot_kind: str, val, acc) -> None:
    operand, okind = val(node.obj)
    if okind == "series":
        name = operand.name
        operand = np.asarray(operand.to_list(), dtype=np.float64)
        okind = "dense"
        series_name: Any = name
    else:
        series_name = None
    if okind != "dense":
        raise _diff_error("norm", "structured operands need explicit selection")
    X = _f64(operand, "norm")
    c = _scalar_cot(cot_value, cot_kind, "norm")
    if node.axis is not None or (node.ord not in (None, 2, 2.0, "fro")):
        raise _diff_error("norm", "only the default vector/Frobenius 2-norm "
                                  "differentiates in v0.4")
    n = float(np.linalg.norm(X))
    grad = np.zeros(X.shape) if n == 0.0 else c * X / n
    if series_name is not None:
        acc(node.obj, pl.Series(series_name, grad.reshape(-1).tolist()), "series")
    else:
        acc(node.obj, _unbroadcast(grad, X.shape))


def _rev_fitted(node: ir.FittedTransform, cot_value, cot_kind: str, val, acc) -> None:
    state = node.state
    if state is None:
        raise TLDiffError(
            f"operator '{node.kind}' is not fitted: differentiation through "
            f"fitting is deferred in v0.4 — call .fit(training_data) first; "
            f"only fitted application differentiates",
            code="TL0805",
            details={"estimators": [node.kind]},
        )
    kind = state.kind
    cfg = dict(state.config)
    if kind == "standardize":
        _affine_back(state, node, acc, val, cot_value, cot_kind,
                     1.0 / state.array("scale_"))
    elif kind == "minmax":
        factor = (float(cfg["hi"]) - float(cfg["lo"])) / state.array("range_")
        _affine_back(state, node, acc, val, cot_value, cot_kind, factor)
    elif kind == "normalize":
        if cfg.get("norm", "l2") != "l2":
            raise _diff_error(f"normalize('{cfg.get('norm')}')",
                              "only l2 normalization differentiates in v0.4")
        _normalize_back(state, node, acc, val, cot_value, cot_kind)
    elif kind == "impute":
        _impute_back(state, node, acc, val, cot_value, cot_kind)
    elif kind == "pca":
        _pca_back(state, node, acc, val, cot_value, cot_kind, cfg)
    else:  # pragma: no cover - guarded at construction
        raise TLDiffError(f"unknown estimator '{kind}'")


def _fitted_matrix(state, node: ir.FittedTransform, val):
    operand, okind = val(node.arg)
    if okind == "dense":
        M = _f64(operand, state.kind)
        if M.ndim == 1:
            M = M.reshape(-1, 1)
        if M.ndim != 2 or M.shape[1] != state.n_features:
            raise TLDiffError(
                f"operator '{state.kind}' fitted on {state.n_features} "
                f"features; input has shape {np.shape(operand)}")
        return M, None
    if okind == "series":
        if state.n_features != 1:
            raise TLDiffError("series/estimator width mismatch")
        if operand.dtype not in (pl.Float32, pl.Float64):
            raise _diff_error(state.kind, "non-floating series")
        return (np.asarray(operand.to_list(), dtype=np.float64).reshape(-1, 1),
                ("series", operand.name))
    _check_all_float_frame(operand, f"operator '{state.kind}'")
    missing = [c for c in state.numeric_cols if c not in operand.columns]
    if missing:
        raise TLDiffError(f"missing features {missing}")
    return (operand.select(list(state.numeric_cols)).to_numpy().astype(np.float64),
            ("frame", operand))


def _reassemble_fitted(D: np.ndarray, tag, state) -> tuple:
    if tag is None:
        return D, "dense"
    if tag[0] == "series":
        return pl.Series(tag[1], np.asarray(D).reshape(-1).tolist()), "series"
    frame = tag[1]
    full = np.zeros((frame.height, len(frame.columns)))
    for j, col in enumerate(state.numeric_cols):
        full[:, frame.columns.index(col)] = np.asarray(D)[:, j]
    return _wrap_frame(full, frame), "frame"


def _fitted_cot(state, cot_value, cot_kind: str, M, tag):
    if cot_kind == "dense":
        C = _f64(cot_value, state.kind)
        return C if C.shape == M.shape else _unbroadcast(C, M.shape)
    if cot_kind == "series":
        dummy = pl.Series("c", [0.0] * M.shape[0])
        return _series_cot(cot_value, cot_kind, dummy).reshape(M.shape)
    if tag is not None and tag[0] == "frame":
        return _frame_cot(cot_value, cot_kind, tag[1])
    raise _diff_error(state.kind, "unsupported cotangent kind")


def _affine_back(state, node, acc, val, cot_value, cot_kind: str, factor) -> None:
    M, tag = _fitted_matrix(state, node, val)
    C = _fitted_cot(state, cot_value, cot_kind, M, tag)
    D = C * np.asarray(factor, dtype=np.float64)
    acc(node.arg, *_reassemble_fitted(D, tag, state))


def _normalize_back(state, node, acc, val, cot_value, cot_kind: str) -> None:
    M, tag = _fitted_matrix(state, node, val)
    C = _fitted_cot(state, cot_value, cot_kind, M, tag)
    norms = np.sqrt((M ** 2).sum(axis=1, keepdims=True))
    norms[norms == 0.0] = 1.0
    Y = M / norms
    dot = (C * Y).sum(axis=1, keepdims=True)
    acc(node.arg, *_reassemble_fitted((C - Y * dot) / norms, tag, state))


def _impute_back(state, node, acc, val, cot_value, cot_kind: str) -> None:
    M, tag = _fitted_matrix(state, node, val)
    C = _fitted_cot(state, cot_value, cot_kind, M, tag)
    acc(node.arg, *_reassemble_fitted(np.where(np.isnan(M), 0.0, C), tag, state))


def _pca_back(state, node, acc, val, cot_value, cot_kind: str, cfg: dict) -> None:
    M, tag = _fitted_matrix(state, node, val)
    comps = np.asarray(state.array("components_"), dtype=np.float64)
    if cfg.get("whiten"):
        comps = comps / np.sqrt(
            np.asarray(state.array("explained_variance_"), dtype=np.float64))[:, None]
    k = comps.shape[0]
    if cot_kind == "dense":
        C = _f64(cot_value, "pca")
    elif cot_kind == "frame":
        C = np.asarray(cot_value.to_numpy(), dtype=np.float64)
    else:
        raise _diff_error("pca", "series cotangents are unsupported")
    if C.shape != (M.shape[0], k):
        raise TLDiffError("pca cotangent shape mismatch")
    acc(node.arg, *_reassemble_fitted(C @ comps, tag, state))


def _rev_apply(node: ir.ApplyOp, cot_value, cot_kind: str,
               shared: _GradShared, cot: dict) -> None:
    op = node.operator
    arg_vals = []
    for a in node.args:
        if id(a) not in shared.values:  # pragma: no cover - defensive
            raise TLDiffError("missing forward value in nested application")
        arg_vals.append(shared.values[id(a)])
    if len(arg_vals) != len(op.params):
        raise TLDiffError("nested operator arity mismatch")  # pragma: no cover
    env, lits = {}, []
    for name, (value, kind) in zip(op.params, arg_vals):
        lit = ir.Literal(value, kind)
        env[(op.uid, name)] = lit
        lits.append(lit)
    nested_cots = _NestedReverse(op, env).run(cot_value, cot_kind)
    for a, lit in zip(node.args, lits):
        hit = nested_cots.get(id(lit))
        if hit is not None:
            old = cot.get(id(a))
            cot[id(a)] = hit if old is None else _add_cot(old[0], old[1],
                                                          hit[0], hit[1])


class _NestedReverse:
    """Reverse through an embedded operator application."""

    def __init__(self, op: Operator, env: dict) -> None:
        self.op = op
        self.env = env

    def run(self, seed_value, seed_kind):
        from ..runtime.executor import trace_evaluate

        sub_root = ir.substitute(self.op.root, self.env)
        # Keep-forever: values are read throughout the nested reverse.
        _, cache = trace_evaluate(sub_root, release=False)
        values = {key: entry[1] for key, entry in cache.items()}
        order = _postorder(sub_root)
        cot: dict[int, tuple] = {id(sub_root): (seed_value, seed_kind)}
        probe = _Probe(values)
        for target in reversed(order):
            entry = cot.get(id(target))
            if entry is None:
                continue
            _propagate(target, entry[0], entry[1], probe, cot)
        return cot


class _Probe:
    """Minimal shared-like facade for nested reverses."""

    def __init__(self, values) -> None:
        self.values = values


# -- operator surface --------------------------------------------------------------


class _GradBase(Operator):
    _tl_grad_ = True

    def explain(self) -> str:
        from ..runtime.inspect import format_node

        lines = [f"{self._grad_head()} over params {list(self.params)}:",
                 format_node(self.root),
                 "",
                 f"differentiable rules: {SUPPORTED}"]
        return "\n".join(lines)


class GradOperator(_GradBase):
    """First-class gradient operator from ``grad(loss, wrt)``."""

    def __init__(self, source: Operator, wrt: list[str]) -> None:
        uid = fresh_uid()
        self.source = source
        self.wrt = list(wrt)
        root = ir.remap_params(source.root, uid)
        super().__init__(tuple(source.params), root, uid)

    def _grad_head(self) -> str:
        return f"grad(wrt={self.wrt})"

    def __call__(self, *args):
        if len(args) != len(self.params):
            raise TLShapeError(
                f"gradient operator expects {len(self.params)} argument(s) "
                f"{list(self.params)}, got {len(args)}")
        target = None
        if len(args) == 1 and isinstance(args[0], Tensor):
            target = args[0]
        env: dict[tuple[str, str], ir.Expr] = {}
        for name, value in zip(self.params, args):
            if isinstance(value, Operator):
                raise TLTypeError("cannot differentiate with respect to operators")
            if name in self.wrt:
                env[(self.uid, name)] = ir.DiffVar(
                    self.uid, name, _bind_snapshot(value))
            else:
                env[(self.uid, name)] = _bind_arg(value)
        shared = _GradShared(ir.substitute(self.root, env),
                             [(self.uid, n) for n in self.wrt])
        if len(self.wrt) == 1:
            return Lazy(ir.AdjointOf(shared, (0,)), target)
        return tuple(Lazy(ir.AdjointOf(shared, (i,)), None)
                     for i in range(len(self.wrt)))


class ValueGradOperator(_GradBase):
    """First-class value-and-gradient operator."""

    def __init__(self, source: Operator, wrt: list[str]) -> None:
        uid = fresh_uid()
        self.source = source
        self.wrt = list(wrt)
        root = ir.remap_params(source.root, uid)
        super().__init__(tuple(source.params), root, uid)

    def _grad_head(self) -> str:
        return f"value_and_grad(wrt={self.wrt})"

    def __call__(self, *args):
        if len(args) != len(self.params):
            raise TLShapeError(
                f"operator expects {len(self.params)} argument(s) "
                f"{list(self.params)}, got {len(args)}")
        env: dict[tuple[str, str], ir.Expr] = {}
        for name, value in zip(self.params, args):
            if isinstance(value, Operator):
                raise TLTypeError("cannot differentiate with respect to operators")
            if name in self.wrt:
                env[(self.uid, name)] = ir.DiffVar(
                    self.uid, name, _bind_snapshot(value))
            else:
                env[(self.uid, name)] = _bind_arg(value)
        shared = _GradShared(ir.substitute(self.root, env),
                             [(self.uid, n) for n in self.wrt])
        value_lazy = Lazy(ir.ForwardOf(shared))
        if len(self.wrt) == 1:
            return ValueGradResult((value_lazy, Lazy(ir.AdjointOf(shared, (0,)))))
        return ValueGradResult((
            value_lazy,
            tuple(Lazy(ir.AdjointOf(shared, (i,)))
                  for i in range(len(self.wrt)))))


class VjpOperator(_GradBase):
    """First-class vector-Jacobian product (cotangent appended at call)."""

    def __init__(self, source: Operator, wrt: list[str]) -> None:
        uid = fresh_uid()
        self.source = source
        self.wrt = list(wrt)
        root = ir.remap_params(source.root, uid)
        super().__init__(tuple(source.params), root, uid)

    def _grad_head(self) -> str:
        return f"vjp(wrt={self.wrt})"

    def __call__(self, *args):
        if len(args) != len(self.params) + 1:
            raise TLShapeError(
                f"vjp operator expects {len(self.params)} data argument(s) "
                f"plus a cotangent seed, got {len(args)}")
        *data_args, seed = args
        env: dict[tuple[str, str], ir.Expr] = {}
        for name, value in zip(self.params, data_args):
            if isinstance(value, Operator):
                raise TLTypeError("cannot differentiate with respect to operators")
            if name in self.wrt:
                env[(self.uid, name)] = ir.DiffVar(
                    self.uid, name, _bind_snapshot(value))
            else:
                env[(self.uid, name)] = _bind_arg(value)
        shared = _GradShared(ir.substitute(self.root, env),
                             [(self.uid, n) for n in self.wrt],
                             seed=_bind_seed(seed))
        if len(self.wrt) == 1:
            return Lazy(ir.AdjointOf(shared, (0,)))
        return tuple(Lazy(ir.AdjointOf(shared, (i,)))
                     for i in range(len(self.wrt)))


class ValueGradResult(tuple):
    """``(value, gradient)`` pair sharing one forward evaluation."""

    @property
    def value(self):
        return self[0]

    @property
    def gradient(self):
        return self[1]


def _bind_snapshot(value) -> ir.Expr:
    if isinstance(value, Lazy):
        return value.root
    if isinstance(value, Tensor):
        return value._snapshot_node()
    if isinstance(value, bool):
        raise TLDiffError("gradient inputs must be floating-point tensors")
    if isinstance(value, int):
        raise TLDiffError(
            "gradient inputs must be floating-point; integer data has no "
            "derivative — pass float tensors")
    if isinstance(value, float):
        return ir.Literal(np.asarray(value), "dense")
    if isinstance(value, np.ndarray):
        return literal_of(value)
    if isinstance(value, (pl.Series, pl.DataFrame)):
        return literal_of(value)
    raise TLTypeError(
        f"cannot differentiate with respect to {type(value).__name__}")


def _bind_seed(seed) -> ir.Expr:
    if isinstance(seed, Lazy):
        return seed.root
    if isinstance(seed, Tensor):
        return seed._snapshot_node()
    if isinstance(seed, np.ndarray):
        return literal_of(seed)
    if isinstance(seed, (pl.Series, pl.DataFrame)):
        return literal_of(seed)
    if isinstance(seed, (int, float)) and not isinstance(seed, bool):
        return ir.Const(float(seed))
    raise TLTypeError(
        f"vjp cotangent must be array-like, got {type(seed).__name__}")


def _resolve_wrt(source: Operator, wrt) -> list[str]:
    if isinstance(source, _GradBase):
        raise TLDiffError(
            "second-order differentiation is not supported in v0.4: grad() "
            "of a gradient operator is rejected",
            code="TL0802",
        )
    if wrt is None:
        if len(source.params) != 1:
            raise TLDiffError(
                f"ambiguous differentiation target: operator has params "
                f"{list(source.params)} — specify wrt= explicitly",
                code="TL0801",
                details={"params": list(source.params)},
            )
        return [source.params[0]]
    names = [wrt] if isinstance(wrt, str) else list(wrt)
    if not names:
        raise TLDiffError("wrt= needs at least one parameter name",
                          code="TL0801")
    unknown = [n for n in names if n not in source.params]
    if unknown:
        raise TLDiffError(
            f"unknown differentiation target(s) {unknown}; operator params "
            f"are {list(source.params)}",
            code="TL0801",
            details={"unknown": unknown, "params": list(source.params)},
        )
    return names


def _reject_unfitted(source: Operator) -> None:
    """Fail fast on estimators that can never differentiate (unfitted)."""
    found: list[str] = []

    def visit(node: ir.Expr) -> None:
        if isinstance(node, ir.FittedTransform):
            if node.state is None and node.kind not in found:
                found.append(node.kind)
            return
        if isinstance(node, ir.ApplyOp):
            visit_all(node.args)
            _reject_unfitted(node.operator)
            return
        for child in _ad_children(node):
            visit(child)

    def visit_all(nodes) -> None:
        for n in nodes:
            visit(n)

    visit(source.root)
    if found:
        raise TLDiffError(
            f"cannot differentiate unfitted estimator(s) {found}: "
            f"differentiation through fitting is deferred in v0.4 — call "
            f".fit(training_data) first; only fitted application differentiates",
            code="TL0805",
            details={"estimators": found},
        )


def grad(loss, wrt=None) -> GradOperator:
    """Reverse-mode gradient operator: ``d_loss := grad(loss, wrt="P")``.

    Declarative (no execution). The result is an ordinary operator:
    ``d_loss(x)`` is lazy, ``.collect()`` materializes the gradient shaped
    like the selected input. The loss must be scalar-valued (checked when
    the gradient executes, since operators are data-independent).
    """
    if not isinstance(loss, Operator):
        raise TLTypeError(f"grad() needs an operator, got {type(loss).__name__}",
            code="TL0420",
            details={"got": type(loss).__name__})
    _reject_unfitted(loss)
    return GradOperator(loss, _resolve_wrt(loss, wrt))


def value_and_grad(loss, wrt=None):
    """Value and gradient sharing one forward evaluation.

    ``both := value_and_grad(loss, wrt="P")`` then
    ``value, gradient = both(x)``.
    """
    if not isinstance(loss, Operator):
        raise TLTypeError(
            f"value_and_grad() needs an operator, got {type(loss).__name__}",
            code="TL0420",
            details={"got": type(loss).__name__})
    _reject_unfitted(loss)
    return ValueGradOperator(loss, _resolve_wrt(loss, wrt))


def vjp(loss, wrt=None):
    """Vector-Jacobian product: ``df := vjp(loss, wrt="P")`` applied as
    ``df(x, seed)`` with the cotangent appended after data arguments."""
    if not isinstance(loss, Operator):
        raise TLTypeError(f"vjp() needs an operator, got {type(loss).__name__}",
            code="TL0420",
            details={"got": type(loss).__name__})
    _reject_unfitted(loss)
    return VjpOperator(loss, _resolve_wrt(loss, wrt))


# -- executor hooks --------------------------------------------------------------


def evaluate_diffvar(node: ir.DiffVar):
    from ..runtime.executor import evaluate

    return evaluate(node.value)


def evaluate_forward(node: ir.ForwardOf):
    value, kind = node.shared.forward()
    return value, kind


def evaluate_adjoint(node: ir.AdjointOf):
    grads = node.shared.reverse()
    if len(node.indices) != 1:  # pragma: no cover - defensive
        raise TLDiffError("multi-index adjoint is unsupported")
    return grads[node.indices[0]]
