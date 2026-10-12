"""Execution introspection: logical plans, physical plans, measurements.

``explain()`` shows the TensorLang logical graph and the physical backend
plan without executing.  ``profile()`` executes and reports *measured*
costs only — wall-clock time, allocator and process memory, backend used,
and result shape.  Nothing here is estimated.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import polars as pl

from ..core import expression as ir
from . import io as _io
from .planner import plan as _plan_graph


def format_node(node: ir.Expr, indent: int = 0) -> str:
    """Pretty-print a (possibly open) expression graph."""
    pad = "  " * indent
    if isinstance(node, ir.Param):
        return f"{pad}Param({node.name} @{node.owner})"
    if isinstance(node, ir.Const):
        return f"{pad}Const({node.value!r})"
    if isinstance(node, ir.Literal):
        return f"{pad}Literal({node.kind})"
    if isinstance(node, ir.Scan):
        return f"{pad}{node.spec.describe()}"
    if isinstance(node, ir.BinOp):
        return (f"{pad}{node.op}\n{format_node(node.left, indent + 1)}\n"
                f"{format_node(node.right, indent + 1)}")
    if isinstance(node, ir.UnOp):
        return f"{pad}{node.op}\n{format_node(node.operand, indent + 1)}"
    if isinstance(node, ir.FuncCall):
        inner = "\n".join(format_node(a, indent + 1) for a in node.args)
        return f"{pad}{node.func}()\n{inner}"
    if isinstance(node, ir.FillNull):
        return (f"{pad}fill_null\n{format_node(node.operand, indent + 1)}\n"
                f"{format_node(node.value, indent + 1)}")
    if isinstance(node, ir.FieldAccess):
        return f"{pad}.{node.field}\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.GetItem):
        return (f"{pad}[]\n{format_node(node.obj, indent + 1)}\n"
                f"{format_node(node.index, indent + 1)}")
    if isinstance(node, ir.Transpose):
        return f"{pad}transpose\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Reshape):
        return f"{pad}reshape{list(node.shape)}\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Flatten):
        return f"{pad}flatten\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Concat):
        parts = "\n".join(format_node(p, indent + 1) for p in node.parts)
        return f"{pad}concat(axis={node.axis})\n{parts}"
    if isinstance(node, ir.Stack):
        parts = "\n".join(format_node(p, indent + 1) for p in node.parts)
        return f"{pad}stack(axis={node.axis})\n{parts}"
    if isinstance(node, ir.Reduce):
        text = f"{pad}{node.op}"
        if node.by is not None:
            text += f" by {format_node(node.by).strip()}"
        return text + f"\n{format_node(node.operand, indent + 1)}"
    if isinstance(node, ir.WithField):
        return (f"{pad}with_field[{node.name}]\n"
                f"{format_node(node.obj, indent + 1)}\n"
                f"{format_node(node.value, indent + 1)}")
    if isinstance(node, ir.Join):
        return (f"{pad}join({node.how} on {list(node.on)})\n"
                f"{format_node(node.left, indent + 1)}\n"
                f"{format_node(node.right, indent + 1)}")
    if isinstance(node, ir.SortBy):
        return (f"{pad}sort({list(node.keys)}, descending={node.descending})\n"
                f"{format_node(node.obj, indent + 1)}")
    if isinstance(node, ir.Unique):
        return f"{pad}unique\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Where):
        return (f"{pad}where\n{format_node(node.cond, indent + 1)}\n"
                f"{format_node(node.a, indent + 1)}\n"
                f"{format_node(node.b, indent + 1)}")
    if isinstance(node, ir.Clip):
        return (f"{pad}clip\n{format_node(node.obj, indent + 1)}\n"
                f"{format_node(node.lo, indent + 1)}\n"
                f"{format_node(node.hi, indent + 1)}")
    if isinstance(node, ir.Over):
        text = f"{pad}{node.op} over"
        if node.by is not None:
            text += f" {format_node(node.by).strip()}"
        return text + f"\n{format_node(node.operand, indent + 1)}"
    if isinstance(node, ir.Rolling):
        return (f"{pad}rolling({node.op}, window={node.window})\n"
                f"{format_node(node.operand, indent + 1)}")
    if isinstance(node, ir.FittedTransform):
        state = "fitted" if node.state is not None else "unfitted"
        cfg = ", ".join(f"{k}={v!r}" for k, v in node.config)
        text = f"{pad}{node.kind}({state}, {cfg})\n{format_node(node.arg, indent + 1)}"
        if node.state is not None:
            names = node.state.feature_names
            text += (f"\n{pad}  features={list(names) if names else None}, "
                     f"n={node.state.n_features}")
        return text
    if isinstance(node, ir.SvdPart):
        return f"{pad}svd[{['U', 'S', 'Vh'][node.index]}]"
    if isinstance(node, ir.EighPart):
        return f"{pad}eigh[{['values', 'vectors'][node.index]}]"
    if isinstance(node, ir.Solve):
        return (f"{pad}solve\n{format_node(node.a, indent + 1)}\n"
                f"{format_node(node.b, indent + 1)}")
    if isinstance(node, ir.Norm):
        return (f"{pad}norm(ord={node.ord!r}, axis={node.axis!r})\n"
                f"{format_node(node.obj, indent + 1)}")
    if isinstance(node, ir.Cov):
        return f"{pad}cov\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Corr):
        return f"{pad}corr\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Quantile):
        return (f"{pad}quantile({list(node.q)})\n"
                f"{format_node(node.obj, indent + 1)}")
    if isinstance(node, ir.Einsum):
        parts = "\n".join(format_node(p, indent + 1) for p in node.parts)
        return f"{pad}einsum({node.spec!r})\n{parts}"
    if isinstance(node, ir.Tensordot):
        return (f"{pad}tensordot(axes={node.axes!r})\n"
                f"{format_node(node.a, indent + 1)}\n"
                f"{format_node(node.b, indent + 1)}")
    if isinstance(node, ir.MoveAxis):
        return (f"{pad}moveaxis({node.source} -> {node.destination})\n"
                f"{format_node(node.obj, indent + 1)}")
    if isinstance(node, ir.OneHot):
        return f"{pad}onehot\n{format_node(node.obj, indent + 1)}"
    if isinstance(node, ir.Polynomial):
        return (f"{pad}polynomial(degree={node.degree}, bias={node.include_bias})\n"
                f"{format_node(node.obj, indent + 1)}")
    if isinstance(node, ir.DiffVar):
        return f"{pad}\u2202{node.name}"
    if isinstance(node, ir.AdjointOf):
        return f"{pad}grad[wrt={list(node.shared.wrt_keys)}]"
    if isinstance(node, ir.ForwardOf):
        return f"{pad}value[shared]"
    if isinstance(node, ir.ApplyOp):
        params = ",".join(node.operator.params)
        inner = "\n".join(format_node(a, indent + 1) for a in node.args)
        body = f"\n{inner}" if inner else ""
        return f"{pad}apply({_describe_operator(node.operator)}: {params}){body}"
    return f"{pad}{type(node).__name__}"  # pragma: no cover - defensive


def _describe_operator(op) -> str:
    """One-line operator identity for plans (fitted state included)."""
    from ..operators.ml import FittableOperator, PipelineOperator

    if isinstance(op, PipelineOperator):
        parts = []
        for stage in op.stages:
            if isinstance(stage, FittableOperator):
                status = "fitted" if stage.is_fitted else "unfitted"
                parts.append(f"{stage.estimator_kind}({status})")
            else:
                parts.append(f"operator{list(stage.params)}")
        return "pipeline[" + " >> ".join(parts) + "]"
    if isinstance(op, FittableOperator):
        status = "fitted" if op.is_fitted else "unfitted"
        return f"{op.estimator_kind}({status})"
    name = getattr(op, "_bound_name", None)
    return f"operator '{name}'" if name else "operator"


def _dot_label(node: ir.Expr) -> str:
    if isinstance(node, ir.Param):
        return f"Param {node.name}"
    if isinstance(node, ir.Const):
        text = repr(node.value)
        return f"Const {text[:24]}"
    if isinstance(node, ir.Literal):
        return f"Literal {node.kind}"
    if isinstance(node, ir.Scan):
        return node.spec.describe()
    if isinstance(node, ir.BinOp):
        return node.op
    if isinstance(node, ir.UnOp):
        return node.op
    if isinstance(node, ir.FuncCall):
        return f"{node.func}()"
    if isinstance(node, ir.FieldAccess):
        return f".{node.field}"
    if isinstance(node, ir.Reduce):
        return node.op
    if isinstance(node, ir.ApplyOp):
        return f"apply({_describe_operator(node.operator)})"
    if isinstance(node, ir.FittedTransform):
        state = "fitted" if node.state is not None else "unfitted"
        return f"{node.kind}({state})"
    return type(node).__name__


def to_dot(root: ir.Expr) -> str:
    """Graphviz DOT rendering of an expression graph (deduped by identity)."""
    from .planner import _children

    lines = ["digraph {", '  rankdir="BT";']
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        label = _dot_label(node).replace('"', "'")
        lines.append(f'  N{id(node)} [label="{label}"];')
        children: list[ir.Expr] = []
        if isinstance(node, ir.ApplyOp):
            children = list(node.args)
        elif isinstance(node, (ir.DiffVar,)):
            children = [node.value]
        elif isinstance(node, (ir.AdjointOf, ir.ForwardOf, ir.SvdPart,
                               ir.EighPart)):
            children = []
        else:
            try:
                children = [c for c in _children(node)
                            if isinstance(c, ir.Expr)]
            except Exception:  # pragma: no cover - defensive
                children = []
        for child in children:
            lines.append(f"  N{id(node)} -> N{id(child)};")
            stack.append(child)
    lines.append("}")
    return "\n".join(lines)


def explain_lazy(root: ir.Expr) -> str:
    """Explain a pending computation: logical graph + physical plan."""
    from . import polars_plan as _lower

    logical = format_node(root)
    physical, _ = _lower.explain_plan(root)
    graph = _plan_graph(root)
    lines = [
        "logical plan (TensorLang IR):",
        logical,
        "",
        physical,
        "",
        f"graph: {graph.node_count} nodes; backends {graph.summary()}",
    ]
    return "\n".join(lines)


@dataclass
class ProfileResult:
    """Measured execution costs. Every field is observed, never estimated."""

    elapsed_ms: float
    backend: str  # 'polars-lazy' | 'materialized'
    lowered: bool
    rows: int | None
    cols: int | None
    shape: tuple | None
    result_kind: str
    peak_python_alloc_mb: float
    peak_rss_delta_mb: float
    nodes: int
    scan_executions: int
    plan: str = ""
    forward_ms: float | None = None  # AD forward phase (gradient graphs)
    backward_ms: float | None = None  # AD reverse phase (gradient graphs)

    def __str__(self) -> str:
        shape = self.shape if self.shape is not None else f"{self.rows}x{self.cols}"
        text = (
            f"profile: backend={self.backend} lowered={self.lowered} "
            f"elapsed={self.elapsed_ms:.2f}ms result={self.result_kind}{shape} "
            f"peak_python_alloc={self.peak_python_alloc_mb:.2f}MB "
            f"peak_rss_delta={self.peak_rss_delta_mb:.2f}MB "
            f"nodes={self.nodes} scan_executions={self.scan_executions}"
        )
        if self.forward_ms is not None and self.backward_ms is not None:
            text += (f" forward={self.forward_ms:.2f}ms "
                     f"backward={self.backward_ms:.2f}ms")
        return text


def _rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import platform

        if platform.system() == "Darwin":  # bytes on macOS
            return rss / 1e6
        return rss / 1e3  # kilobytes on Linux
    except Exception:  # pragma: no cover - platform dependent
        return float("nan")


def profile_lazy(root: ir.Expr):
    """Execute ``root`` and measure real costs. Returns ``ProfileResult``."""
    from . import polars_plan as _lower
    from .executor import collect_value, evaluate

    graph = _plan_graph(root)
    scans_before = _io.SCAN_EXECUTIONS
    rss_before = _rss_mb()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        try:
            value, kind, plan_text = _lower.collect_lowered(root)
            backend, lowered = "polars-lazy", True
        except Exception:
            value, kind = evaluate(root)
            plan_text, backend, lowered = "", "materialized", False
        result = collect_value(value, kind)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss_after = _rss_mb()
    rows, cols, shape = _result_shape(result, kind)
    forward_ms, backward_ms = _ad_phases(root)
    return ProfileResult(
        elapsed_ms=elapsed_ms,
        backend=backend,
        lowered=lowered,
        rows=rows,
        cols=cols,
        shape=shape,
        result_kind=kind,
        peak_python_alloc_mb=peak_python / 1e6,
        peak_rss_delta_mb=rss_after - rss_before,
        nodes=graph.node_count,
        scan_executions=_io.SCAN_EXECUTIONS - scans_before,
        plan=plan_text,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
    )


def _ad_phases(root):
    """Forward/backward split for gradient graphs (None otherwise)."""
    from ..core import expression as _ir

    if isinstance(root, (_ir.AdjointOf, _ir.ForwardOf)):
        return root.shared.last_forward_ms, root.shared.last_backward_ms
    return None, None


def _result_shape(result, kind: str):
    if kind == "dense":
        return None, None, tuple(result.shape)
    if kind == "frame":
        return result.height, result.width, None
    if kind == "series":
        return len(result), 1, None
    return None, None, ()
