"""Abstract operator/tensor summaries: one structured, JSON-serializable
introspection API shared by the REPL, the error analyser, `tensorlang
graph`, and future editor tooling (instead of three separate formatters).
"""
from __future__ import annotations


def _dtype_str(dtype) -> str:
    return str(dtype)


def operator_abstract(op) -> dict:
    """Structured summary of an operator (never executes)."""
    from ..operators.ml import FittableOperator, PipelineOperator
    from .operator import Operator

    from ..operators.autodiff import _GradBase
    from ..runtime.planner import plan

    if isinstance(op, PipelineOperator):
        stages = []
        for stage in op.stages:
            if isinstance(stage, FittableOperator):
                stages.append({
                    "kind": ("fitted" if stage.is_fitted else "fittable"),
                    "estimator": stage.estimator_kind,
                    "params": list(stage.params),
                })
            else:
                stages.append({"kind": "stateless",
                               "params": list(stage.params)})
        kind = "pipeline"
        extra: dict = {"stages": stages}
    elif isinstance(op, FittableOperator):
        kind = "fitted" if op.is_fitted else "fittable"
        extra = {"estimator": op.estimator_kind, "config": dict(op.config)}
        if op.is_fitted:
            state = op.root.state
            extra["features"] = list(state.feature_names) if state.feature_names else None
            extra["n_features"] = state.n_features
            extra["learned"] = sorted(k for k, _ in state.arrays) + sorted(
                k for k, _ in state.objects)
    elif isinstance(op, _GradBase):
        kind = "gradient"
        extra = {"wrt": list(op.wrt), "source_params": list(op.source.params)}
    elif isinstance(op, Operator):
        kind = "stateless"
        extra = {}
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot abstract {type(op).__name__}")
    graph = plan(op.root)
    return {
        "kind": kind,
        "params": list(op.params),
        "bound_as": op._bound_name,
        "nodes": graph.node_count,
        "backends": graph.summary(),
        **extra,
    }


def tensor_abstract(t) -> dict:
    """Structured summary of a tensor (never reads scan data)."""
    out = {
        "kind": t._kind,
        "provenance": t._provenance,
        "version": t._version,
        "name": t._tl_name,
    }
    if t._kind == "dense":
        out["shape"] = list(t._data.shape)
        out["dtype"] = _dtype_str(t._data.dtype)
    elif t._kind == "frame":
        out["shape"] = [t._data.height, t._data.width]
        out["schema"] = {k: _dtype_str(v) for k, v in t._data.schema.items()}
        out["fields"] = list(t._data.columns)
    elif t._kind == "series":
        out["length"] = len(t._data)
        out["dtype"] = _dtype_str(t._data.dtype)
        out["field"] = t._data.name
    elif t._kind == "scan":
        out["shape"] = None
        out["note"] = "lazy descriptor: shape unknown until collected"
        try:
            schema = t.schema
            out["schema"] = {k: _dtype_str(v) for k, v in schema.items()}
            out["fields"] = list(schema)
        except Exception as exc:  # pragma: no cover - unreadable source
            out["schema_error"] = str(exc)
    elif t._kind == "chunked":
        out["shape"] = None
        out["note"] = "lazy persistent array: metadata only, data untouched"
        try:
            from ..backends import dask_backend as _be

            spec = _be.describe(t._data)
            out["shape"] = list(spec.shape)
            out["dtype"] = spec.dtype
            out["storage"] = "zarr"
            out["source"] = spec.path
            out["chunks"] = list(spec.chunks)
            out["partitions"] = spec.npartitions
            out["nbytes"] = spec.nbytes
            out["chunk_nbytes"] = spec.chunk_nbytes
            out["materialized"] = False
        except Exception as exc:
            out["metadata_error"] = str(exc)
    return out


def lazy_abstract(lz) -> dict:
    """Structured summary of a pending computation (never executes)."""
    from ..runtime.planner import plan

    graph = plan(lz._root)
    target = lz._target
    try:
        from ..runtime.chunk_plan import current_policy

        memory_policy = current_policy()
    except Exception:  # pragma: no cover - defensive
        memory_policy = {"memory_budget_bytes": None, "max_workers": None}
    return {
        "nodes": graph.node_count,
        "backends": graph.summary(),
        "target": ({
            "kind": target._kind,
            "name": target._tl_name,
        } if target is not None else None),
        "cached": lz._has_cached,
        "memory_policy": memory_policy,
    }
