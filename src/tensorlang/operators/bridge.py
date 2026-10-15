"""Lazy structured-to-dense bridge (v0.7 M1).

Pure conversion logic behind ``ir.ToDense``: schema validation, dtype
resolution, null policies, memory estimation, and the introspection
record. No execution machinery here — the executor calls
:func:`convert_frame`; planning calls :func:`describe`.
"""
from __future__ import annotations

import time

import numpy as np

SUPPORTED_POLICIES = ("error", "nan")

# Last bridge execution record (timings/actuals for profile()).
LAST_BRIDGE: dict = {}


def validate_columns(frame, columns: tuple) -> dict:
    """Check existence and dtype support against a materialized frame.

    Returns {"dtypes": [...], "null_counts": [...]}. Raises TL0418 for
    unknown columns or unsupported (non-numeric, non-Boolean) dtypes.
    """
    from ..core.errors import TLTypeError as _TLTypeError

    schema = frame.schema
    dtypes, nulls = [], []
    for name in columns:
        if name not in schema:
            raise _TLTypeError(
                f"to_dense() unknown column '{name}'; available: "
                f"{sorted(schema)}",
                code="TL0418",
                details={"column": name,
                         "available": sorted(str(k) for k in schema)})
        dtype = schema[name]
        if not (dtype.is_numeric() or str(dtype) == "Boolean"):
            raise _TLTypeError(
                f"to_dense() column '{name}' has unsupported dtype "
                f"{dtype}; select numeric columns (categorical/string "
                f"columns need explicit encoding first)",
                code="TL0418",
                details={"column": name, "dtype": str(dtype)})
        dtypes.append(dtype)
    counts = frame.select(list(columns)).null_count().row(0)
    return {"dtypes": [str(d) for d in dtypes],
            "null_counts": list(counts)}


def check_null_policy(report: dict, columns: tuple, nulls: str) -> None:
    """Enforce nulls='error' (fail on any null, naming columns)."""
    if nulls != "error":
        return
    from ..core.errors import TLTypeError as _TLTypeError

    bad = [name for name, count in zip(columns, report["null_counts"])
           if count]
    if bad:
        raise _TLTypeError(
            f"to_dense(nulls='error') found missing values in columns "
            f"{bad}; use nulls='nan' or impute explicitly",
            code="TL0418",
            details={"columns": bad, "policy": nulls})


def resolve_dtype(dtypes: list) -> str:
    """Planning-time dtype prediction (documented): float64 if any float
    column is present, else int64-or-float64 depending on null presence
    (resolvable only at execution — reported as uncertain, never
    guessed). Mirrors the NumPy promotion used at conversion time."""
    names = [str(d).lower() for d in dtypes]
    if any("float" in name for name in names):
        return "float64"
    if any("boolean" in name or "int" in name for name in names):
        return "int64-or-float64 (null-dependent)"
    return "float64"


def _np_equiv(name: str) -> str:
    if "boolean" in name:
        return "bool"
    if "int" in name:
        bits = "".join(c for c in name if c.isdigit()) or "64"
        return ("uint" if name.startswith("u") else "int") + bits
    return "float64"


def convert_frame(frame, columns: tuple, nulls: str):
    """Select, cast, and stack columns into a dense float/int array.

    Returns (array, report). Boolean columns bearing nulls are cast to
    Float64 explicitly (polars would otherwise yield object arrays);
    everything else converts per column with NumPy promotion across the
    stack. No rows are dropped, ever.
    """
    import polars as _pl

    report = validate_columns(frame, columns)
    check_null_policy(report, columns, nulls)
    selected = frame.select(list(columns))
    cols = []
    for name, dtype, nuls in zip(columns, report["dtypes"],
                                 report["null_counts"]):
        series = selected[name]
        if "boolean" in dtype.lower() and nuls:
            series = series.cast(_pl.Float64)
        cols.append(series.to_numpy())
    start = time.perf_counter()
    out = np.asarray(np.column_stack(cols))
    convert_ms = (time.perf_counter() - start) * 1000.0
    return np.asarray(out), {"report": report, "convert_ms": convert_ms,
                             "dtype": str(out.dtype),
                             "shape": [int(d) for d in out.shape]}


def estimate_bytes(rows: int | None, ncols: int, itemsize: int = 8) -> dict:
    """Conservative conversion estimate (documented heuristic): dense
    bytes plus one to_numpy copy plus one ownership copy (3x). Unknown
    rows stay explicitly unknown — never zero."""
    if rows is None:
        return {"dense_bytes": None, "estimated_peak_bytes": None,
                "uncertainty": "unknown row count"}
    dense = rows * ncols * itemsize
    return {"dense_bytes": dense, "estimated_peak_bytes": 3 * dense,
            "uncertainty": "modeled (frame + to_numpy copy + ownership)"}


def record(report: dict) -> None:
    global LAST_BRIDGE
    LAST_BRIDGE = dict(report)


def reset() -> None:
    global LAST_BRIDGE
    LAST_BRIDGE = {}


def schema_of(root) -> dict | None:
    """Best-effort column schema for a ToDense root (metadata only:
    LazyFrame schema or scan footer; never data rows)."""
    try:
        from ..core import expression as _ir
        from ..runtime import polars_plan as _lower

        node = root
        if not isinstance(node, _ir.ToDense):
            return None
        ctx = _lower._LowerContext()
        frame = _lower.lower_frame(node.operand, ctx)
        schema = frame.collect_schema()
        return {c: str(schema[c]) for c in node.columns if c in schema}
    except Exception:
        return None


def known_rows(root) -> tuple[int | None, bool]:
    """Known output row count without executing (materialized frames)."""
    try:
        from ..core import expression as _ir

        node = root if isinstance(root, _ir.ToDense) else None
        operand = node.operand if node is not None else root
        if isinstance(operand, _ir.Literal) and operand.kind == "frame":
            return int(operand.value.height), True
    except Exception:
        pass
    return None, False


def evaluate(node, evaluate):
    """Execute a ToDense node: materialize the frame, admit the
    conversion, convert, and wrap. Returns (ndarray, "dense").

    Admission model (documented heuristic): rows are known only after
    the operand materializes, so the budget gates the conversion step
    (dense bytes + to_numpy copy + ownership copy = 3x) with measured
    numbers — never estimated from unknown cardinality. Planning text
    always labels unknown rows as unknown.
    """
    import time as _time

    from ..core.errors import TLTypeError as _TLTypeError
    from ..runtime import chunk_plan as _chunk

    t_materialize = _time.perf_counter()
    frame, kind = evaluate(node.operand)
    materialize_ms = (_time.perf_counter() - t_materialize) * 1000.0
    if kind != "frame":
        raise _TLTypeError(
            f"to_dense() needs a structured frame input, got {kind}; "
            f"pass scan/filter/join/group results (not dense data)",
            code="TL0418",
            details={"kind": kind, "columns": list(node.columns)})
    import polars as _pl

    if not isinstance(frame, _pl.DataFrame):
        raise _TLTypeError(  # pragma: no cover - defensive
            f"to_dense() needs a materialized frame, got "
            f"{type(frame).__name__}",
            code="TL0418", details={"columns": list(node.columns)})
    rows, ncols = frame.height, len(node.columns)
    report = validate_columns(frame, node.columns)
    check_null_policy(report, node.columns, node.nulls)
    est = estimate_bytes(rows, ncols)
    budget = _chunk.MEMORY_BUDGET_BYTES
    if budget is not None and est["estimated_peak_bytes"] > budget:
        from ..core.errors import TLUnsupportedError as _TLU

        raise _TLU(
            f"to_dense() conversion needs ~{est['estimated_peak_bytes'] / 1e6:.1f}MB "
            f"working memory (budget {budget / 1e6:.1f}MB): select fewer "
            f"columns or raise the budget",
            code="TL1020",
            details={"operation": "to_dense",
                     "estimated_peak_bytes": est["estimated_peak_bytes"],
                     "budget_bytes": budget,
                     "rows": rows, "columns": list(node.columns)})
    array, conv = convert_frame(frame, node.columns, node.nulls)
    record({"materialize_ms": materialize_ms,
            "convert_ms": conv["convert_ms"],
            "rows": rows, "cols": ncols, "dtype": conv["dtype"],
            "dense_mb": est["dense_bytes"] / 1e6,
            "budget_mb": budget / 1e6 if budget is not None else None,
            "unknown": False})
    return array, "dense"


def describe(columns: tuple, nulls: str, schema: dict | None,
            rows: int | None) -> str:
    """Read-only bridge plan text (never reads data rows)."""
    lines = ["conversion: structured frame -> dense tensor (lazy)"]
    lines.append(f"  columns ({len(columns)}): {list(columns)}")
    if schema is not None:
        lines.append("  dtypes: " + ", ".join(
            f"{c}={schema.get(c, '?')}" for c in columns))
    else:
        lines.append("  dtypes: unknown until execution (schema unreadable)")
    lines.append(f"  nulls: '{nulls}'"
                 + (" (fail on any missing value)" if nulls == "error"
                    else " (missing as NaN; nullable-int promotes to float64, "
                         "Boolean-with-null casts to float64)"))
    lines.append(f"  rows: {rows if rows is not None else 'unknown '
                '(relational ops can change cardinality)'}")
    est = estimate_bytes(rows, len(columns))
    if est["dense_bytes"] is None:
        lines.append("  estimate: unknown row count — admission at execution "
                     "with measured rows (never treated as zero)")
    else:
        lines.append(f"  estimate: ~{est['dense_bytes'] / 1e6:.2f}MB dense, "
                     f"~{est['estimated_peak_bytes'] / 1e6:.2f}MB working "
                     f"({est['uncertainty']})")
    lines.append("  guarantee scope: the budget gates conversion working "
                 "memory AFTER the upstream frame materializes; upstream "
                 "collection itself is not bounded by this policy")
    return "\n".join(lines)
