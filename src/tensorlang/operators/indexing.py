"""Selection and filtering: indexing evaluation and explicit assignment."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..backends import numpy_backend as _np_be
from ..backends import polars_backend as _pl_be
from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError, TLUnsupportedError


def _resolve_index(index: ir.Expr, evaluate):
    """Turn an index IR node into a materialized index value."""
    if isinstance(index, ir.Const):
        return index.value
    value, _ = evaluate(index)
    return value


def apply_index(obj, obj_kind: str, index: ir.Expr, evaluate):
    """Evaluate ``obj[index]``. Returns ``(value, kind)``.

    - dense + int/slice/int-array/bool-mask: NumPy indexing.
    - dense + str/list[str]: descriptive error (no named fields).
    - frame + str: field series; + list[str]: sub-frame; + bool mask: rows.
    - series + int/slice/mask: filtered series.
    """
    idx = _resolve_index(index, evaluate)
    if obj_kind == "dense":
        if isinstance(idx, str) or (
            isinstance(idx, list) and all(isinstance(x, str) for x in idx)
        ):
            raise TLTypeError(
                f"cannot select field {idx!r} from a dense tensor; "
                f"named fields exist only on structured tensors"
            )
        if isinstance(idx, pl.Series):
            idx = np.asarray(idx.to_list())
        result = _np_be.getitem(np.asarray(obj), idx)
        if isinstance(result, np.ndarray):
            return result, "dense"
        if isinstance(result, np.generic):
            return result.item(), "scalar"
        return result, "scalar"
    if obj_kind == "frame":
        if isinstance(idx, pl.Series):
            pass
        elif isinstance(idx, np.ndarray):
            idx = idx.tolist()
        out = _pl_be.getitem(obj, idx)
        if isinstance(out, pl.DataFrame):
            return out, "frame"
        return out, "series"
    if obj_kind == "series":
        if isinstance(idx, str):
            raise TLTypeError(
                f"cannot select field '{idx}' from a single field series"
            )
        if isinstance(idx, np.ndarray):
            idx = idx.tolist()
        out = _pl_be.getitem(obj, idx)
        if isinstance(out, pl.Series):
            return out, "series"
        return out, "scalar"
    # scalar obj[index]: only degenerate 0-d numpy-style indexing
    raise TLTypeError("cannot index a scalar value")


def set_item(current: Any, kind: str | None, key: Any, new_value: Any) -> Any:
    """Explicit assignment target handling. Returns the updated value.

    - frame + str key: add/replace a named field (derived-field assignment).
    - dense + mask/slice/int key: elementwise update on a copy.
    - anything else ambiguous: descriptive error.
    """
    from ..core.tensor import Lazy, Tensor  # local import: no cycle at runtime

    if isinstance(current, Tensor):
        current = current._material()
    if isinstance(new_value, (Tensor, Lazy)):
        raise TLTypeError("internal error: unevaluated value reached assignment")

    if isinstance(current, pl.DataFrame):
        if not isinstance(key, str):
            raise TLUnsupportedError(
                f"cannot assign rows of a structured tensor with index "
                f"{key!r}: filter rows explicitly instead"
            )
        return _pl_be.set_field(current, key, new_value)

    if isinstance(current, pl.Series):
        arr = np.asarray(_as_list(new_value))
        return _series_set(current, key, arr)

    if isinstance(current, np.ndarray):
        if isinstance(key, str):
            raise TLTypeError(
                f"cannot assign field '{key}' on a dense tensor; named fields "
                f"exist only on structured tensors"
            )
        updated = np.array(current, copy=True)
        idx = _numpy_key(key)
        try:
            updated[idx] = _as_assignable(new_value, updated[idx].shape)
        except (IndexError, ValueError) as exc:
            raise TLShapeError(f"assignment failed: {exc}") from exc
        return updated

    raise TLTypeError(f"cannot assign into {type(current).__name__}")


def _as_list(value) -> list:
    if isinstance(value, pl.Series):
        return value.to_list()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _series_set(series: pl.Series, key, arr: np.ndarray) -> pl.Series:
    vals = series.to_list()
    n = len(vals)
    if isinstance(key, slice):
        idx = range(*key.indices(n))
        items = arr.tolist() if arr.ndim else [arr.item()] * len(list(idx))
        idx = list(idx)
        if len(items) != len(idx):
            raise TLShapeError(
                f"cannot assign {len(items)} values to {len(idx)} positions"
            )
        for i, v in zip(idx, items):
            vals[i] = v
        return pl.Series(vals, dtype=series.dtype)
    if isinstance(key, (int, np.integer)):
        i = int(key)
        vals[i] = arr.item() if arr.ndim == 0 else arr.tolist()[0]
        return pl.Series(vals, dtype=series.dtype)
    mask = np.asarray(key.tolist() if isinstance(key, pl.Series) else key)
    if mask.dtype == bool:
        if len(mask) != n:
            raise TLShapeError(
                f"Boolean mask of length {len(mask)} does not match series "
                f"of length {n}",
                code="TL0504",
                details={"mask": len(mask), "length": n},
            )
        items = arr.tolist()
        if len(items) != int(mask.sum()):
            raise TLShapeError("mask assignment length mismatch",
                               code="TL0505")
        it = iter(items)
        vals = [next(it) if m else v for m, v in zip(mask, vals)]
        return pl.Series(vals, dtype=series.dtype)
    raise TLTypeError(f"unsupported series assignment index {key!r}")


def _numpy_key(key):
    if isinstance(key, pl.Series):
        return np.asarray(key.to_list())
    if isinstance(key, np.ndarray):
        return key
    if isinstance(key, (slice, int, tuple)):
        return key
    if isinstance(key, list):
        return np.asarray(key)
    raise TLTypeError(f"unsupported dense assignment index {key!r}")


def _as_assignable(value, shape: tuple[int, ...]):
    if isinstance(value, pl.Series):
        value = np.asarray(value.to_list())
    if isinstance(value, pl.DataFrame):
        raise TLShapeError("cannot assign structured data into a dense tensor")
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise TLShapeError(f"cannot broadcast {arr.shape} to {shape}") from exc
