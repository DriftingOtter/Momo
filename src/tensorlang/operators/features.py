"""Feature engineering: one-hot encoding and polynomial expansion."""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import polars as pl

from ..core import expression as ir
from ..core.errors import TLTypeError


def onehot(x, categories=None):
    """One-hot encoding with deterministic (sorted) category order.

    Dense integer vectors → dense indicator matrix; field series (any
    dtype) → frame with one column per category.  ``categories`` pins
    coverage and order; unknown values raise instead of being dropped.
    """
    from ._lift import lift_unary_operand

    if categories is not None:
        cats = tuple(categories)
        if not cats:
            raise TLTypeError("onehot needs at least one category")
        try:
            hash(cats)
        except TypeError:
            raise TLTypeError("onehot categories must be hashable") from None
    else:
        cats = None
    return lift_unary_operand(x, lambda root: ir.OneHot(root, cats))


def polynomial(x, degree: int = 2, include_bias: bool = True):
    """Polynomial features up to ``degree`` (sklearn-compatible ordering:
    bias first, then degree-major combinations with replacement)."""
    from ._lift import lift_unary_operand

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise TLTypeError(f"polynomial needs an integer degree >= 1, got {degree!r}")
    return lift_unary_operand(
        x, lambda root: ir.Polynomial(root, degree, bool(include_bias)))


# -- evaluation -----------------------------------------------------------------


def evaluate_onehot(value, kind: str, categories):
    if kind == "series":
        return _series_onehot(value, categories)
    if kind == "dense":
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise TLTypeError("dense onehot needs a 1-D vector of codes")
        if arr.dtype.kind not in ("i", "u", "b"):
            raise TLTypeError(
                f"dense onehot needs integer codes, got {arr.dtype}; use a "
                f"field series for labeled categories"
            )
        cats = _resolve_cats([int(v) for v in np.unique(arr)], categories, "onehot")
        out = np.zeros((len(arr), len(cats)), dtype=np.float64)
        index_of = {c: j for j, c in enumerate(cats)}
        unknown = sorted({int(v) for v in arr} - set(index_of))
        if unknown:
            raise TLTypeError(f"onehot found unknown categories {unknown}")
        for i, v in enumerate(arr):
            out[i, index_of[int(v)]] = 1.0
        return out, "dense"
    raise TLTypeError("onehot needs a 1-D vector or a field series")


def _resolve_cats(observed: list, categories, name: str) -> list:
    if categories is None:
        return sorted(observed)
    cats = list(categories)
    unknown = [v for v in observed if v not in cats]
    if unknown:
        raise TLTypeError(f"{name} found unknown categories {unknown}")
    return cats


def _series_onehot(series: pl.Series, categories):
    if series.dtype == pl.Categorical:
        series = series.cast(pl.String)
    observed = sorted({v for v in series.to_list() if v is not None},
                      key=repr)
    if series.has_nulls():
        raise TLTypeError(
            "onehot cannot encode missing values; handle nulls explicitly first"
        )
    cats = _resolve_cats(observed, categories, "onehot")
    cols = {}
    for cat in cats:
        cols[str(cat)] = (series == cat).to_list()
    return pl.DataFrame(cols), "frame"


def evaluate_polynomial(value, kind: str, degree: int, include_bias: bool):
    if kind == "series":
        matrix = np.asarray(value.to_list(), dtype=np.float64).reshape(-1, 1)
        names = [value.name or "x"]
    elif kind == "dense":
        arr = np.asarray(value)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise TLTypeError("polynomial needs 1-D or 2-D dense data")
        if arr.dtype.kind not in ("i", "u", "f", "b"):
            raise TLTypeError(f"polynomial needs numeric data, got {arr.dtype}")
        matrix = arr.astype(np.float64)
        names = [f"x{i}" for i in range(matrix.shape[1])]
    elif kind == "frame":
        from ..backends import polars_backend as _pl_be

        numeric = _pl_be.numeric_columns(value)
        if not numeric:
            raise TLTypeError("polynomial has no applicable numerical fields")
        if len(numeric) != value.width:
            raise TLTypeError(
                "polynomial needs an unambiguous numeric feature selection; "
                f"non-numeric fields present — select explicitly, e.g. "
                f"data[{numeric}]"
            )
        matrix = value.select(numeric).to_numpy().astype(np.float64)
        names = list(numeric)
    else:
        raise TLTypeError("polynomial needs dense data or a structured selection")
    cols, labels = _poly_matrix(matrix, names, degree, include_bias)
    if kind == "frame":
        return pl.DataFrame({lab: cols[:, j].tolist()
                             for j, lab in enumerate(labels)}), "frame"
    return cols, "dense"


def _poly_matrix(matrix: np.ndarray, names: list, degree: int, include_bias: bool):
    n, d = matrix.shape
    cols = [np.ones(n)] if include_bias else []
    labels = ["1"] if include_bias else []
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            part = np.ones(n)
            for j in combo:
                part = part * matrix[:, j]
            cols.append(part)
            if len(set(combo)) == 1:
                labels.append(names[combo[0]] if deg == 1 else f"{names[combo[0]]}^{deg}")
            else:
                labels.append(" ".join(names[j] for j in combo))
    return np.column_stack(cols), labels
