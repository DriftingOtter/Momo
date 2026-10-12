"""Polars backend: structured (DataFrame / Series) execution primitives."""
from __future__ import annotations

import polars as pl

from ..core.errors import TLShapeError, TLTypeError, TLUnsupportedError


def numeric_columns(df: pl.DataFrame) -> list[str]:
    return [n for n, d in df.schema.items() if d.is_numeric()]


def boolean_columns(df: pl.DataFrame) -> list[str]:
    return [n for n, d in df.schema.items() if d == pl.Boolean]


def _require_frame(value, op: str) -> pl.DataFrame:
    if not isinstance(value, pl.DataFrame):
        raise TLTypeError(
            f"operator '{op}' on structured data requires a structured tensor"
        )
    return value


def _bool_frame(df: pl.DataFrame) -> pl.DataFrame:
    out = df.select(boolean_columns(df))
    if out.width == 0:
        raise TLTypeError("logical operation requires Boolean fields")
    return out


def _combine_bool_frames(op: str, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    lcols = boolean_columns(left)
    rcols = boolean_columns(right)
    if set(lcols) != set(rcols):
        raise TLShapeError(
            f"cannot combine Boolean results with different fields "
            f"{lcols} vs {rcols}: combine field-level predicates explicitly"
        )
    exprs = [
        (pl.col(c + "_l") & pl.col(c + "_r")).alias(c) if op == "and"
        else (pl.col(c + "_l") | pl.col(c + "_r")).alias(c)
        for c in lcols
    ]
    l = left.select(lcols).rename({c: c + "_l" for c in lcols})
    r = right.select(rcols).rename({c: c + "_r" for c in rcols})
    if l.height != r.height:
        raise TLShapeError(
            f"cannot combine Boolean results with different lengths "
            f"{l.height} vs {r.height}"
        )
    return pl.concat([l, r], how="horizontal").select(exprs)


def _series_op(op: str, left: pl.Series, right) -> pl.Series:
    ops = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        "pow": lambda a, b: a**b, "mod": lambda a, b: a % b,
        "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b, "lt": lambda a, b: a < b,
        "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b,
        "and": lambda a, b: a & b, "or": lambda a, b: a | b,
    }
    try:
        fn = ops[op]
    except KeyError:
        raise TLTypeError(f"unknown binary operator '{op}'") from None
    try:
        if isinstance(right, pl.Series):
            if len(left) != len(right):
                raise TLShapeError(
                    f"cannot combine series of length {len(left)} and {len(right)}",
                    code="TL0504",
                    details={"lengths": [len(left), len(right)]},
                )
            return fn(left, right)
        return fn(left, right)
    except (pl.exceptions.InvalidOperationError, TypeError) as exc:
        raise TLTypeError(f"cannot apply '{op}' to series: {exc}") from exc


def _frame_scalar_op(op: str, df: pl.DataFrame, scalar, reflected: bool = False) -> pl.DataFrame:
    cols = numeric_columns(df)
    if not cols:
        raise TLTypeError(
            f"operator '{op}' has no applicable numerical fields "
            f"in schema {list(df.schema)}"
        )
    out = df.clone()
    for c in cols:
        s = df[c]
        try:
            out = out.with_columns(
                _series_op(op, scalar, s).alias(c) if reflected
                else _series_op(op, s, scalar).alias(c)
            )
        except TLTypeError as exc:
            raise TLTypeError(f"field '{c}': {exc}") from exc
    return out


def apply_binop(op: str, left, right):
    """Structured binary semantics.

    - frame op scalar: transform applicable numerical fields, preserve the
      rest (names, order, non-numeric fields).
    - frame op frame: operate on shared numerical columns, preserve
      non-numeric fields from the left operand.
    - series op series/scalar: elementwise.
    - comparisons produce Boolean frames / series (never silently merged
      into a record condition).
    """
    comparisons = {"eq", "ne", "gt", "lt", "ge", "le"}
    logicals = {"and", "or"}

    if isinstance(left, pl.Series) or isinstance(right, pl.Series):
        if isinstance(left, pl.Series) and isinstance(right, pl.Series):
            return _series_op(op, left, right)
        series = left if isinstance(left, pl.Series) else right
        other = right if isinstance(left, pl.Series) else left
        if isinstance(other, (pl.DataFrame,)):
            raise TLTypeError(
                f"cannot combine a field series with a whole structured tensor "
                f"for '{op}': select matching fields explicitly"
            )
        if reflected := not isinstance(left, pl.Series):
            if op in comparisons or op in logicals:
                raise TLTypeError(f"unsupported reflected operator '{op}'")
            return _series_op(op, _const_series(other, len(series)), series)
        return _series_op(op, series, other)

    if isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame):
        if op in logicals:
            return _combine_bool_frames(op, _bool_frame(left), _bool_frame(right))
        lnum = numeric_columns(left)
        if not lnum:
            raise TLTypeError(
                f"operator '{op}' has no applicable numerical fields"
            )
        if op in comparisons:
            common = [c for c in lnum if c in numeric_columns(right)]
            if not common:
                raise TLTypeError(
                    f"comparison '{op}' has no shared numerical fields"
                )
            if left.height != right.height:
                raise TLShapeError(
                    f"cannot compare frames with {left.height} and "
                    f"{right.height} rows"
                )
            return left.select([
                _series_op(op, left[c], right[c]).alias(c) for c in common
            ])
        common = [c for c in lnum if c in numeric_columns(right)]
        if not common:
            raise TLTypeError(
                f"operator '{op}' has no shared numerical fields"
            )
        if left.height != right.height:
            raise TLShapeError(
                f"cannot combine frames with {left.height} and "
                f"{right.height} rows"
            )
        out = left.clone()
        for c in common:
            out = out.with_columns(_series_op(op, left[c], right[c]).alias(c))
        return out

    if isinstance(left, pl.DataFrame):
        if op in logicals:
            raise TLTypeError(
                f"logical '{op}' needs Boolean operands; got a structured tensor"
            )
        if op in comparisons:
            cols = numeric_columns(left)
            if not cols:
                raise TLTypeError(
                    f"comparison '{op}' has no applicable numerical fields"
                )
            return left.select([_series_op(op, left[c], right).alias(c) for c in cols])
        return _frame_scalar_op(op, left, right)

    if isinstance(right, pl.DataFrame):
        if op in comparisons or op in logicals:
            raise TLTypeError(f"unsupported reflected operator '{op}'")
        return _frame_scalar_op(op, right, left, reflected=True)

    raise TLTypeError(f"unsupported operands for '{op}'")  # pragma: no cover


def _const_series(value, length: int) -> pl.Series:
    return pl.Series([value] * length)


def apply_unop(op: str, operand):
    if isinstance(operand, pl.Series):
        if op == "neg":
            return -operand
        if op == "not":
            return ~operand
        raise TLTypeError(f"unsupported series operator '{op}'")
    df = _require_frame(operand, op)
    if op == "not":
        return _bool_frame(df).__invert__()
    if op == "neg":
        cols = numeric_columns(df)
        if not cols:
            raise TLTypeError("negation has no applicable numerical fields")
        out = df.clone()
        for c in cols:
            out = out.with_columns((-df[c]).alias(c))
        return out
    raise TLTypeError(f"unsupported structured operator '{op}'")


def apply_math_func(func: str, arg):
    import math

    table = {
        "sin": "sin", "cos": "cos", "tan": "tan",
        "exp": "exp", "log": "log", "sqrt": "sqrt",
    }
    name = table[func]
    if isinstance(arg, pl.Series):
        vals = arg.to_list()
        fn = getattr(math, name)
        try:
            return pl.Series([fn(v) if v is not None else None for v in vals])
        except (TypeError, ValueError) as exc:
            raise TLTypeError(f"cannot apply '{func}': {exc}") from exc
    df = _require_frame(arg, func)
    cols = numeric_columns(df)
    if not cols:
        raise TLTypeError(f"'{func}' has no applicable numerical fields")
    out = df.clone()
    for c in cols:
        vals = df[c].to_list()
        fn = getattr(math, name)
        try:
            new = [fn(v) if v is not None else None for v in vals]
        except (TypeError, ValueError) as exc:
            raise TLTypeError(f"field '{c}': cannot apply '{func}': {exc}") from exc
        out = out.with_columns(pl.Series(new).alias(c))
    return out


def apply_reduce(op: str, value, by=None):
    """Whole-frame reductions return a one-row frame preserving labels."""
    agg = {"sum": "sum", "mean": "mean", "min": "min", "max": "max",
           "std": "std", "var": "var"}[op]
    if isinstance(value, pl.Series):
        return getattr(value, agg)()
    df = _require_frame(value, op)
    if by is not None:
        return _grouped(op, df, by)
    cols = numeric_columns(df)
    if not cols:
        raise TLTypeError(f"reduction '{op}' has no applicable numerical fields")
    return df.select([getattr(pl.col(c), agg)().alias(c) for c in cols])


def grouped_series(op: str, series: pl.Series, key: pl.Series) -> pl.DataFrame:
    """Group a single field's values by a per-row key (e.g. ``net.sum(by=region)``)."""
    import numpy as _np

    if not isinstance(key, pl.Series):
        arr = _np.asarray(key)
        key = pl.Series(arr.tolist())
    if len(key) != len(series):
        raise TLShapeError(
            f"group key has length {len(key)} but series has {len(series)} values"
        )
    value_name = series.name or "value"
    key_name = key.name or "group"
    agg = {"sum": "sum", "mean": "mean", "min": "min", "max": "max",
           "std": "std", "var": "var"}[op]
    tmp = pl.DataFrame({"__tl_key__": key, "__tl_val__": series})
    return (
        tmp.group_by("__tl_key__")
        .agg(getattr(pl.col("__tl_val__"), agg)().alias(value_name))
        .rename({"__tl_key__": key_name})
        .sort(key_name)
    )


def _grouped(op: str, df: pl.DataFrame, by) -> pl.DataFrame:
    import numpy as _np

    if isinstance(by, str):
        if by not in df.columns:
            raise TLTypeError(f"unknown group field '{by}'")
        key = df[by]
        key_name = by
    elif isinstance(by, pl.Series):
        key, key_name = by, by.name or "group"
        if len(key) != df.height:
            raise TLShapeError(
                f"group key has length {len(key)} but frame has {df.height} rows"
            )
    else:
        arr = _np.asarray(by)
        if arr.ndim != 1 or len(arr) != df.height:
            raise TLShapeError("group key must be one value per row")
        key, key_name = pl.Series(arr.tolist()), "group"
    cols = [c for c in numeric_columns(df) if c != key_name]
    if not cols:
        raise TLTypeError(f"grouped '{op}' has no applicable numerical fields")
    tmp = df.with_columns(key.alias("__tl_group__"))
    agg = {"sum": "sum", "mean": "mean", "min": "min", "max": "max",
           "std": "std", "var": "var"}[op]
    out = tmp.group_by("__tl_group__").agg(
        [getattr(pl.col(c), agg)().alias(c) for c in cols]
    ).rename({"__tl_group__": key_name}).sort(key_name)
    return out


def field_access(value, field_name: str) -> pl.Series:
    df = _require_frame(value, f".{field_name}")
    if field_name not in df.columns:
        raise TLTypeError(
            f"unknown field '{field_name}'; available fields: {df.columns}"
        )
    return df[field_name]


def getitem(value, index):
    import numpy as _np

    if isinstance(value, pl.Series):
        if isinstance(index, str):
            raise TLTypeError(f"cannot select field '{index}' from a series")
        idx = _np.asarray(index.to_list()) if isinstance(index, pl.Series) else _np.asarray(index)
        if idx.dtype == bool:
            if len(idx) != len(value):
                raise TLShapeError(
                    f"Boolean mask of length {len(idx)} does not match "
                    f"series of length {len(value)}"
                )
            return value.filter(pl.Series(idx.tolist()))
        if idx.dtype.kind in ("i", "u"):
            return value.gather(idx.tolist())
        if isinstance(index, slice):
            return value[index]
        if isinstance(index, (int, _np.integer)):
            return value[int(index)]
        raise TLTypeError(f"unsupported series index {index!r}")
    df = _require_frame(value, "[]")
    if isinstance(index, str):
        return field_access(df, index)
    if isinstance(index, list) and all(isinstance(x, str) for x in index):
        missing = [x for x in index if x not in df.columns]
        if missing:
            raise TLTypeError(f"unknown fields {missing}; have {df.columns}")
        return df.select(index)
    if isinstance(index, tuple) and all(isinstance(x, str) for x in index):
        return getitem(df, list(index))
    if isinstance(index, slice):
        return df[index]
    if isinstance(index, (int,)):
        return df[index : index + 1]
    mask = _np.asarray(index.to_list()) if isinstance(index, pl.Series) else _np.asarray(index)
    if mask.dtype == bool:
        if mask.ndim == 2 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.ndim != 1 or len(mask) != df.height:
            raise TLShapeError(
                f"Boolean mask of shape {mask.shape} does not match "
                f"{df.height} structured records",
                code="TL0504",
                details={"mask": str(tuple(mask.shape)), "rows": df.height},
            )
        return df.filter(pl.Series(mask.tolist()))
    if mask.dtype.kind in ("i", "u"):
        if mask.ndim != 1:
            raise TLShapeError("row indices must be one-dimensional")
        return df.gather(mask.tolist())
    raise TLTypeError(f"unsupported structured index {index!r}")


def set_field(df: pl.DataFrame, name: str, values) -> pl.DataFrame:
    import numpy as _np

    _require_frame(df, "field assignment")
    if isinstance(values, pl.Series):
        series = values
    elif isinstance(values, pl.DataFrame):
        if values.width != 1:
            raise TLShapeError(
                f"cannot assign {values.width} fields to single field '{name}'"
            )
        series = values.to_series()
    else:
        arr = _np.asarray(values)
        if arr.ndim == 0:
            series = pl.Series([arr.item()] * df.height)
        else:
            if len(arr) != df.height:
                raise TLShapeError(
                    f"cannot assign {len(arr)} values to {df.height} rows"
                )
            series = pl.Series(arr.tolist())
    if len(series) != df.height:
        raise TLShapeError(
            f"cannot assign {len(series)} values to {df.height} rows",
            code="TL0505",
            details={"values": len(series), "rows": df.height},
        )
    return df.with_columns(series.alias(name))


def concat_frames(parts: list, how: str = "vertical"):
    try:
        return pl.concat(parts, how=how)
    except Exception as exc:
        raise TLShapeError(f"cannot concatenate structured tensors: {exc}") from exc


def transpose(value) -> pl.DataFrame:
    raise TLUnsupportedError(
        "transpose is undefined for heterogeneous structured tensors: "
        "select numerical fields explicitly"
    )


def reshape(value, shape: tuple[int, ...]):
    raise TLUnsupportedError(
        "reshape is undefined for heterogeneous structured tensors"
    )


def is_null(value):
    if isinstance(value, pl.Series):
        return value.is_null()
    df = _require_frame(value, "is_null")
    return df.select([pl.col(c).is_null().alias(c) for c in df.columns])


def not_null(value):
    if isinstance(value, pl.Series):
        return value.is_not_null()
    df = _require_frame(value, "not_null")
    return df.select([pl.col(c).is_not_null().alias(c) for c in df.columns])


def fill_null(value, fill):
    if isinstance(value, pl.Series):
        return value.fill_null(fill)
    df = _require_frame(value, "fill_null")
    return df.with_columns([pl.col(c).fill_null(fill) for c in df.columns])


def drop_null(value):
    if isinstance(value, pl.Series):
        return value.drop_nulls()
    return _require_frame(value, "drop_null").drop_nulls()
