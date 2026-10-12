"""Elementary mathematics: sin, cos, tan, exp, log, sqrt."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..backends import numpy_backend as _np_be
from ..backends import polars_backend as _pl_be
from ..core import expression as ir
from ..core.errors import TLTypeError
from ._lift import lift_unary_operand


def _make(name: str, scalar_fn):
    def func(x):
        return lift_unary_operand(
            x,
            lambda root: ir.FuncCall(name, (root,)),
            const_build=lambda v: scalar_fn(v),
        )

    func.__name__ = name
    func.__doc__ = f"Elementwise {name} (tensor, lazy, operator or scalar)."
    return func


sin = _make("sin", lambda v: math.sin(v))
cos = _make("cos", lambda v: math.cos(v))
tan = _make("tan", lambda v: math.tan(v))
exp = _make("exp", lambda v: math.exp(v))
log = _make("log", lambda v: math.log(v))
sqrt = _make("sqrt", lambda v: math.sqrt(v))


def apply(func: str, arg):
    """Evaluate an elementary function over a materialized value."""
    if isinstance(arg, np.ndarray):
        return _np_be.apply_math_func(func, arg), "dense"
    if isinstance(arg, (pl.Series, pl.DataFrame)):
        return _pl_be.apply_math_func(func, arg), (
            "series" if isinstance(arg, pl.Series) else "frame"
        )
    table = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    }
    try:
        return table[func](arg), "scalar"
    except (TypeError, ValueError) as exc:
        raise TLTypeError(f"cannot apply '{func}': {exc}") from exc
