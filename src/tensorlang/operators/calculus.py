"""Symbolic calculus: differentiation, simplification, exact integration."""
from __future__ import annotations

from ..backends import sympy_backend as _sym
from ..core import expression as ir
from ..core.errors import TLSymbolicError, TLTypeError


def diff(op):
    """Symbolic derivative of a single-parameter operator: ``df := diff(f)``."""
    from ..core.operator import Operator

    if not isinstance(op, Operator):
        raise TLTypeError(f"diff() needs an operator, got {type(op).__name__}")
    uid = _fresh()
    root = ir.remap_params(
        _sym.differentiate(op.root, op.uid, op.params), uid
    )
    return Operator(op.params, root, uid)


def simplify(op):
    """Symbolic simplification of a mathematical operator."""
    from ..core.operator import Operator

    if not isinstance(op, Operator):
        raise TLTypeError(f"simplify() needs an operator, got {type(op).__name__}")
    uid = _fresh()
    root = ir.remap_params(_sym.simplify(op.root, op.uid, op.params), uid)
    return Operator(op.params, root, uid)


def integrate(op, lo=None, hi=None):
    """Exact integration of a single-parameter operator.

    ``integrate(f, a, b)`` returns a zero-parameter constant operator holding
    the exact symbolic result (e.g. ``125/3``).  ``integrate(f)`` returns the
    antiderivative as a unary operator.  No numerical fallback is applied
    implicitly: inexact integrals raise.
    """
    from ..core.operator import Operator

    if not isinstance(op, Operator):
        raise TLTypeError(
            f"integrate() needs an operator, got {type(op).__name__}"
        )
    if (lo is None) != (hi is None):
        raise TLSymbolicError("integrate() needs both limits or neither")
    if lo is not None and hi is not None:
        value = _sym.definite_integral(op.root, op.uid, op.params, lo, hi)
        return Operator((), ir.Const(value), _fresh())
    uid = _fresh()
    root = ir.remap_params(
        _sym.indefinite_integral(op.root, op.uid, op.params), uid
    )
    return Operator(op.params, root, uid)


def _fresh() -> str:
    from ..core.operator import fresh_uid

    return fresh_uid()
