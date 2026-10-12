"""SymPy backend: symbolic mathematics for operator expressions."""
from __future__ import annotations

import sympy as sp

from ..core import expression as ir
from ..core.errors import TLSymbolicError


def to_sympy(node: ir.Expr, symbols: dict[tuple[str, str], sp.Symbol]) -> sp.Expr:
    """Convert a pure-mathematical operator expression to a SymPy expression.

    Only mathematical nodes over parameters and literal constants are
    convertible. Indexing, field access, reductions and embedded operator
    applications are rejected with a descriptive error.
    """
    if isinstance(node, ir.Param):
        key = (node.owner, node.name)
        if key not in symbols:
            symbols[key] = sp.Symbol(node.name)
        return symbols[key]
    if isinstance(node, ir.Const):
        v = node.value
        if isinstance(v, bool):
            return sp.true if v else sp.false
        if isinstance(v, (int, sp.Integer)):
            return sp.Integer(v)
        if isinstance(v, sp.Basic):
            return v
        if isinstance(v, float):
            return sp.Float(v)
        raise TLSymbolicError(
            f"constant {v!r} has no exact symbolic representation"
        )
    if isinstance(node, ir.Literal):
        raise TLSymbolicError(
            "symbolic differentiation needs a pure mathematical expression; "
            "got embedded tensor data"
        )
    if isinstance(node, ir.BinOp):
        left = to_sympy(node.left, symbols)
        right = to_sympy(node.right, symbols)
        table = {
            "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
            "pow": lambda a, b: a**b, "mod": lambda a, b: a % b,
        }
        if node.op not in table:
            raise TLSymbolicError(
                f"operator '{node.op}' has no symbolic meaning "
                f"(comparisons and logic are not differentiable)"
            )
        return table[node.op](left, right)
    if isinstance(node, ir.UnOp):
        operand = to_sympy(node.operand, symbols)
        if node.op == "neg":
            return -operand
        if node.op == "abs":
            return sp.Abs(operand)
        raise TLSymbolicError(f"operator '{node.op}' has no symbolic meaning")
    if isinstance(node, ir.FuncCall):
        args = [to_sympy(a, symbols) for a in node.args]
        table = {
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        }
        if node.func not in table:
            raise TLSymbolicError(
                f"function '{node.func}' has no symbolic meaning"
            )
        return table[node.func](*args)
    if isinstance(node, (ir.GetItem, ir.FieldAccess, ir.Reduce, ir.ApplyOp,
                         ir.Transpose, ir.Reshape, ir.Flatten, ir.Concat,
                         ir.Stack, ir.FillNull)):
        raise TLSymbolicError(
            f"symbolic mathematics is undefined for '{type(node).__name__}' "
            f"expressions; differentiate pure mathematical operators only"
        )
    raise TLSymbolicError(  # pragma: no cover - defensive
        f"cannot convert '{type(node).__name__}' to a symbolic expression"
    )


def from_sympy(expr: sp.Basic, param_names: set[str], owner: str) -> ir.Expr:
    """Convert a SymPy expression back into TensorLang IR nodes."""
    expr = sp.sympify(expr)
    if expr.is_Symbol:
        if str(expr) not in param_names:
            raise TLSymbolicError(f"unexpected symbol '{expr}' in result")
        return ir.Param(owner, str(expr))
    if expr.is_Number:
        return ir.Const(expr)
    if expr in (sp.true, sp.false):
        return ir.Const(bool(expr))
    if expr.is_Add:
        result: ir.Expr | None = None
        for term in expr.args:
            part = from_sympy(term, param_names, owner)
            result = part if result is None else ir.BinOp("add", result, part)
        assert result is not None
        return result
    if expr.is_Mul:
        result = None
        for factor in expr.args:
            part = from_sympy(factor, param_names, owner)
            result = part if result is None else ir.BinOp("mul", result, part)
        assert result is not None
        return result
    if expr.is_Pow:
        base, exp = expr.args
        return ir.BinOp("pow", from_sympy(base, param_names, owner),
                        from_sympy(exp, param_names, owner))
    if expr.func is sp.Abs:
        return ir.UnOp("abs", from_sympy(expr.args[0], param_names, owner))
    for name, fn in (("sin", sp.sin), ("cos", sp.cos), ("tan", sp.tan),
                     ("exp", sp.exp), ("log", sp.log), ("sqrt", sp.sqrt)):
        if expr.func is fn:
            return ir.FuncCall(name, (from_sympy(expr.args[0], param_names, owner),))
    if expr is sp.pi:
        return ir.Const(sp.pi)
    if expr is sp.E:
        return ir.Const(sp.E)
    raise TLSymbolicError(
        f"cannot represent symbolic result '{expr}' as a TensorLang operator; "
        f"only elementary functions of the operator parameters are supported"
    )


def differentiate(root: ir.Expr, owner: str, params: tuple[str, ...]) -> ir.Expr:
    if len(params) != 1:
        raise TLSymbolicError(
            f"diff requires a single-parameter operator, got params {list(params)}"
        )
    symbols: dict[tuple[str, str], sp.Symbol] = {}
    sym = to_sympy(root, symbols)
    var = symbols.get((owner, params[0]))
    if var is None:  # operator does not depend on its parameter: derivative is 0
        return ir.Const(sp.Integer(0))
    try:
        result = sp.diff(sym, var)
    except Exception as exc:
        raise TLSymbolicError(f"could not differentiate expression: {exc}") from exc
    return from_sympy(result, set(params), owner)


def simplify(root: ir.Expr, owner: str, params: tuple[str, ...]) -> ir.Expr:
    symbols: dict[tuple[str, str], sp.Symbol] = {}
    sym = to_sympy(root, symbols)
    try:
        result = sp.simplify(sym)
    except Exception as exc:
        raise TLSymbolicError(f"could not simplify expression: {exc}") from exc
    return from_sympy(result, set(params), owner)


def definite_integral(root: ir.Expr, owner: str, params: tuple[str, ...],
                      lo, hi) -> sp.Basic:
    """Exact definite integral of a single-parameter operator. No numerical
    fallback: if SymPy cannot produce an exact result, this raises."""
    if len(params) != 1:
        raise TLSymbolicError(
            f"integrate requires a single-parameter operator, got {list(params)}"
        )
    symbols: dict[tuple[str, str], sp.Symbol] = {}
    sym = to_sympy(root, symbols)
    var = symbols.get((owner, params[0]))
    if var is None:
        raise TLSymbolicError("cannot integrate an operator independent of its parameter")
    try:
        result = sp.integrate(sym, (var, sp.sympify(lo), sp.sympify(hi)))
    except Exception as exc:
        raise TLSymbolicError(f"could not integrate expression: {exc}") from exc
    if isinstance(result, sp.Integral):
        raise TLSymbolicError(
            f"no exact closed form for the integral of '{sym}'; "
            f"numerical quadrature is not applied implicitly"
        )
    return result


def indefinite_integral(root: ir.Expr, owner: str,
                        params: tuple[str, ...]) -> ir.Expr:
    if len(params) != 1:
        raise TLSymbolicError(
            f"integrate requires a single-parameter operator, got {list(params)}"
        )
    symbols: dict[tuple[str, str], sp.Symbol] = {}
    sym = to_sympy(root, symbols)
    var = symbols.get((owner, params[0]))
    if var is None:
        raise TLSymbolicError("cannot integrate an operator independent of its parameter")
    try:
        result = sp.integrate(sym, var)
    except Exception as exc:
        raise TLSymbolicError(f"could not integrate expression: {exc}") from exc
    if isinstance(result, sp.Integral):
        raise TLSymbolicError(
            f"no exact antiderivative for '{sym}'; "
            f"numerical quadrature is not applied implicitly"
        )
    return from_sympy(result, set(params), owner)


def symbols_for(owner: str, params: tuple[str, ...]) -> dict[tuple[str, str], sp.Symbol]:
    return {(owner, p): sp.Symbol(p) for p in params}


def to_numeric(value) -> int | float | bool:
    """Lower an exact SymPy number for vectorized numerical evaluation.

    Exactness is preserved for symbolic results (bare constant operators);
    this is only the numeric-lowering boundary, where float data already
    makes the result approximate.
    """
    import math

    if isinstance(value, bool):
        return value
    if value is sp.true:
        return True
    if value is sp.false:
        return False
    if isinstance(value, sp.Integer):
        return int(value)
    if isinstance(value, sp.Rational):
        return float(value)
    if isinstance(value, sp.Float):
        return float(value)
    if value is sp.pi:
        return math.pi
    if value is sp.E:
        return math.e
    raise TLSymbolicError(
        f"cannot lower symbolic value '{value}' to a number for "
        f"numerical evaluation"
    )
