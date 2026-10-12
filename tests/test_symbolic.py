"""Phase 6 tests: symbolic mathematics (SymPy backend)."""
import numpy as np
import pytest
import sympy as sp

from tensorlang import (
    P,
    T,
    TLSymbolicError,
    TLTypeError,
    cos,
    diff,
    exp,
    integrate,
    log,
    sin,
    simplify,
    sqrt,
)


def test_diff_polynomial():
    df = diff(P("X") ** 3)
    assert df(T(2)).collect() == 12
    assert df(T(0)).collect() == 0


def test_diff_matches_sympy():
    x = sp.Symbol("X")
    f = sin(P("X")) * exp(-P("X") ** 2)
    df = diff(f)
    pts = np.linspace(0, 5, 50)
    expected = sp.lambdify(x, sp.diff(sp.sin(x) * sp.exp(-x**2), x), "numpy")(pts)
    assert np.allclose(df(T(pts)).collect(), expected)


def test_diff_trig_and_log():
    assert diff(sin(P("X")))(T(0.0)).collect() == pytest.approx(1.0)
    assert diff(cos(P("X")))(T(0.0)).collect() == pytest.approx(0.0)
    assert diff(log(P("X")))(T(float(np.e))).collect() == pytest.approx(1 / np.e)
    assert diff(sqrt(P("X")))(T(4.0)).collect() == pytest.approx(0.25)


def test_diff_constant_is_zero():
    assert diff(P("X") * 0 + 5)(T(99.0)).collect() == 0


def test_diff_requires_unary_operator():
    with pytest.raises(TLSymbolicError):
        diff(P("X") + P("Y"))


def test_diff_rejects_non_symbolic():
    D = P("D")
    with pytest.raises(TLSymbolicError):
        diff(D.amount + 1)


def test_definite_integral_is_exact():
    area = integrate(P("X") ** 2, 0, 5)
    result = area.collect()
    assert result == sp.Rational(125, 3)
    assert isinstance(result, sp.Rational)  # no silent float fallback
    assert float(result) == pytest.approx(125 / 3)


def test_definite_integral_trig():
    area = integrate(sin(P("X")), 0, sp.pi)
    assert area.collect() == 2


def test_indefinite_integral():
    F = integrate(P("X") ** 2)
    assert F.params == ("X",)
    # d/dx (x^3/3) = x^2
    assert diff(F)(T(4)).collect() == pytest.approx(16.0)


def test_integral_needs_both_limits():
    with pytest.raises(TLSymbolicError):
        integrate(P("X"), 0, None)


def test_no_exact_form_raises_instead_of_approximating():
    X = P("X")
    hopeless = exp(-X**2) * sin(X**3)
    with pytest.raises(TLSymbolicError):
        integrate(hopeless, 0, 1)


def test_simplify():
    f = simplify((P("X") + 1) ** 2 - P("X") ** 2 - 2 * P("X"))
    assert f(T(7)).collect() == pytest.approx(1.0)


def test_symbolic_operator_evaluates_numerically():
    f = sin(P("X")) * exp(-P("X") ** 2)
    pts = np.array([0.0, 0.5, 1.0])
    x = sp.Symbol("X")
    expected = sp.lambdify(x, sp.sin(x) * sp.exp(-x**2), "numpy")(pts)
    assert np.allclose(f(T(pts)).collect(), expected)


def test_diff_of_non_operator_rejected():
    with pytest.raises(TLTypeError):
        diff(T([1, 2]))


def test_exact_rational_arithmetic_preserved():
    op = P("X") ** 2
    area = integrate(op, 0, 5)
    assert area.collect() * 3 == 125
