"""Phase 3 tests: first-class operators, application, composition, laziness."""
import numpy as np
import pytest

from tensorlang import (
    P,
    T,
    TLShapeError,
    TLTypeError,
    cos,
    exp,
    log,
    sin,
    sqrt,
)
from tensorlang.runtime.planner import plan


def test_operator_application_is_non_mutating():
    v = T([1.0, 2.0, 3.0])
    before = v.collect().copy()
    f = P("X") ** 2
    out = f(v)
    assert out.collect().tolist() == [1.0, 4.0, 9.0]
    assert (v.collect() == before).all()


def test_application_returns_pending_computation():
    from tensorlang import Lazy

    out = (P("X") + 1)(T([1]))
    assert isinstance(out, Lazy)
    assert out.collect().tolist() == [2]


def test_composition_order():
    f = P("X") ** 2
    g = P("X") + 5
    h = f >> g
    assert h(T(3)).collect() == 14  # (3**2) + 5, not (3+5)**2
    assert (g >> f)(T(3)).collect() == 64


def test_composition_keeps_scopes_independent():
    X = P("X")
    f = X**2
    g = X + 1
    h = f >> g
    assert f.uid != g.uid
    assert h.params == ("X",)
    assert h(T(2)).collect() == 5


def test_chained_composition():
    h = (P("A") + 1) >> (P("B") * 2) >> (P("C") ** 2)
    assert h(T(1)).collect() == 16  # ((1+1)*2)**2


def test_composition_rejects_non_unary_rhs():
    with pytest.raises(TLTypeError):
        P("X") >> (P("Y") + P("Z"))


def test_arity_checked_at_application():
    h = P("X") + P("Y")
    with pytest.raises(TLShapeError):
        h(T(1))
    assert h(T(1), T(2)).collect() == 3


def test_multi_param_binding_is_positional():
    h = P("X") - P("Y")
    assert h(T(10), T(4)).collect() == 6


def test_standalone_application_does_not_mutate_input():
    v = T([1, 2, 3])
    (P("X") * 100)(v)
    assert v.collect().tolist() == [1, 2, 3]


def test_math_functions_match_numpy():
    a = np.linspace(0.1, 2.0, 9)
    t = T(a)
    assert np.allclose(sin(t).collect(), np.sin(a))
    assert np.allclose(cos(t).collect(), np.cos(a))
    assert np.allclose(exp(t).collect(), np.exp(a))
    assert np.allclose(log(t).collect(), np.log(a))
    assert np.allclose(sqrt(t).collect(), np.sqrt(a))


def test_math_functions_build_operators():
    f = sin(P("X")) * exp(-P("X") ** 2)
    assert f.params == ("X",)
    assert float(f(T(0.0)).collect()) == pytest.approx(0.0)


def test_collect_caches_and_reuses():
    v = T([1.0, 2.0])
    out = (P("X") + 1)(v)
    first = out.collect()
    (P("X") * 0)(v).inplace()
    # The earlier lazy graph holds a snapshot: unaffected by the mutation.
    assert first.tolist() == [2.0, 3.0]
    assert out.collect().tolist() == [2.0, 3.0]


def test_planner_assigns_backends():
    p = plan(((P("X") ** 2) + 1).root)
    assert p.node_count > 0
    assert set(p.summary()) <= {"numpy", "polars", "sympy", "scalar", "mixed", "symbolic"}


def test_constant_folding():
    from tensorlang.core import expression as ir
    from tensorlang.runtime.planner import fold_constants

    folded = fold_constants(ir.BinOp("add", ir.Const(2), ir.Const(3)))
    assert isinstance(folded, ir.Const) and folded.value == 5


def test_operator_in_operator_expression():
    base = P("X") ** 2
    h = base + 1
    assert h(T(3)).collect() == 10


def test_applying_operator_to_operator_rejected():
    with pytest.raises(TLTypeError):
        P("X")(P("Y"))


def test_wrong_composition_operand_rejected():
    with pytest.raises(TLTypeError):
        P("X") >> 5
