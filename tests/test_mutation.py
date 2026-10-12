"""Phase 4 tests: explicit in-place execution semantics."""
import numpy as np
import pytest

from tensorlang import P, T, TLMutationError


def test_inplace_replaces_contents():
    v = T([1, 5, 12, 20, 30])
    (P("X") ** 2 >> (P("Y") + 5))(v).inplace()
    assert v.collect().tolist() == [6, 30, 149, 405, 905]


def test_inplace_may_change_shape():
    v = T([1, 5, 12, 20, 30])
    (P("X")[P("X") > 100])(v).inplace()
    assert v.shape == (0,)
    w = T([50, 150, 250])
    (P("X")[P("X") > 100])(w).inplace()
    assert w.collect().tolist() == [150, 250]


def test_inplace_may_change_dtype_and_kind():
    v = T([1, 2, 3])
    (P("X") / 2)(v).inplace()
    assert v.collect().tolist() == [0.5, 1.0, 1.5]


def test_aliasing_same_object_sees_update():
    v = T([1, 2, 3])
    alias = v
    (P("X") + 10)(v).inplace()
    assert alias.collect().tolist() == [11, 12, 13]
    # Rebinding a name never mutates: `w = ...` only changes the binding.
    w = v
    w = T([0])
    assert v.collect().tolist() == [11, 12, 13]


def test_lazy_snapshot_survives_later_mutation():
    v = T([1.0, 2.0, 3.0])
    pending = (P("X") * 2)(v)
    (P("X") + 100)(v).inplace()
    assert v.collect().tolist() == [101.0, 102.0, 103.0]
    assert pending.collect().tolist() == [2.0, 4.0, 6.0]


def test_mult_input_application_rejects_inplace():
    h = P("X") + P("Y")
    pending = h(T(1), T(2))
    with pytest.raises(TLMutationError):
        pending.inplace()


def test_plain_tensor_expression_rejects_inplace():
    v = T([1, 2, 3])
    with pytest.raises(TLMutationError):
        (v * 2).inplace()


def test_inplace_on_lazy_input_rejected():
    v = T([1, 2, 3])
    selected = v[v > 1]
    with pytest.raises(TLMutationError):
        (P("X") + 1)(selected).inplace()


def test_failed_inplace_leaves_input_unchanged():
    v = T(np.ones((2, 3)))
    before = v.collect().copy()
    with pytest.raises(Exception):
        (P("X") @ P("X"))(v).inplace()  # (2,3) @ (2,3) is invalid
    assert (v.collect() == before).all()
    assert v.shape == (2, 3)


def test_inplace_returns_target_tensor():
    v = T([1])
    out = (P("X") + 1)(v).inplace()
    assert out is v
    assert v.collect().tolist() == [2]


def test_inplace_structured_row_filter():
    sales = T({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    (P("D")[P("D").a > 1])(sales).inplace()
    assert sales.collect()["a"].to_list() == [2, 3]
