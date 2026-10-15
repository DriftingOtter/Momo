"""M3 tests: structured vs dense NaN/null comparison semantics.

NaN and null are DISTINCT missing-value spellings with
representation-dependent comparison behavior (NumPy IEEE vs Polars
total-order equality).  These tests pin the documented contract in
``docs/COMPARISONS.md``: use ``is_null()``/``not_null()`` predicates,
never ``==`` self-filters.
"""
import numpy as np
import polars as pl
import pytest

from tensorlang import T, TLTypeError, fill_null, is_null, not_null

NAN = float("nan")


def test_dense_nan_inequality():
    d = T(np.array([1.0, NAN, 3.0]))
    assert (d == NAN).collect().tolist() == [False, False, False]
    assert (d == d).collect().tolist() == [True, False, True]


def test_structured_nan_equality_and_null_propagation():
    s = T(pl.Series("v", [1.0, NAN, None, 4.0]))
    assert (s == NAN).collect().to_list() == [False, True, None, False]


def test_eq_none_rejected_both():
    with pytest.raises(TLTypeError):
        (T(np.array([1.0])) == None).collect()  # noqa: E711
    with pytest.raises(TLTypeError):
        (T(pl.Series("v", [1.0])) == None).collect()  # noqa: E711


def test_is_null_divergence_documented():
    # Dense treats NaN as missing; structured only tracks true nulls.
    assert is_null(T(np.array([1.0, NAN]))).collect().tolist() == [False, True]
    assert is_null(T(pl.Series("v", [1.0, NAN, None]))).collect().to_list() == [
        False, False, True]


def test_not_null_filter_keeps_nan_drops_null():
    f = T(pl.DataFrame({"v": [1.0, NAN, None, 4.0]}))
    assert f[not_null(f.v)].collect().height == 3
    assert f[f.v == NAN].collect().height == 1


def test_self_filter_antipattern_fails():
    # `x == x` drops NaN in dense but KEEPS NaN in structured: never use
    # it as a missing-value filter.
    d = T(np.array([1.0, NAN]))
    assert d[d == d].collect().tolist() == [1.0]
    f = T(pl.DataFrame({"v": [1.0, NAN, None]}))
    assert f[f.v == f.v].collect().height == 2


def test_fill_null_then_filter():
    f = T(pl.DataFrame({"v": [1.0, None, 3.0]}))
    filled = fill_null(f.v, 0.0).collect().to_list()
    assert filled == [1.0, 0.0, 3.0]


def test_to_dense_null_policies():
    g = T(pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]}))
    arr = np.asarray(g.to_dense(columns=["a", "b"], nulls="nan").collect())
    assert arr[1, 0] != arr[1, 0]  # NaN promotion, no silent fill
    assert str(arr.dtype) == "float64"
    with pytest.raises(Exception) as exc_info:
        g.to_dense(columns=["a", "b"], nulls="error").collect()
    assert getattr(exc_info.value, "code", "") == "TL0418"
    clean = T(pl.DataFrame({"a": [1.0, 2.0]}))
    assert clean.to_dense(columns=["a"], nulls="error").collect().shape == (2, 1)


def test_nullable_int_promotion():
    i = T(pl.DataFrame({"n": [1, None, 3]}))
    arr = np.asarray(i.to_dense(columns=["n"], nulls="nan").collect())
    assert str(arr.dtype) == "float64"
    assert arr[1, 0] != arr[1, 0]


def test_grouped_agg_skips_nulls():
    f = T(pl.DataFrame({"k": [0, 0, 1], "v": [1.0, None, 3.0]}))
    assert f.v.mean(by=f.k).collect().sort("k")["v"].to_list() == [1.0, 3.0]


def test_nullable_boolean_conversion():
    b = T(pl.DataFrame({"f": [True, None, False]}))
    arr = np.asarray(b.to_dense(columns=["f"], nulls="nan").collect())
    assert str(arr.dtype) == "float64"
    assert arr[0, 0] == 1.0 and arr[2, 0] == 0.0
    assert arr[1, 0] != arr[1, 0]
    with pytest.raises(Exception) as exc_info:
        b.to_dense(columns=["f"], nulls="error").collect()
    assert getattr(exc_info.value, "code", "") == "TL0418"
