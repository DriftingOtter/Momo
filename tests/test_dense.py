"""Phase 2 tests: dense runtime (NumPy backend) vs NumPy itself."""
import numpy as np
import pytest

from tensorlang import T, TLTypeError, TLShapeError, TLUnsupportedError


def test_scalar_tensor():
    s = T(5)
    assert s.kind == "dense"
    assert s.shape == ()
    assert s.rank == 0
    assert s.collect() == 5


def test_vector_matrix_metadata():
    v = T([1, 2, 3])
    assert (v.shape, v.rank) == ((3,), 1)
    m = T([[1, 2], [3, 4]])
    assert (m.shape, m.rank) == ((2, 2), 2)
    assert m.dtype == np.dtype("int64")


def test_nd_tensor():
    a = np.arange(24).reshape(2, 3, 4)
    assert T(a).collect().tolist() == a.tolist()


def test_arithmetic_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.random((4, 3)), rng.random((4, 3))
    ta, tb = T(a), T(b)
    assert np.allclose((ta + tb).collect(), a + b)
    assert np.allclose((ta - tb).collect(), a - b)
    assert np.allclose((ta * tb).collect(), a * b)
    assert np.allclose((ta / tb).collect(), a / b)
    assert np.allclose((ta**2).collect(), a**2)
    assert np.allclose((-ta).collect(), -a)
    assert np.allclose(abs(ta - tb).collect(), np.abs(a - b))


def test_scalar_broadcast_matches_numpy():
    a = np.arange(6).reshape(2, 3)
    t = T(a)
    assert ((t * 2 + 1).collect() == a * 2 + 1).all()
    assert ((2 * t).collect() == 2 * a).all()
    assert ((10 - t).collect() == 10 - a).all()


def test_broadcast_vector_matrix_matches_numpy():
    a = np.ones((3, 4))
    v = np.arange(4)
    assert ((T(a) + T(v)).collect() == a + v).all()


def test_broadcast_mismatch_raises():
    with pytest.raises(TLShapeError):
        (T(np.ones((3,))) + T(np.ones((4,)))).collect()


def test_comparisons_and_logic_match_numpy():
    a = np.array([1, 5, 12, 20, 30])
    t = T(a)
    assert (((t > 10) & (t < 25)).collect() == ((a > 10) & (a < 25))).all()
    assert (((t == 5) | (t == 30)).collect() == ((a == 5) | (a == 30))).all()
    assert ((~(t > 10)).collect() == (~(a > 10))).all()
    assert ((t != 5).collect() == (a != 5)).all()
    assert ((t >= 20).collect() == (a >= 20)).all()
    assert ((t <= 1).collect() == (a <= 1)).all()


def test_boolean_masking_matches_numpy():
    a = np.array([1, 5, 12, 20, 30])
    t = T(a)
    mask = (t > 10) & (t < 25)
    assert t[mask].collect().tolist() == a[(a > 10) & (a < 25)].tolist()
    # mask length mismatch is an error, not silent truncation
    with pytest.raises(TLShapeError):
        t[T([True, False])].collect()


def test_indexing_slicing():
    a = np.arange(10)
    t = T(a)
    assert t[2].collect() == 2
    assert t[2:5].collect().tolist() == [2, 3, 4]
    m = T(np.arange(12).reshape(3, 4))
    assert m[1, 2].collect() == 6


def test_field_selection_on_dense_rejected():
    with pytest.raises(TLTypeError):
        T([1, 2, 3])["amount"].collect()
    with pytest.raises(TLTypeError):
        T([1, 2, 3])["amount"]


def test_matmul_matches_numpy():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert (T(a) @ T(a.T)).collect().tolist() == (a @ a.T).tolist()
    with pytest.raises(TLShapeError):
        (T(np.ones((2, 3))) @ T(np.ones((2, 3)))).collect()
    with pytest.raises(TLTypeError):
        (T({"x": [1]}) @ T({"x": [1]})).collect()


def test_transpose_reshape_flatten():
    a = np.arange(6).reshape(2, 3)
    t = T(a)
    assert t.T.collect().tolist() == a.T.tolist()
    assert t.reshape(3, 2).collect().tolist() == a.reshape(3, 2).tolist()
    assert t.flatten().collect().tolist() == a.flatten().tolist()
    with pytest.raises(TLShapeError):
        t.reshape(4, 4).collect()


def test_reductions_match_numpy():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = T(a)
    assert t.sum().collect() == pytest.approx(a.sum())
    assert t.mean().collect() == pytest.approx(a.mean())
    assert t.min().collect() == a.min()
    assert t.max().collect() == a.max()
    assert t.std().collect() == pytest.approx(a.std())
    assert t.var().collect() == pytest.approx(a.var())


def test_concat_stack_match_numpy():
    a, b = np.ones((2, 2)), np.zeros((2, 2))
    from tensorlang import concat, stack

    assert (concat([T(a), T(b)]).collect() == np.concatenate([a, b])).all()
    assert (stack([T(a), T(b)]).collect() == np.stack([a, b])).all()
    with pytest.raises(TLShapeError):
        concat([T(np.ones((2, 2))), T(np.ones((3, 3)))]).collect()


def test_modulo():
    assert (T([7, 8]) % 3).collect().tolist() == [1, 2]


def test_pow_matches_numpy_including_fast_paths():
    import warnings

    warnings.filterwarnings("ignore")
    for base in [np.array([2, 3, 4]), np.array([2.0, 3.0]), np.array([-2.0, 4.0])]:
        for exp in [0, 1, 2, 3, 2.5]:
            assert np.allclose(
                (T(base) ** exp).collect(), base**exp, equal_nan=True
            ), (base, exp)
    assert np.allclose(
        (T(np.array([2.0, 4.0])) ** -1).collect(), np.array([2.0, 4.0]) ** -1
    )
    # Native domain errors surface as descriptive type errors, not shape errors.
    with pytest.raises(TLTypeError):
        (T(np.array([2, 3])) ** -1).collect()


def test_linspace_arange():
    assert np.allclose(T.linspace(0, 1, 5).collect(), np.linspace(0, 1, 5))
    assert T.arange(4).collect().tolist() == [0, 1, 2, 3]


def test_transpose_structured_rejected():
    with pytest.raises(TLUnsupportedError):
        T({"x": [1, 2]}).T
    with pytest.raises(TLUnsupportedError):
        T({"x": [1, 2]}).reshape(2, 1)


def test_masked_assignment():
    t = T([1, 5, 12])
    t[t > 10] = 0
    assert t.collect().tolist() == [1, 5, 0]


def test_to_numpy_and_polars_guards():
    assert isinstance(T([1, 2]).to_numpy(), np.ndarray)
    with pytest.raises(TLTypeError):
        T({"a": [1], "b": ["x"]}).to_numpy()
    with pytest.raises(TLTypeError):
        T([1, 2]).to_polars()
