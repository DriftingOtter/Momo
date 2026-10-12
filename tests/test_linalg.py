"""v0.3 tests: numerical linear algebra, statistics, tensor algebra."""
import numpy as np
import polars as pl
import pytest

from tensorlang import (
    T,
    TLShapeError,
    TLTypeError,
    corr,
    cov,
    eigh,
    einsum,
    moveaxis,
    norm,
    onehot,
    polynomial,
    quantile,
    solve,
    svd,
    tensordot,
)

sklearn = pytest.importorskip("sklearn")


@pytest.fixture
def mat():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_svd_matches_scipy(mat):
    import scipy.linalg as la

    U, S, Vh = svd(T(mat))
    assert isinstance((U, S, Vh), tuple) and len((U, S, Vh)) == 3
    u, s, vh = U.collect(), S.collect(), Vh.collect()
    ru, rs, rvh = la.svd(mat)
    assert np.allclose(s, rs)
    Ue, Se, Vhe = svd(T(mat), full_matrices=False)
    ue, se, vhe = Ue.collect(), Se.collect(), Vhe.collect()
    assert ue.shape == (3, 2) and vhe.shape == (2, 2)
    assert np.allclose(ue @ np.diag(se) @ vhe, mat, atol=1e-8)
    # Shared computation: each component collects independently and agrees.
    assert np.allclose(np.abs(u[:, :2]), np.abs(ru[:, :2]))
    with pytest.raises(TLShapeError):
        svd(T(np.ones(3))).S.collect()
    with pytest.raises(TLTypeError):
        svd(T({"a": [1.0]})).S.collect()


def test_svd_named_access(mat):
    result = svd(T(mat))
    assert np.allclose(result.S.collect(), result[1].collect())


def test_eigh_matches_numpy():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    values, vectors = eigh(T(A))
    w, v = np.linalg.eigh(A)
    assert np.allclose(values.collect(), w)
    assert np.allclose(np.abs(vectors.collect()), np.abs(v))
    assert np.allclose(vectors.collect() @ np.diag(w) @ vectors.collect().T, A,
                       atol=1e-8)
    assert list(values.collect()) == sorted(values.collect())  # ascending
    with pytest.raises(TLShapeError):
        eigh(T(np.ones((2, 3)))).values.collect()
    with pytest.raises(TLTypeError):
        eigh(T(np.array([[1.0, 2.0], [3.0, 4.0]]))).values.collect()


def test_solve_matches_numpy():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    assert np.allclose(solve(T(A), T(b)).collect(), np.linalg.solve(A, b))
    B = np.array([[9.0, 1.0], [8.0, 2.0]])
    assert np.allclose(solve(T(A), T(B)).collect(), np.linalg.solve(A, B))
    with pytest.raises(TLShapeError):
        solve(T(np.eye(2) * 0), T([1.0, 1.0])).collect()
    with pytest.raises(TLShapeError):
        solve(T(np.ones((2, 3))), T([1.0, 1.0])).collect()
    with pytest.raises(TLShapeError):
        solve(T(np.eye(2)), T([1.0])).collect()


def test_norm_matches_numpy(mat):
    assert norm(T(mat)).collect() == pytest.approx(np.linalg.norm(mat))
    assert norm(T(mat), ord="fro").collect() == pytest.approx(
        np.linalg.norm(mat, ord="fro"))
    v = np.array([3.0, 4.0])
    assert norm(T(v), ord=1).collect() == pytest.approx(7.0)
    assert norm(T(v), ord=np.inf).collect() == pytest.approx(4.0)
    assert norm(T({"x": [3.0, 4.0]}).x).collect() == pytest.approx(5.0)
    with pytest.raises(TLTypeError):
        norm(T({"x": [1.0]})).collect()
    with pytest.raises(TLTypeError):
        norm(T(mat), ord="bogus").collect()


def test_cov_corr_match_numpy():
    rng = np.random.default_rng(11)
    X = rng.random((40, 3)) + 0.5
    assert np.allclose(cov(T(X)).collect(), np.cov(X))
    assert np.allclose(corr(T(X)).collect(), np.corrcoef(X))
    frame = T({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
    got = cov(frame).collect()
    assert got.columns == ["variable", "a", "b", "c"]
    assert np.allclose(got.select(["a", "b", "c"]).to_numpy(), np.cov(X.T))
    assert np.allclose(corr(frame).collect().select(["a", "b", "c"]).to_numpy(),
                       np.corrcoef(X.T))
    with pytest.raises(TLTypeError):
        cov(T({"a": [1.0], "s": ["x"]})).collect()
    with pytest.raises(TLTypeError):
        corr(T({"a": [1.0], "s": ["x"]})).collect()
    with pytest.raises(TLShapeError):
        corr(T([1.0, 2.0])).collect()


def test_quantile_matches_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert quantile(T(a), 0.5).collect() == pytest.approx(2.5)
    assert np.allclose(quantile(T(a), [0.0, 0.5, 1.0]).collect(),
                       np.quantile(a, [0.0, 0.5, 1.0]))
    s = T({"x": [1.0, 2.0, 3.0, 4.0]})
    assert quantile(s.x, 0.25).collect() == pytest.approx(1.75)
    frame = quantile(s, 0.5).collect()
    assert frame.columns == ["quantile", "x"]
    with pytest.raises(TLTypeError):
        quantile(T(a), 1.5)
    with pytest.raises(TLTypeError):
        quantile(T(a), "mid")


def test_einsum_matches_numpy():
    rng = np.random.default_rng(5)
    A, B = rng.random((3, 4)), rng.random((4, 5))
    assert np.allclose(einsum("ij,jk->ik", T(A), T(B)).collect(), A @ B)
    assert np.allclose(einsum("ii->i", T(A @ A.T)).collect(),
                       np.einsum("ii->i", A @ A.T))
    v = rng.random(6)
    assert np.allclose(einsum("i,i->", T(v), T(v)).collect(), v @ v)
    with pytest.raises(TLTypeError):
        einsum("no-arrow", T(A))
    with pytest.raises(TLShapeError):
        einsum("ij,jk->ik", T(A), T(B.T)).collect()


def test_tensordot_matches_numpy():
    a = np.arange(24.0).reshape(2, 3, 4)
    b = np.arange(24.0).reshape(4, 3, 2)
    assert np.allclose(tensordot(T(a), T(b), axes=([2], [0])).collect(),
                       np.tensordot(a, b, axes=([2], [0])))
    assert np.allclose(tensordot(T(a), T(b), axes=1).collect(),
                       np.tensordot(a, b, axes=1))
    with pytest.raises(TLShapeError):
        tensordot(T(a), T(b)).collect()


def test_moveaxis_matches_numpy():
    a = np.zeros((2, 3, 4))
    assert moveaxis(T(a), 0, -1).collect().shape == (3, 4, 2)
    assert np.array_equal(moveaxis(T(a), 0, -1).collect(), np.moveaxis(a, 0, -1))
    with pytest.raises(TLShapeError):
        moveaxis(T(a), 5, 0).collect()


def test_onehot_dense_and_series():
    assert onehot(T([0, 2, 1])).collect().tolist() == [
        [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    assert onehot(T([1, 0]), categories=[0, 1, 2]).collect().shape == (2, 3)
    s = T({"c": ["b", "a", "b"]})
    got = onehot(s.c).collect()
    assert got.columns == ["a", "b"]
    assert got["a"].to_list() == [False, True, False]
    with pytest.raises(TLTypeError):
        onehot(T([0, 5]), categories=[0, 1]).collect()
    with pytest.raises(TLTypeError):
        onehot(T({"c": ["a", None]}).c).collect()
    with pytest.raises(TLTypeError):
        onehot(T(1.5)).collect()


def test_polynomial_matches_sklearn():
    from sklearn.preprocessing import PolynomialFeatures

    X = np.array([[2.0, 3.0], [1.0, 0.5], [0.0, 4.0]])
    for degree in (1, 2, 3):
        for bias in (True, False):
            ref = PolynomialFeatures(degree, include_bias=bias).fit_transform(X)
            got = polynomial(T(X), degree, include_bias=bias).collect()
            assert np.allclose(got, ref), (degree, bias)
    frame = T({"x0": [2.0, 1.0], "x1": [3.0, 0.5]})
    out = polynomial(frame, 2).collect()
    assert out.columns == ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]
    with pytest.raises(TLTypeError):
        polynomial(T(X), 0)


def test_structured_rejections_for_dense_only_ops():
    frame = T({"a": [1.0, 2.0]})
    with pytest.raises(TLTypeError):
        einsum("i,i->", frame, frame).collect()
    with pytest.raises(TLTypeError):
        tensordot(frame, frame).collect()
    with pytest.raises(TLTypeError):
        moveaxis(frame, 0, 1).collect()
