"""v0.6 M2 tests: axis-aware reductions and operator transpose."""
import numpy as np
import pytest
import zarr

from tensorlang import P, T, TLTypeError, TLUnsupportedError
from tensorlang import grad, vjp


@pytest.fixture
def a2():
    rng = np.random.default_rng(11)
    return (rng.random((6, 8)) + 0.5) * 2.0


@pytest.fixture
def a3():
    rng = np.random.default_rng(12)
    return (rng.random((4, 5, 6)) + 0.5) * 2.0


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "var", "std"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, -2, (0, 1), ()])
def test_dense_axis_matches_numpy(a2, op, axis):
    t = T(a2)
    got = getattr(t, op)(axis=axis).collect()
    ref = getattr(np, op)(a2, axis=axis)
    assert np.allclose(np.asarray(got), ref, equal_nan=True)


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "var", "std"])
def test_keepdims(a2, op):
    for axis in (None, 0, 1, (0, 1)):
        got = np.asarray(getattr(T(a2), op)(axis=axis,
                                            keepdims=True).collect())
        assert got.shape == np.asarray(getattr(np, op)(
            a2, axis=axis, keepdims=True)).shape
        assert np.allclose(got, getattr(np, op)(a2, axis=axis,
                                                keepdims=True),
                           equal_nan=True)


def test_rank3_axes(a3):
    assert T(a3).sum(axis=1).collect().shape == (4, 6)
    assert np.allclose(T(a3).mean(axis=(0, 2)).collect(),
                       a3.mean(axis=(0, 2)))
    assert T(a3).sum(axis=(2, 0, 2)).collect().shape == (5,)


def test_operator_form(a2):
    X = P("X")
    assert np.allclose(X.sum(axis=1)(T(a2)).collect(), a2.sum(axis=1))
    assert X.mean(axis=0).params == ("X",)


def test_invalid_axes(a2):
    t = T(a2)
    with pytest.raises(Exception) as exc:
        t.sum(axis=2).collect()
    assert exc.value.code == "TL0511"
    with pytest.raises(Exception) as exc:
        t.sum(axis=-3).collect()
    assert exc.value.code == "TL0511"
    with pytest.raises(Exception) as exc:
        t.sum(axis=1.5).collect()
    assert exc.value.code == "TL0511"
    with pytest.raises(Exception) as exc:
        t.sum(axis=True).collect()
    assert exc.value.code == "TL0511"
    with pytest.raises(Exception) as exc:
        t.sum(axis="x").collect()
    assert exc.value.code == "TL0511"
    with pytest.raises(Exception) as exc:
        T(np.ones(())).sum(axis=0).collect()
    assert exc.value.code == "TL0511"


def test_empty_axes_numpy_compatible():
    z = np.zeros((0, 4))
    assert np.allclose(T(z).sum(axis=0).collect(), np.zeros(4))
    assert T(z).sum(axis=1).collect().shape == (0,)
    with pytest.warns(RuntimeWarning):
        ref = np.mean(z, axis=0)
    assert np.allclose(T(z).mean(axis=0).collect(), ref, equal_nan=True)


def test_by_axis_conflict(a2):
    with pytest.raises(TLTypeError) as exc:
        T(a2).sum(by="g", axis=0).collect()
    assert exc.value.code == "TL0511"


def test_frame_axis_rejected():
    import polars as pl

    df = T(pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    with pytest.raises(TLTypeError) as exc:
        df.sum(axis=0).collect()
    assert exc.value.code == "TL0511"
    assert "ambiguous" in str(exc.value)


def test_series_axis():
    import polars as pl

    s = T(pl.Series("v", [1.0, 2.0, 3.0]))
    assert s.sum(axis=0).collect() == 6.0
    assert s.mean(axis=-1).collect() == 2.0
    with pytest.raises(TLTypeError) as exc:
        s.sum(axis=0, keepdims=True).collect()
    assert exc.value.code == "TL0511"


def test_operator_transpose(a2):
    X = P("X")
    f = X.T + 1
    assert f.params == ("X",)
    assert np.allclose(f(T(a2)).collect(), a2.T + 1)
    g = X.T @ X
    assert np.allclose(g(T(a2)).collect(), a2.T @ a2)


def test_tensor_lazy_transpose(a2):
    assert np.allclose(T(a2).T.collect(), a2.T)
    assert np.allclose(T(a2).transpose().collect(), a2.T)


def test_field_T_collision():
    import polars as pl

    X = P("X")
    # Structured field literally named "T" stays reachable explicitly.
    df = T(pl.DataFrame({"T": [1.0, 2.0], "x": [3.0, 4.0]}))
    assert df["T"].collect().to_list() == [1.0, 2.0]
    # ...while .T on operators means transpose (tested above).
    assert (X.T)(T(np.ones((2, 3)))).collect().shape == (3, 2)


def test_structured_transpose_errors():
    import polars as pl

    df = T(pl.DataFrame({"a": [1.0]}))
    with pytest.raises(TLUnsupportedError) as exc:
        df.T
    assert exc.value.code == "TL0901"


def test_field_hint_names_operator():
    X = P("X")
    with pytest.raises(TLTypeError) as exc:
        (X.reshape + 1)(T(np.ones(3))).collect()
    assert "reshape(X)" in str(exc.value)


def test_composed_axis_reductions(a2):
    X = P("X")
    f = X.sum(axis=1).mean()
    assert f(T(a2)).collect() == pytest.approx(a2.sum(axis=1).mean())
    g = (X - X.mean(axis=0)).sum(axis=1)
    assert np.allclose(g(T(a2)).collect(),
                       (a2 - a2.mean(axis=0)).sum(axis=1))


def test_axis_gradients_finite_difference():
    rng = np.random.default_rng(13)
    a = (rng.random((4, 5)) + 0.5) * 2.0
    X = P("X")

    def fd(op, ax):
        f = (lambda p: getattr(np, op)(p, axis=ax)
             if ax is not None else getattr(np, op)(p))
        out = np.zeros_like(a)
        eps = 1e-6
        for idx in np.ndindex(a.shape):
            dp = np.zeros_like(a)
            dp[idx] = eps
            out[idx] = ((f(a + dp) - f(a - dp)) / (2 * eps)).sum()
        return out

    for op in ("sum", "mean", "min", "max", "var", "std"):
        for ax in (None, 0, 1, -1, (0, 1)):
            f = getattr(X, op)(axis=ax) if ax is not None else getattr(X, op)()
            probe = (getattr(np, op)(a, axis=ax)
                     if ax is not None else getattr(np, op)(a))
            if np.ndim(probe) == 0:
                g = np.asarray(grad(f)(T(a)).collect())
            else:
                g = np.asarray(
                    vjp(f)(T(a), T(np.ones_like(probe))).collect())
            assert np.allclose(g, fd(op, ax), rtol=1e-3, atol=1e-6), (op, ax)


def test_min_ties_split_per_slice():
    X = P("X")
    a = np.array([[1.0, 5.0], [1.0, 2.0], [3.0, 2.0]])
    g = np.asarray(vjp(X.min(axis=0))(T(a), T([1.0, 1.0])).collect())
    assert np.allclose(g, [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    c = np.full((3, 2), 4.0)
    g2 = np.asarray(vjp(X.std(axis=0))(T(c), T([1.0, 1.0])).collect())
    assert np.allclose(g2, np.zeros((3, 2)))


def test_shared_axis_reduction_evaluates_once():
    from tensorlang.runtime import executor as _ex

    X = P("X")
    r = X.sum(axis=1)
    loss = (r + r).sum()
    counts = {}

    def hook(node):
        counts[type(node).__name__] = counts.get(type(node).__name__, 0) + 1
        return hook.orig(node)

    hook.orig = _ex._evaluate
    _ex._evaluate = hook
    try:
        assert loss(T(np.ones((3, 4)))).collect() == 24.0
    finally:
        _ex._evaluate = hook.orig
    assert counts.get("Reduce", 0) == 2  # axis-sum once, total once


def mstore(path, data, chunks):
    arr = zarr.open_array(path, mode="w", shape=data.shape,
                          chunks=chunks, dtype="float64")
    arr[:] = data
    return path


def test_chunked_axis_reductions(tmp_path):
    rng = np.random.default_rng(14)
    a = rng.random((12, 8)) + 0.5
    path = mstore(str(tmp_path / "a.zarr"), a, (5, 8))
    t = T.scan_zarr(path)
    assert np.allclose(t.sum(axis=1).collect(), a.sum(axis=1))
    assert np.allclose(t.mean(axis=0).collect(), a.mean(axis=0))
    assert t.sum(axis=1, keepdims=True).collect().shape == (12, 1)
    assert np.allclose(t.sum(axis=(0, 1)).collect(), a.sum())
    text = t.sum(axis=1).explain()
    assert "axes [1]" in text and "tree-reducible" in text


def test_chunked_axis_invalid_is_shape_error(tmp_path):
    a = np.ones((8, 4))
    path = mstore(str(tmp_path / "a.zarr"), a, (4, 4))
    with pytest.raises(Exception) as exc:
        T.scan_zarr(path).sum(axis=3).collect()
    assert exc.value.code == "TL0511"


def test_chunked_axis_under_budget(tmp_path):
    from tensorlang.runtime import chunk_plan as _cp

    rng = np.random.default_rng(15)
    a = rng.random((200, 8)) + 0.5
    path = mstore(str(tmp_path / "a.zarr"), a, (50, 8))
    _cp.set_chunk_policy(memory_budget_bytes=20 * 1024, max_workers=1)
    try:
        # Full output (12.8KB) fits, but the axis path is what matters:
        # small working set admitted with recorded estimates.
        got = T.scan_zarr(path).sum(axis=1).collect()
        assert np.allclose(got, a.sum(axis=1))
        assert _cp.LAST_DECISION["outcome"] == "admitted"
    finally:
        _cp.clear_chunk_policy()
