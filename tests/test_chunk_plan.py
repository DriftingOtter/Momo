"""Milestone 2 tests: chunk-aware planning — alignment, rechunking,
contraction subset, rejections, and diagnostics."""
import numpy as np
import pytest
import zarr

from tensorlang import P, T, TLTypeError, TLUnsupportedError
from tensorlang.runtime import chunk_plan as _cp


def mstore(path, data, chunks):
    arr = zarr.open_array(path, mode="w", shape=data.shape,
                          chunks=chunks, dtype="float64")
    arr[:] = data
    return path


@pytest.fixture
def pair(tmp_path):
    rng = np.random.default_rng(21)
    a = rng.random((120, 4)) + 0.5
    b = rng.random((120, 4)) + 0.5
    pa = mstore(str(tmp_path / "a.zarr"), a, (30, 4))
    pb = mstore(str(tmp_path / "b.zarr"), b, (40, 4))
    return pa, pb, a, b


def test_aligned_chunks_no_movement(pair):
    pa, _, a, _ = pair
    X = P("X")
    out = ((X + 1.0) * 2)(T.scan_zarr(pa)).collect()
    assert np.allclose(out, (a + 1.0) * 2)
    from tensorlang import sin

    text = (sin(X) + X)(T.scan_zarr(pa)).explain()
    assert "dask-chunked" in text
    assert "no movement" in text


def test_misaligned_chunks_unify_correctly(pair):
    pa, pb, a, b = pair
    X, Y = P("X"), P("Y")
    got = (X + Y)(T.scan_zarr(pa), T.scan_zarr(pb)).collect()
    assert np.allclose(got, a + b)
    text = (X + Y)(T.scan_zarr(pa), T.scan_zarr(pb)).explain()
    assert "[rechunk]" in text  # explicit subdivision recorded


def test_broadcast_scalar_and_vector(pair):
    pa, _, a, _ = pair
    X = P("X")
    assert np.allclose((X * 2.0)(T.scan_zarr(pa)).collect(), a * 2.0)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose((X + T(v))(T.scan_zarr(pa)).collect(), a + v)


def test_concat_stack_mixed_layouts(tmp_path):
    rng = np.random.default_rng(5)
    a = rng.random((60, 4))
    b = rng.random((30, 4))
    pa = mstore(str(tmp_path / "a.zarr"), a, (30, 4))
    pb = mstore(str(tmp_path / "b.zarr"), b, (10, 4))
    from tensorlang import concat, stack

    assert np.allclose(concat([T.scan_zarr(pa), T.scan_zarr(pb)]).collect(),
                       np.concatenate([a, b]))
    assert np.allclose(stack([T.scan_zarr(pa), T.scan_zarr(pa)]).collect(),
                       np.stack([a, a]))


def test_irregular_chunks(pair):
    pa, pb, a, b = pair
    X, Y = P("X"), P("Y")
    # (30,30,30,30)x(4,) vs (40,40,40)x(4,): union has 7 row chunks.
    text = (X + Y)(T.scan_zarr(pa), T.scan_zarr(pb)).explain()
    assert "union" in text or "rechunk" in text
    assert np.allclose((X * Y)(T.scan_zarr(pa), T.scan_zarr(pb)).collect(),
                       a * b)


def test_reshape_supported_and_rejected(tmp_path):
    rng = np.random.default_rng(7)
    a = rng.random((100, 4))
    p = mstore(str(tmp_path / "a.zarr"), a, (25, 4))
    from tensorlang import reshape as _reshape
    X = P("X")
    assert _reshape(X, (400,))(T.scan_zarr(p)).collect().shape == (400,)
    assert np.allclose(_reshape(X, (400,))(T.scan_zarr(p)).collect(), a.reshape(400))
    with pytest.raises(TLUnsupportedError) as exc:
        _reshape(X, (50, 8))(T.scan_zarr(p)).collect()
    assert exc.value.code == "TL1010"
    with pytest.raises(Exception):
        _reshape(X, (7, 7))(T.scan_zarr(p)).collect()  # size mismatch: math error


def test_transpose_moveaxis(tmp_path):
    rng = np.random.default_rng(9)
    a = rng.random((30, 40))
    p = mstore(str(tmp_path / "a.zarr"), a, (10, 40))
    from tensorlang import moveaxis, transpose
    X = P("X")
    assert np.allclose(transpose(X)(T.scan_zarr(p)).collect(), a.T)
    assert np.allclose(moveaxis(X, 0, 1)(T.scan_zarr(p)).collect(),
                       np.moveaxis(a, 0, 1))


def test_tree_reduction_axes(tmp_path):
    a = np.arange(24.0).reshape(6, 4)
    p = mstore(str(tmp_path / "a.zarr"), a, (2, 4))
    assert T.scan_zarr(p).sum().collect() == pytest.approx(a.sum())
    assert T.scan_zarr(p).mean().collect() == pytest.approx(a.mean())


def test_matmul_bounded_subset(tmp_path):
    rng = np.random.default_rng(13)
    A = rng.random((12, 8)) + 0.5
    B = rng.random((8, 6)) + 0.5
    pa = mstore(str(tmp_path / "a.zarr"), A, (6, 8))
    pb = mstore(str(tmp_path / "b.zarr"), B, (8, 3))
    X, Y = P("X"), P("Y")
    got = (X @ Y)(T.scan_zarr(pa), T.scan_zarr(pb)).collect()
    assert np.allclose(got, A @ B)
    text = (X @ Y)(T.scan_zarr(pa), T.scan_zarr(pb)).explain()
    assert "matmul" in text and "unchunked" in text


def test_matmul_partitioned_contract_rejected(tmp_path):
    rng = np.random.default_rng(17)
    A = rng.random((8, 6)) + 0.5
    B = rng.random((6, 4)) + 0.5
    pa = mstore(str(tmp_path / "a.zarr"), A, (4, 3))
    pb = mstore(str(tmp_path / "b.zarr"), B, (3, 4))
    X, Y = P("X"), P("Y")
    with pytest.raises(TLUnsupportedError) as exc:
        (X @ Y)(T.scan_zarr(pa), T.scan_zarr(pb)).collect()
    assert exc.value.code == "TL1010"
    assert "contract" in str(exc.value)


def test_matmul_shape_error_is_math_error(tmp_path):
    A = np.ones((4, 3))
    B = np.ones((5, 2))
    pa = mstore(str(tmp_path / "a.zarr"), A, (2, 3))
    pb = mstore(str(tmp_path / "b.zarr"), B, (5, 2))
    X, Y = P("X"), P("Y")
    with pytest.raises(Exception) as exc:
        (X @ Y)(T.scan_zarr(pa), T.scan_zarr(pb)).collect()
    assert exc.value.code == "TL0502"


def test_contraction_rejected(tmp_path):
    a = np.ones((4, 4))
    p = mstore(str(tmp_path / "a.zarr"), a, (2, 4))
    from tensorlang import einsum

    with pytest.raises(TLUnsupportedError) as exc:
        einsum("ii->i", T.scan_zarr(p)).collect()
    assert exc.value.code == "TL1001"


def test_policy_gates(tmp_path):
    a = np.ones((100, 4))
    p = mstore(str(tmp_path / "a.zarr"), a, (50, 4))
    old = _cp.set_chunk_policy(max_block_bytes=1024)
    try:
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(p)).collect()
        assert exc.value.code == "TL1011"
    finally:
        _cp.set_chunk_policy(**old)
    old = _cp.set_chunk_policy(max_tasks=5)
    try:
        with pytest.raises(TLUnsupportedError) as exc:
            (P("X") + 1)(T.scan_zarr(p)).collect()
        assert exc.value.code == "TL1012"
    finally:
        _cp.set_chunk_policy(**old)


def test_global_ops_rejected_with_codes(tmp_path):
    a = np.ones((10, 2))
    p = mstore(str(tmp_path / "a.zarr"), a, (5, 2))
    X = P("X")
    from tensorlang import sort

    with pytest.raises(TLUnsupportedError) as exc:
        sort(X)(T.scan_zarr(p)).collect()
    assert exc.value.code == "TL1013"
    with pytest.raises(TLUnsupportedError) as exc:
        X.sum(by="g")(T.scan_zarr(p)).collect()
    assert exc.value.code == "TL1013"


def test_planner_bypass_impossible(tmp_path):
    from tensorlang.runtime.executor import evaluate

    a = np.ones((8, 2))
    p = mstore(str(tmp_path / "a.zarr"), a, (4, 2))
    from tensorlang.core import expression as ir

    with pytest.raises(TLUnsupportedError) as exc:
        evaluate(ir.Chunked(p))
    assert exc.value.code == "TL1001"


def test_zero_sized_axis(tmp_path):
    a = np.zeros((0, 4))
    p = mstore(str(tmp_path / "a.zarr"), a, (4, 4))
    X = P("X")
    assert (X + 1)(T.scan_zarr(p)).collect().shape == (0, 4)
    assert (X.sum())(T.scan_zarr(p)).collect() == 0.0


def test_cross_layout_numerical_correctness(tmp_path):
    rng = np.random.default_rng(23)
    a = rng.random((90, 5))
    b = rng.random((90, 5))
    pa = mstore(str(tmp_path / "a.zarr"), a, (30, 5))
    pb = mstore(str(tmp_path / "b.zarr"), b, (18, 5))
    X, Y = P("X"), P("Y")
    expr = ((X - Y) ** 2).mean()
    assert expr(T.scan_zarr(pa), T.scan_zarr(pb)).collect() == pytest.approx(
        float(((a - b) ** 2).mean()))


def test_type_mismatch_is_type_error(tmp_path):
    a = np.ones((4, 4))
    p = mstore(str(tmp_path / "a.zarr"), a, (2, 4))
    with pytest.raises(TLTypeError):
        (P("X") + "nope")(T.scan_zarr(p)).collect()
