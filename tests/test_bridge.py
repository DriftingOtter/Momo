"""v0.7 M1 tests: lazy structured-to-dense bridge."""
import numpy as np
import polars as pl
import pytest

from tensorlang import P, T, TLTypeError, TLUnsupportedError


@pytest.fixture
def frame_path(tmp_path):
    df = pl.DataFrame({
        "a": [1.0, 2.0, None, 4.0],
        "b": [10, 20, 30, 40],
        "s": ["x", "y", "z", "w"],
        "t": [1, None, 3, 4],
    }).with_columns(pl.col("t").cast(pl.Int64))
    path = str(tmp_path / "f.parquet")
    df.write_parquet(path)
    return path


def test_lazy_construction_reads_no_rows(frame_path):
    lazy = T.scan_parquet(frame_path).to_dense(columns=["a", "b"])
    # Delete the source: construction already finished without rows.
    import os
    os.remove(frame_path)
    with pytest.raises(Exception):
        lazy.collect()


def test_explain_abstract_without_execution(frame_path):
    lazy = T.scan_parquet(frame_path).to_dense(columns=["a", "b"])
    text = lazy.explain()
    assert "to_dense" in text or "conversion" in text
    assert "unknown" in text  # row count honestly unknown
    summary = lazy.abstract()
    assert summary["bridge"]["columns"] == ["a", "b"]
    assert summary["bridge"]["row_known"] is False
    assert summary["bridge"]["nulls"] == "nan"


def test_materialized_rows_known():
    df = T(pl.DataFrame({"a": [1.0, 2.0], "b": [3, 4]}))
    summary = df.to_dense(columns=["a", "b"]).abstract()
    assert summary["bridge"]["rows"] == 2
    assert summary["bridge"]["row_known"] is True
    assert summary["bridge"]["predicted_dtype"] == "float64"


def test_column_order_and_values(frame_path):
    got = T.scan_parquet(frame_path).to_dense(
        columns=["b", "a"]).collect()
    assert np.allclose(got, [[10.0, 1.0], [20.0, 2.0], [30.0, np.nan],
                             [40.0, 4.0]], equal_nan=True)
    assert got.dtype == np.float64


def test_unknown_duplicate_and_bad_policy(frame_path):
    scan = T.scan_parquet(frame_path)
    with pytest.raises(TLTypeError) as exc:
        scan.to_dense(columns=["nope"]).collect()
    assert exc.value.code == "TL0418"
    with pytest.raises(TLTypeError) as exc:
        scan.to_dense(columns=["a", "a"])
    assert exc.value.code == "TL0418"
    with pytest.raises(TLTypeError) as exc:
        scan.to_dense(columns=[])
    assert exc.value.code == "TL0418"
    with pytest.raises(TLTypeError) as exc:
        scan.to_dense(columns=["a"], nulls="drop")
    assert exc.value.code == "TL0418"


def test_unsupported_columns(frame_path):
    with pytest.raises(TLTypeError) as exc:
        T.scan_parquet(frame_path).to_dense(columns=["a", "s"]).collect()
    assert exc.value.code == "TL0418"
    assert "'s'" in str(exc.value)


def test_nullable_int_promotion(frame_path):
    got = T.scan_parquet(frame_path).to_dense(columns=["t"]).collect()
    assert got.dtype == np.float64
    assert np.isnan(got[1, 0])


def test_null_policies(frame_path):
    scan = T.scan_parquet(frame_path)
    assert np.isnan(scan.to_dense(columns=["a"]).collect()[2, 0])
    with pytest.raises(TLTypeError) as exc:
        scan.to_dense(columns=["a"], nulls="error").collect()
    assert exc.value.code == "TL0418"
    assert scan.to_dense(columns=["b"], nulls="error").collect().shape == (4, 1)


def test_filtered_join_query(frame_path, tmp_path):
    other = pl.DataFrame({"a": [1.0, 4.0], "k": ["u", "v"]})
    opath = str(tmp_path / "o.parquet")
    other.write_parquet(opath)
    base = T.scan_parquet(frame_path)
    other_scan = T.scan_parquet(opath)
    filt = base[base.a > 1.5]
    joined = filt.join(other_scan, on="a")
    got = joined.to_dense(columns=["b"]).collect()
    assert got.shape == (1, 1)


def test_memory_admission(frame_path):
    from tensorlang.runtime import chunk_plan as _cp

    scan = T.scan_parquet(frame_path)
    _cp.set_chunk_policy(memory_budget_bytes=1024 * 1024, max_workers=1)
    try:
        got = scan.to_dense(columns=["a", "b"]).collect()
        assert got.shape == (4, 2)
    finally:
        _cp.clear_chunk_policy()
    _cp.set_chunk_policy(memory_budget_bytes=10, max_workers=1)
    try:
        with pytest.raises(TLUnsupportedError) as exc:
            scan.to_dense(columns=["a", "b"]).collect()
        assert exc.value.code == "TL1020"
    finally:
        _cp.clear_chunk_policy()


def test_repeated_collect_caches(frame_path):
    lazy = T.scan_parquet(frame_path).to_dense(columns=["a"])
    first = lazy.collect()
    second = lazy.collect()
    assert first is second


def test_no_cross_lazy_caching(frame_path):
    import shutil

    lazy1 = T.scan_parquet(frame_path).to_dense(columns=["a"])
    assert lazy1.collect().shape == (4, 1)
    df = pl.DataFrame({"a": [9.0], "b": [1], "s": ["q"], "t": [2]})
    df.write_parquet(frame_path)
    lazy2 = T.scan_parquet(frame_path).to_dense(columns=["a"])
    assert lazy2.collect().tolist() == [[9.0]]


def test_fit_from_lazy_dense(frame_path):
    from tensorlang import impute, standardize

    scan = T.scan_parquet(frame_path)
    dense = scan.to_dense(columns=["a", "b"])
    fitted = (impute("mean") >> standardize()).fit(dense)
    out = np.asarray(fitted(dense).collect())
    assert out.shape == (4, 2)
    assert abs(out.mean()) < 1e-9
    # Holdout reuse: frozen parameters, identical results.
    again = np.asarray(fitted(dense).collect())
    assert np.allclose(out, again)


def test_fit_lazy_frame_suggests_bridge(frame_path):
    from tensorlang import standardize

    scan = T.scan_parquet(frame_path)
    lazy_frame = scan[scan.a > 0.0]
    with pytest.raises(TLTypeError) as exc:
        standardize().fit(lazy_frame)
    assert exc.value.code == "TL0418"
    assert "to_dense" in str(exc.value)


def test_grad_through_bridge_rejected(frame_path):
    import polars as pl

    from tensorlang import grad
    from tensorlang.core import expression as ir
    from tensorlang.core.operator import Operator

    inner = ir.ToDense(ir.Param("g", "X"), ("a",), "nan")
    op = Operator(("X",), ir.Reduce("sum", inner, None, None, False), "g")
    df = T(pl.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(Exception) as exc:
        grad(op)(df).collect()
    assert "conversion" in str(exc.value).lower()


def test_matches_explicit_reference(frame_path):
    scan = T.scan_parquet(frame_path)
    got = scan.to_dense(columns=["b", "a", "t"]).collect()
    frame = pl.read_parquet(frame_path)
    ref = np.column_stack([
        frame["b"].to_numpy().astype(np.float64),
        frame["a"].to_numpy().astype(np.float64),
        frame["t"].cast(pl.Float64).to_numpy()])
    assert got.dtype == np.float64
    assert np.allclose(got, ref, equal_nan=True)


def test_profile_reports_bridge(frame_path):
    prof = T.scan_parquet(frame_path).to_dense(columns=["a"]).profile()
    text = str(prof)
    assert "bridge_materialize" in text and "bridge_convert" in text
    assert "bridge_rows=4" in text
