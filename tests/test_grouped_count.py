"""v0.7 M2 tests: grouped row/non-null counts and bridge memory gates."""
import numpy as np
import polars as pl
import pytest
import zarr

from tensorlang import P, T, TLDiffError, TLTypeError, TLUnsupportedError
from tensorlang import grad


def frame():
    return T(pl.DataFrame({
        "k": ["a", "b", "a", None, "b", "a"],
        "v": [1.0, None, 3.0, 4.0, 5.0, None],
        "w": [1, 2, 3, 4, 5, 6],
    }))


def sort_key(df):
    # Null keys sort first under polars ordering on both sides.
    return df.sort("k", nulls_last=False)


def native_row_counts():
    return (pl.DataFrame({
        "k": ["a", "b", "a", None, "b", "a"],
        "v": [1.0, None, 3.0, 4.0, 5.0, None]})
        .group_by("k").agg(pl.len().alias("count")).sort("k"))


def native_value_counts():
    return (pl.DataFrame({
        "k": ["a", "b", "a", None, "b", "a"],
        "v": [1.0, None, 3.0, 4.0, 5.0, None]})
        .group_by("k").agg(pl.col("v").count().alias("v")).sort("k"))


def test_group_row_counts():
    got = sort_key(frame().count(by="k").collect())
    ref = native_row_counts()
    assert got.columns == ["k", "count"]
    assert got["k"].to_list() == ref["k"].to_list()
    assert got["count"].to_list() == [1, 3, 2]  # null:1, a:3, b:2
    assert got["count"].dtype == pl.UInt32  # backend dtype preserved


def test_group_nonnnull_counts():
    got = sort_key(frame().v.count(by=frame().k).collect())
    ref = native_value_counts()
    assert got["v"].to_list() == ref["v"].to_list() == [1, 2, 1]
    assert got["v"].dtype == pl.UInt32


def test_row_vs_value_distinction():
    # Group b has 2 rows but only 1 non-null v: the two operations
    # must never be conflated.
    rc = frame().count(by="k").collect()
    vc = frame().v.count(by=frame().k).collect()
    rows = dict(zip(rc["k"].to_list(), rc["count"].to_list()))
    vals = dict(zip(vc["k"].to_list(), vc["v"].to_list()))
    assert rows == {"a": 3, "b": 2, None: 1}
    assert vals == {"a": 2, "b": 1, None: 1}


def test_null_keys_form_own_group():
    got = frame().count(by="k").collect()
    assert got.filter(pl.col("k").is_null())["count"].to_list() == [1]


def test_all_null_group():
    df = T(pl.DataFrame({"k": ["a", "a"], "v": [None, None]}).with_columns(
        pl.col("v").cast(pl.Float64)))
    assert df.v.count(by=df.k).collect()["v"].to_list() == [0]
    assert df.count(by="k").collect()["count"].to_list() == [2]


def test_nan_is_a_value_not_null():
    df = T(pl.DataFrame({"k": [1, 1, 2], "v": [float("nan"), 1.0, 2.0]}))
    assert df.v.count(by=df.k).collect().sort("k")["v"].to_list() == [2, 1]
    assert df.count(by="k").collect().sort("k")["count"].to_list() == [2, 1]


def test_empty_input():
    df = T(pl.DataFrame({"k": pl.Series([], dtype=pl.String),
                         "v": pl.Series([], dtype=pl.Float64)}))
    assert df.count(by="k").collect().height == 0
    assert df.v.count(by=df.k).collect().height == 0


def test_multiple_keys_rejected():
    df = frame()
    with pytest.raises(TLTypeError) as exc:
        df.v.sum(by=["k", "w"]).collect()
    assert exc.value.code == "TL0499"


def test_dtypes_preserved_and_mixed():
    df = T(pl.DataFrame({"k": [1, 1, 2],
                         "i": [1, None, 3],
                         "f": [1.5, 2.5, None],
                         "s": ["x", "y", "z"],
                         "b": [True, False, True]}))
    got = df.count(by="k").collect().sort("k")
    assert got["count"].to_list() == [2, 1]
    assert got["count"].dtype == pl.UInt32
    assert df.i.count(by=df.k).collect().sort("k")["i"].to_list() == [1, 1]


def test_count_key_collision():
    df = T(pl.DataFrame({"count": ["a", "a"], "v": [1.0, 2.0]}))
    with pytest.raises(TLTypeError) as exc:
        df.count(by="count").collect()
    assert "count" in str(exc.value)


def test_count_with_sum_mean_min_max(tmp_path):
    df = frame()
    total = df.v.sum(by=df.k)
    cnt = df.count(by="k")
    assert total.collect().height == 3
    assert cnt.collect().height == 3
    # Filtered grouping composes.
    big = df[df.v > 2.0]
    assert big.count(by="k").collect()["count"].sum() == 3


def test_grouped_then_joined():
    df = frame()
    counts = df.count(by="k")
    enriched = df.join(counts, on="k")
    # Null keys do not join (polars semantics, verified identical
    # natively): 5 of 6 rows match.
    assert enriched.collect().height == 5


def test_grouped_then_bridge():
    df = frame()
    counts = df.count(by="k")
    arr = counts.to_dense(columns=["count"]).collect()
    assert arr.dtype == np.uint32
    assert sorted(arr.ravel().tolist()) == [1, 2, 3]


def test_dense_counts():
    a = np.array([[1.0, np.nan], [np.inf, 2.0]])
    assert T(a).count().collect() == 3
    assert T(a).count(axis=0).collect().tolist() == [2, 1]
    assert T(np.ones((3, 4), dtype=int)).count().collect() == 12
    assert T(5.0).count().collect() == 1
    assert T(float("nan")).count().collect() == 0


def test_count_grad_rejected():
    X = P("X")
    with pytest.raises(TLDiffError) as exc:
        grad(X.count())(T([1.0, 2.0])).collect()
    assert "discrete" in str(exc.value)


def mstore(path, data, chunks):
    arr = zarr.open_array(path, mode="w", shape=data.shape,
                          chunks=chunks, dtype="float64")
    arr[:] = data
    return path


def test_chunked_counts(tmp_path):
    rng = np.random.default_rng(21)
    a = rng.random((12, 8))
    a[0, 0] = np.nan
    path = mstore(str(tmp_path / "a.zarr"), a, (5, 8))
    t = T.scan_zarr(path)
    assert t.count().collect() == 12 * 8 - 1
    assert np.allclose(t.count(axis=1).collect(),
                       np.count_nonzero(~np.isnan(a), axis=1))
    assert "census" in t.count(axis=1).explain()
    with pytest.raises(TLUnsupportedError) as exc:
        t.count(by="k").collect()
    assert exc.value.code == "TL1013"


def test_bridge_memory_gate_is_post_operand(tmp_path):
    # §6: the gate runs after the operand materializes (rows known), so
    # it bounds conversion — not the upstream query. Introspection must
    # say exactly that.
    from tensorlang.runtime import chunk_plan as _cp

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    path = str(tmp_path / "s.parquet")
    df.write_parquet(path)
    lazy = T.scan_parquet(path).to_dense(columns=["a", "b"])
    text = lazy.explain()
    assert "unknown" in text  # cardinality unknown pre-execution
    assert "guarantee scope" in text  # conversion gate, not upstream bound
    _cp.set_chunk_policy(memory_budget_bytes=10, max_workers=1)
    try:
        with pytest.raises(TLUnsupportedError) as exc:
            lazy.collect()
        assert exc.value.code == "TL1020"
    finally:
        _cp.clear_chunk_policy()
    # Unknown cardinality is never zero: the estimate text names it.
    assert "unknown" in lazy.explain()


def test_lazy_execution_and_caching(tmp_path):
    df = pl.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]})
    path = str(tmp_path / "s.parquet")
    df.write_parquet(path)
    lazy = T.scan_parquet(path).count(by="k")
    first, second = lazy.collect(), lazy.collect()
    assert first is second
    assert first.sort("k")["count"].to_list() == [1, 1]
