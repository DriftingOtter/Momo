"""Priority 2 tests: joins, sorting, unique, conditional replacement,
grouped transforms, windows, scans and writes — differential vs NumPy/Polars.
"""
import numpy as np
import polars as pl
import pytest

from tensorlang import (
    P,
    T,
    TLShapeError,
    TLTypeError,
    clip,
    join,
    over,
    rolling,
    sort,
    unique,
    where,
)


@pytest.fixture
def people():
    return T({
        "id": [1, 2, 3, 4],
        "name": ["a", "b", "c", "d"],
        "dept": ["x", "x", "y", "y"],
        "pay": [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def dept():
    return T({"dept": ["x", "y", "z"], "budget": [100.0, 200.0, 300.0]})


def test_join_inner_matches_polars(people, dept):
    got = join(people, dept, on="dept").collect().sort("id")
    expected = (
        people.collect()
        .join(dept.collect(), on="dept", how="inner")
        .sort("id")
    )
    assert got.equals(expected)


@pytest.mark.parametrize("how", ["inner", "left", "full", "semi", "anti", "cross"])
def test_join_modes_match_polars(how):
    left = T({"k": [1, 2, 3], "x": [1, 2, 3]})
    right = T({"k": [2, 3, 4], "y": [1, 2, 3]})
    kwargs = {} if how == "cross" else {"on": "k"}
    got = join(left, right, how=how, **kwargs).collect()
    if how == "cross":
        expected = left.collect().join(right.collect(), how="cross")
    else:
        expected = left.collect().join(right.collect(), on="k", how=how)
    # Row order without an explicit sort is backend-defined (lazy and eager
    # plans may disagree), so compare order-insensitively.
    assert got.shape == expected.shape
    assert got.sort(got.columns).equals(expected.sort(expected.columns))


def test_join_suffix_and_multi_key():
    left = T({"a": [1, 1, 2], "b": [1, 2, 2], "v": [1, 2, 3]})
    right = T({"a": [1, 2], "b": [2, 2], "v": [10, 20]})
    got = join(left, right, on=["a", "b"]).collect().sort("v")
    expected = (
        left.collect().join(right.collect(), on=["a", "b"], how="inner").sort("v")
    )
    assert got.equals(expected)
    assert "v_right" in got.columns


def test_join_inside_operator():
    j = join(P("A"), P("B"), on="k")
    assert j.params == ("A", "B")
    left = T({"k": [1, 2], "x": [5, 6]})
    right = T({"k": [2], "y": [7]})
    assert j(left, right).collect().equals(
        left.collect().join(right.collect(), on="k", how="inner")
    )


def test_join_errors():
    left = T({"k": [1]})
    with pytest.raises(TLTypeError):
        join(left, T({"k": [1]}), on="k", how="sideways")
    with pytest.raises(TLTypeError):
        join(left, T({"k": [1]}), on=None)
    with pytest.raises(TLTypeError):
        join(left, T({"k": [1]}), on="nope").collect()
    with pytest.raises(TLTypeError):
        join(T([1, 2]), T({"k": [1]}), on="k").collect()


def test_sort_matches_polars_and_numpy(people):
    assert sort(people, by="pay").collect()["id"].to_list() == [1, 2, 3, 4]
    assert sort(people, by="pay", descending=True).collect()["id"].to_list() == [4, 3, 2, 1]
    assert sort(people, by=["dept", "pay"]).collect()["id"].to_list() == [1, 2, 3, 4]
    s = T([3.0, 1.0, 2.0])
    assert sort(s).collect().tolist() == [1.0, 2.0, 3.0]
    assert sort(s, descending=True).collect().tolist() == [3.0, 2.0, 1.0]
    assert np.array_equal(sort(T(np.array([2, 1]))).collect(), np.array([1, 2]))
    with pytest.raises(TLTypeError):
        sort(people).collect()
    with pytest.raises(TLTypeError):
        sort(people, by="nope").collect()


def test_unique_matches_polars_and_numpy(people):
    assert unique(T([3, 1, 3, 2, 1])).collect().tolist() == [1, 2, 3]
    assert unique(people.dept).collect().to_list() == ["x", "y"]
    dupes = T({"a": [1, 1, 2], "b": ["p", "p", "q"]})
    assert unique(dupes).collect().height == 2


def test_where_matches_numpy_and_polars():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    cond = T(a > 2.0)
    assert where(cond, T(a), -1.0).collect().tolist() == [-1.0, -1.0, 3.0, 4.0]
    assert np.array_equal(
        where(cond, T(a), T(np.full(4, -1.0))).collect(),
        np.where(a > 2.0, a, -1.0),
    )
    s = T({"x": [1.0, 2.0, 3.0]})
    got = where(s.x > 1.5, s.x, 0.0).collect()
    expected = pl.when(pl.col("x") > 1.5).then(pl.col("x")).otherwise(0.0)
    assert got.to_list() == s.collect().select(expected).to_series().to_list()
    # operator form
    X = P("X")
    assert where(X > 1, X, 0)(T([0, 5])).collect().tolist() == [0, 5]


def test_where_frame_branches():
    df = T({"a": [True, False], "b": [True, True]})
    got = where(df, 1, 0).collect()
    assert got["a"].to_list() == [1, 0]
    with pytest.raises(TLShapeError):
        where(T([True, True]), T([1, 2, 3]), T([1])).collect()


def test_clip_matches_numpy_and_polars():
    a = np.array([1.0, 5.0, 10.0])
    assert clip(T(a), 2.0, 8.0).collect().tolist() == [2.0, 5.0, 8.0]
    assert clip(T(a), lo=4.0).collect().tolist() == [4.0, 5.0, 10.0]
    assert clip(T(a), hi=4.0).collect().tolist() == [1.0, 4.0, 4.0]
    s = T({"x": [1.0, 5.0, 10.0], "g": ["a", "b", "c"]})
    assert clip(s, 2.0, 8.0).collect()["x"].to_list() == [2.0, 5.0, 8.0]
    with pytest.raises(TLTypeError):
        clip(T(a))


def test_over_series_matches_polars():
    t = T({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    for op, fn in [("sum", "sum"), ("mean", "mean"), ("min", "min"), ("max", "max")]:
        got = over(t.x, t.g, op).collect().to_list()
        expected = (
            t.collect()
            .with_columns(getattr(pl.col("x"), fn)().over("g").alias("x"))["x"]
            .to_list()
        )
        assert got == pytest.approx(expected), op


def test_over_by_field_name_in_pipeline():
    t = T({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    t["dm"] = over(t.x, "g", "mean")
    assert t.collect()["dm"].to_list() == pytest.approx([1.5, 1.5, 3.5, 3.5])


def test_over_dense_matches_manual():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    keys = np.array([0, 0, 1, 1])
    assert over(T(x), T(keys), "mean").collect().tolist() == [1.5, 1.5, 3.5, 3.5]
    assert over(T(x), T(keys), "sum").collect().tolist() == [3.0, 3.0, 7.0, 7.0]
    assert over(T(x), T(keys), "min").collect().tolist() == [1.0, 1.0, 3.0, 3.0]
    assert over(T(x), T(keys), "max").collect().tolist() == [2.0, 2.0, 4.0, 4.0]
    with pytest.raises(Exception):
        over(T(x), T([0, 1])).collect()
    with pytest.raises(TLTypeError):
        over(T(x), "g").collect()


def test_rolling_matches_polars():
    s = T({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    for op, fn in [("mean", "rolling_mean"), ("sum", "rolling_sum"),
                   ("min", "rolling_min"), ("max", "rolling_max")]:
        got = rolling(s.x, 3, op).collect().to_list()
        expected = getattr(s.collect()["x"], fn)(window_size=3).to_list()
        assert got == pytest.approx(expected, nan_ok=True), op
    assert rolling(T([1.0, 2.0, 3.0]), 2, "mean").collect().tolist() == \
        pytest.approx([np.nan, 1.5, 2.5], nan_ok=True)
    with pytest.raises(TLTypeError):
        rolling(T([[1.0]]), 2).collect()


def test_scan_and_write_roundtrip(tmp_path):
    frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    src = tmp_path / "in.parquet"
    frame.write_parquet(src)
    s = T.scan_parquet(str(src))
    assert s.fields == ["a", "b"]
    assert s.collect().equals(frame)
    out = tmp_path / "out.parquet"
    assert s.write_parquet(str(out)) == str(out)
    assert pl.read_parquet(out).equals(frame)
    csv_out = tmp_path / "out.csv"
    s.write_csv(str(csv_out))
    assert pl.read_csv(csv_out).equals(frame)


def test_scan_glob(tmp_path):
    for i in range(3):
        pl.DataFrame({"a": [i]}).write_parquet(tmp_path / f"p{i}.parquet")
    s = T.scan_parquet(str(tmp_path / "p*.parquet"))
    assert s.collect().height == 3


def test_scan_csv(tmp_path):
    csv = tmp_path / "d.csv"
    pl.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).write_csv(csv)
    s = T.scan_csv(str(csv))
    assert s.collect().height == 2


def test_heterogeneous_schema_preserved_through_pipeline(people):
    D = P("D")
    senior = people[(D.pay >= 20.0)(people)]
    senior["band"] = where(senior.pay > 30.0, "hi", "mid")
    out = senior.collect()
    assert out.columns == ["id", "name", "dept", "pay", "band"]
    assert out["band"].to_list() == ["mid", "mid", "hi"]


def test_nulls_in_group_keys_match_polars():
    t = T({"x": [1.0, 2.0, 3.0], "g": ["a", None, "a"]})
    got = t.x.sum(by=t.g).collect().sort("g")
    expected = (
        t.collect().group_by("g").agg(pl.col("x").sum()).sort("g")
    )
    assert got["x"].to_list() == expected["x"].to_list()
