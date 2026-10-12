"""Phase 5 tests: structured tensors (Polars backend) vs Polars itself."""
import numpy as np
import polars as pl
import pytest

from tensorlang import (
    P,
    T,
    TLShapeError,
    TLTypeError,
    TLUnsupportedError,
    drop_null,
    fill_null,
    is_null,
    not_null,
)


@pytest.fixture
def sales():
    return T({
        "amount": [80, 180, 240, 110],
        "fee": [2, 10, 20, 5],
        "status": ["pending", "completed", "completed", "completed"],
        "region": ["East", "East", "West", "West"],
    })


@pytest.fixture
def frame(sales):
    return sales.collect()


def test_structured_metadata(sales):
    assert sales.kind == "frame"
    assert sales.fields == ["amount", "fee", "status", "region"]
    assert sales.schema["amount"] == pl.Int64
    assert sales.shape == (4, 4)


def test_field_access_dot_and_bracket(sales, frame):
    assert sales.amount.collect().to_list() == frame["amount"].to_list()
    assert sales["amount"].collect().to_list() == frame["amount"].to_list()
    sub = sales[["amount", "region"]].collect()
    assert sub.columns == ["amount", "region"]
    with pytest.raises(AttributeError):
        sales.nope


def test_row_filtering_with_predicate(sales):
    D = P("D")
    eligible = ((D.amount > 100) & (D.status == "completed"))
    filtered = sales[eligible(sales)].collect()
    assert filtered["amount"].to_list() == [180, 240, 110]
    assert filtered.height == 3


def test_derived_field_assignment(sales):
    D = P("D")
    net = D.amount - D.fee
    filtered = sales[(D.amount > 100)(sales)]
    filtered["net"] = net(filtered)
    assert filtered.collect()["net"].to_list() == [170, 220, 105]
    assert filtered.collect().columns == ["amount", "fee", "status", "region", "net"]


def test_grouped_sum_matches_polars(sales, frame):
    D = P("D")
    filtered = sales[(D.amount > 100)(sales)]
    filtered["net"] = (D.amount - D.fee)(filtered)
    got = filtered.net.sum(by=filtered.region).collect()
    expected = (
        frame.filter(pl.col("amount") > 100)
        .with_columns((pl.col("amount") - pl.col("fee")).alias("net"))
        .group_by("region")
        .agg(pl.col("net").sum())
        .sort("region")
    )
    assert got.sort("region").equals(expected)


def test_grouped_mean(sales, frame):
    got = sales.amount.mean(by=sales.region).collect()
    expected = (
        frame.group_by("region").agg(pl.col("amount").mean()).sort("region")
    )
    assert got["region"].to_list() == expected["region"].to_list()
    assert got["amount"].to_list() == pytest.approx(expected["amount"].to_list())


def test_whole_frame_reduction_preserves_labels(sales):
    out = sales.sum().collect()
    assert isinstance(out, pl.DataFrame)
    assert out["amount"].to_list() == [610]
    assert out["fee"].to_list() == [37]


def test_series_scalar_reduction(sales):
    assert sales.amount.sum().collect() == 610
    assert sales.amount.mean().collect() == pytest.approx(152.5)


def test_arithmetic_preserves_non_numeric_fields(sales, frame):
    doubled = (sales * 2).collect()
    assert doubled.columns == frame.columns  # order preserved
    assert doubled["amount"].to_list() == [160, 360, 480, 220]
    assert doubled["status"].to_list() == frame["status"].to_list()
    assert doubled["region"].to_list() == frame["region"].to_list()


def test_comparison_produces_boolean_frame(sales):
    out = (sales > 100).collect()
    assert out["amount"].to_list() == [False, True, True, True]
    assert "status" not in out.columns  # only applicable numerical fields


def test_combining_field_predicates_is_explicit(sales):
    D = P("D")
    both = ((D.amount > 100) & (D.fee < 10))(sales).collect()
    assert both.to_list() == [False, False, False, True]


def test_ambiguous_frame_mask_rejected(sales):
    with pytest.raises((TLShapeError, TLTypeError)):
        sales[sales > 100].collect()


def test_matmul_structured_rejected(sales):
    with pytest.raises(TLTypeError):
        (sales @ sales).collect()


def test_missing_values():
    t = T({"x": [1.0, None, 3.0], "g": ["a", "a", "b"]})
    assert is_null(t).collect()["x"].to_list() == [False, True, False]
    assert not_null(t).collect()["x"].to_list() == [True, False, True]
    assert fill_null(t, 0.0).collect()["x"].to_list() == [1.0, 0.0, 3.0]
    assert drop_null(t).collect().height == 2
    # dense float NaN handling matches NumPy
    d = T([1.0, float("nan"), 3.0])
    assert is_null(d).collect().tolist() == [False, True, False]
    assert drop_null(d).collect().tolist() == [1.0, 3.0]
    assert fill_null(d, -1.0).collect().tolist() == [1.0, -1.0, 3.0]


def test_null_ops_inside_operator_definitions():
    clean = drop_null(P("X"))
    assert clean(T([1.0, float("nan")])).collect().tolist() == [1.0]


def test_parquet_csv_roundtrip(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    pq = tmp_path / "data.parquet"
    csv = tmp_path / "data.csv"
    df.write_parquet(pq)
    df.write_csv(csv)
    assert T.read(str(pq)).collect().equals(df)
    assert T.read(str(csv)).collect().equals(df)
    with pytest.raises(TLTypeError):
        T.read(str(tmp_path / "data.xlsx"))


def test_mixed_dense_frame_combination_rejected(sales):
    with pytest.raises(TLTypeError):
        (sales + T([1, 2, 3, 4])).collect()


def test_unknown_field_rejected(sales):
    D = P("D")
    with pytest.raises(TLTypeError):
        (D.nope)(sales).collect()
