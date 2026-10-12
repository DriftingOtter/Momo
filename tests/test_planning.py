"""Priorities 3-4 tests: single-query lowering, execution boundaries,
explain() and profile()."""
import polars as pl
import pytest

from tensorlang import P, T
from tensorlang.runtime import polars_plan as _lower


@pytest.fixture
def sales_parquet(tmp_path):
    frame = pl.DataFrame({
        "amount": [80, 180, 240, 110, 300, 50],
        "fee": [2, 10, 20, 5, 15, 1],
        "status": ["pending", "completed", "completed", "completed",
                   "completed", "pending"],
        "region": ["East", "East", "West", "West", "East", "West"],
    })
    path = tmp_path / "sales.parquet"
    frame.write_parquet(path)
    return str(path), frame


@pytest.fixture
def pipeline(sales_parquet):
    path, _ = sales_parquet
    from tensorlang import T

    sales = T.scan_parquet(path)
    D = P("D")
    valid = ((D.amount > 100) & (D.status == "completed"))
    filtered = sales[valid(sales)]
    net = D.amount - D.fee
    filtered["net"] = net(filtered)
    return filtered.net.sum(by=filtered.region)


def test_pipeline_stays_lazy_until_boundary(sales_parquet):
    from tensorlang import Lazy

    path, _ = sales_parquet
    sales = T.scan_parquet(path)
    D = P("D")
    out = sales[(D.amount > 1)(sales)]
    assert isinstance(out, Lazy)
    out["z"] = 5
    assert isinstance(out, Lazy)  # WithField, not a materialization


def test_pipeline_matches_native_polars(pipeline, sales_parquet):
    _, frame = sales_parquet
    got = pipeline.collect().sort("region")
    expected = (
        frame.filter((pl.col("amount") > 100) & (pl.col("status") == "completed"))
        .with_columns((pl.col("amount") - pl.col("fee")).alias("net"))
        .group_by("region")
        .agg(pl.col("net").sum())
        .sort("region")
    )
    assert got.equals(expected)


def test_explain_shows_single_lowered_query(pipeline):
    text = pipeline.explain()
    assert "logical plan" in text
    assert "scan_parquet" in text
    assert "polars-lazy" in text
    assert "SELECTION" in text  # predicate pushdown delegated to Polars
    assert "SCAN" in text


def test_explain_fallback_is_explicit():
    t = T({"a": [1, 2]})
    text = (t * 2).explain()
    assert "materialized fallback" in text


def test_profile_reports_measurements(pipeline):
    prof = pipeline.profile()
    assert prof.lowered is True
    assert prof.backend == "polars-lazy"
    assert prof.elapsed_ms > 0
    assert (prof.rows, prof.cols) == (2, 2)
    assert prof.nodes > 0
    assert prof.scan_executions == 1  # one optimized query, one scan
    assert prof.peak_python_alloc_mb >= 0
    assert "backend=polars-lazy" in str(prof)


def test_profile_fallback_backend():
    prof = (T([1.0, 2.0]) * 3).profile()
    assert prof.lowered is False
    assert prof.backend == "materialized"
    assert prof.shape == (2,)


def test_sink_streams_without_materializing(pipeline, tmp_path):
    out = tmp_path / "rev.parquet"
    ok, plan = _lower.try_sink(pipeline.root, str(out), "parquet")
    assert ok is True
    assert pl.read_parquet(out).equals(pipeline.collect())


def test_write_parquet_uses_sink(pipeline, tmp_path):
    out = tmp_path / "rev.parquet"
    assert pipeline.write_parquet(str(out)) == str(out)
    assert pl.read_parquet(out).equals(pipeline.collect())


def test_join_pipeline_lowers(tmp_path):
    l = tmp_path / "l.parquet"
    r = tmp_path / "r.parquet"
    pl.DataFrame({"k": [1, 2], "x": [1, 2]}).write_parquet(l)
    pl.DataFrame({"k": [2, 3], "y": [1, 2]}).write_parquet(r)
    from tensorlang import join

    lazy = join(T.scan_parquet(str(l)), T.scan_parquet(str(r)), on="k")
    text = lazy.explain()
    assert "polars-lazy" in text
    assert "JOIN" in text
    assert lazy.collect().height == 1


def test_sort_unique_lower(tmp_path):
    path = tmp_path / "s.parquet"
    pl.DataFrame({"a": [2, 1, 2], "b": ["x", "y", "x"]}).write_parquet(path)
    from tensorlang import sort, unique

    s = T.scan_parquet(str(path))
    assert "SORT" in sort(s, by="a").explain()
    assert "UNIQUE" in unique(s).explain()
    assert sort(s, by="a").collect()["a"].to_list() == [1, 2, 2]
    assert unique(s).collect().height == 2


def test_scan_is_never_read_by_construction(tmp_path):
    missing = tmp_path / "ghost.parquet"
    s = T.scan_parquet(str(missing))
    D = P("D")
    lazy = s[(D.amount > 1)(s)]
    assert "scan_parquet" in lazy.explain()  # planned, not executed
    with pytest.raises(Exception):
        lazy.collect()  # only the boundary reads


def test_operator_and_tensor_explain():
    f = (P("X") ** 2) >> (P("Y") + 1)
    text = f.explain()
    assert "params=['X']" in text
    assert T([1]).explain().startswith("logical plan")


def test_inplace_after_derive_rejected(sales_parquet):
    from tensorlang import TLMutationError

    path, _ = sales_parquet
    sales = T.scan_parquet(path)
    D = P("D")
    view = sales[(D.amount > 1)(sales)]
    view["z"] = 1  # re-roots the graph: no single mutable target remains
    with pytest.raises(TLMutationError):
        (P("Q") + 1)(view).inplace()
