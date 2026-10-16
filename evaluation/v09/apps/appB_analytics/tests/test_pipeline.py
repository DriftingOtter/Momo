"""App B tests: counts, nulls, isolation, determinism, reuse (public API)."""
import os

import polars as pl
import pytest

from tensorlang import T, run_file, run_source

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_main():
    return run_file(os.path.join(APP, "main.tl"))


def test_row_counts_and_nulls():
    ns = run_main()
    assert ns["enriched"].collect().height == 21954
    assert ns["feat"].collect().height == 2000
    assert ns["clean"].collect().height == 1950
    assert ns["feat"].collect()["credit_limit"].null_count() == 50


def test_train_valid_isolation():
    ns = run_main()
    tr = set(ns["train"].collect()["cust_id"].to_list())
    va = set(ns["valid"].collect()["cust_id"].to_list())
    assert len(tr) == 1461 and len(va) == 489
    assert not (tr & va)
    assert all(c < 1500 for c in tr) and all(c >= 1500 for c in va)


def test_metrics_match_reference():
    run_main()
    got = pl.read_parquet(os.path.join(APP, "out_metrics.parquet"))
    mse = got.filter(pl.col("metric") == "mse")["value"][0]
    assert mse == pytest.approx(116777.2719754062, rel=1e-9)


def test_big_ticket_exact_and_stable():
    # M1: combined cross-file predicate is exact and deterministic.
    counts = set()
    for _ in range(3):
        ns = run_main()
        counts.add(ns["clean"][ns["big_ticket"](ns["clean"])].collect().height)
    assert counts == {1774}


def test_deterministic_metrics():
    run_main()
    first = pl.read_parquet(os.path.join(APP, "out_metrics.parquet"))["value"].to_list()
    run_main()
    second = pl.read_parquet(os.path.join(APP, "out_metrics.parquet"))["value"].to_list()
    assert first == pytest.approx(second, rel=1e-12)


def test_reexport_identity_within_session():
    # Single spelling per file (F-A2): probe lives beside the stages and
    # uses bare sibling imports only.
    probe = os.path.join(APP, "features", "_probe_reexp.tl")
    with open(probe, "w") as fh:
        fh.write("from transforms import is_high\n"
                 "from aggregations import high_value\n"
                 "same = is_high.uid == high_value.uid\n")
    try:
        ns = run_source(open(probe).read(), filename=probe)
    finally:
        os.remove(probe)
    assert ns["same"] is True


def test_missing_value_predicates():
    ns = run_main()
    feat = ns["feat"].collect()
    assert feat["credit_limit"].null_count() == 50
    clean = ns["clean"].collect()
    assert clean["credit_limit"].null_count() == 0
    assert clean.height == feat.height - 50
