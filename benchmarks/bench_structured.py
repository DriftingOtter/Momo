"""Structured workloads: filter / derive / group vs native Polars."""
from __future__ import annotations

import polars as pl

from harness import Measurement, measure, peak_rss_mb

from tensorlang import P, T

SIZES = {"small": 100_000, "medium": 1_000_000, "large": 5_000_000}


def run(size: str, path: str) -> list[Measurement]:
    n = SIZES[size]
    out: list[Measurement] = []
    frame = pl.read_parquet(path)
    assert frame.height == n
    D = P("D")

    # -- two-predicate filter ------------------------------------------------
    t = T(frame)

    def tl_filter():
        return t[(D.amount > 100)(t) & (D.status == "completed")(t)].collect()

    def pl_filter():
        return frame.filter(
            (pl.col("amount") > 100) & (pl.col("status") == "completed"))

    secs, alloc, got = measure(tl_filter)
    _, _, ref = measure(pl_filter)
    assert got.equals(ref)
    out.append(Measurement("filter-2-predicates", size, "tensorlang", secs,
                           alloc, peak_rss_mb(), backend="materialized",
                           materializations=1))
    secs, alloc, _ = measure(pl_filter)
    out.append(Measurement("filter-2-predicates", size, "native", secs, alloc,
                           peak_rss_mb(), backend="polars"))

    # -- filter + derived field, lazy scan ------------------------------------
    def tl_derive():
        sales = T.scan_parquet(path)
        filt = sales[((D.amount > 100) & (D.status == "completed"))(sales)]
        filt["net"] = (D.amount - D.fee)(filt)
        return filt.collect()

    def pl_derive():
        return (
            pl.scan_parquet(path)
            .filter((pl.col("amount") > 100) & (pl.col("status") == "completed"))
            .with_columns((pl.col("amount") - pl.col("fee")).alias("net"))
            .collect()
        )

    secs, alloc, got = measure(tl_derive)
    _, _, ref = measure(pl_derive)
    assert got.equals(ref)
    prof = None
    sales = T.scan_parquet(path)
    filt = sales[((D.amount > 100) & (D.status == "completed"))(sales)]
    filt["net"] = (D.amount - D.fee)(filt)
    prof = filt.profile()
    out.append(Measurement("filter-derive", size, "tensorlang", secs, alloc,
                           peak_rss_mb(),
                           backend="polars-lazy" if prof.lowered else "materialized",
                           materializations=prof.scan_executions))
    secs, alloc, _ = measure(pl_derive)
    out.append(Measurement("filter-derive", size, "native", secs, alloc,
                           peak_rss_mb(), backend="polars-lazy"))

    # -- grouped aggregation ---------------------------------------------------
    def tl_group():
        sales = T.scan_parquet(path)
        filt = sales[((D.amount > 100) & (D.status == "completed"))(sales)]
        filt["net"] = (D.amount - D.fee)(filt)
        return filt.net.sum(by=filt.region).collect()

    def pl_group():
        return (
            pl.scan_parquet(path)
            .filter((pl.col("amount") > 100) & (pl.col("status") == "completed"))
            .with_columns((pl.col("amount") - pl.col("fee")).alias("net"))
            .group_by("region")
            .agg(pl.col("net").sum())
            .sort("region")
            .collect()
        )

    secs, alloc, got = measure(tl_group)
    _, _, ref = measure(pl_group)
    from polars.testing import assert_frame_equal

    assert_frame_equal(got.sort("region"), ref, check_exact=False,
                       abs_tol=1e-6, rel_tol=1e-9)
    out.append(Measurement("grouped-sum", size, "tensorlang", secs, alloc,
                           peak_rss_mb(), backend="polars-lazy",
                           materializations=1))
    secs, alloc, _ = measure(pl_group)
    out.append(Measurement("grouped-sum", size, "native", secs, alloc,
                           peak_rss_mb(), backend="polars-lazy"))
    return out
