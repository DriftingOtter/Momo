"""End-to-end pipeline: scan -> filter -> derive -> group -> write."""
from __future__ import annotations

import polars as pl

from harness import Measurement, measure, peak_rss_mb

from tensorlang import P, T

SIZES = {"small": 100_000, "medium": 1_000_000, "large": 5_000_000}


def run(size: str, path: str, outdir: str) -> list[Measurement]:
    out: list[Measurement] = []
    D = P("D")

    def tl_pipeline():
        sales = T.scan_parquet(path)
        valid = ((D.amount > 100) & (D.status == "completed"))
        filtered = sales[valid(sales)]
        filtered["net"] = (D.amount - D.fee)(filtered)
        summary = filtered.net.sum(by=filtered.region)
        # Single execution: stream straight to Parquet (no extra collect).
        summary.write_parquet(f"{outdir}/tl-revenue-{size}.parquet")

    def pl_pipeline():
        (
            pl.scan_parquet(path)
            .filter((pl.col("amount") > 100) & (pl.col("status") == "completed"))
            .with_columns((pl.col("amount") - pl.col("fee")).alias("net"))
            .group_by("region")
            .agg(pl.col("net").sum())
            .sort("region")
            .collect()
            .write_parquet(f"{outdir}/pl-revenue-{size}.parquet")
        )

    secs, alloc, _ = measure(tl_pipeline)
    _, _, _ = measure(pl_pipeline)
    # Correctness out of band: written files must agree (float aggregates
    # compared approximately: intra-group summation order is engine-defined).
    from polars.testing import assert_frame_equal

    got = pl.read_parquet(f"{outdir}/tl-revenue-{size}.parquet").sort("region")
    ref = pl.read_parquet(f"{outdir}/pl-revenue-{size}.parquet").sort("region")
    assert_frame_equal(got, ref, check_exact=False, abs_tol=1e-6, rel_tol=1e-9)
    sales = T.scan_parquet(path)
    filtered = sales[((D.amount > 100) & (D.status == "completed"))(sales)]
    filtered["net"] = (D.amount - D.fee)(filtered)
    prof = filtered.net.sum(by=filtered.region).profile()
    out.append(Measurement("pipeline-write", size, "tensorlang", secs, alloc,
                           peak_rss_mb(),
                           backend="polars-lazy" if prof.lowered else "materialized",
                           materializations=prof.scan_executions))
    secs, alloc, _ = measure(pl_pipeline)
    out.append(Measurement("pipeline-write", size, "native", secs, alloc,
                           peak_rss_mb(), backend="polars-lazy"))
    return out
