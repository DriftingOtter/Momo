"""Workload 2: customer/transaction pipeline (structured data + ML).

Covers: parquet scans, mask filtering, joins, derived columns, grouped
reductions, left-join feature assembly with nulls, an explicit
structured->dense materialization boundary (no lazy frame-to-dense
bridge exists), fitted impute/standardize/PCA with train/held-out
isolation.

Recorded as findings (not hidden): the collect() boundary between the
relational and ML stages; NaN-poisoned group sums (verified identical
to native Polars semantics); no lazy count aggregation (features use
sum/mean only).
"""
from __future__ import annotations

import numpy as np

FEATURES = ["total_net", "mean_net", "age"]


def make_data(cust_path: str, txn_path: str, n_cust: int = 20000,
              n_txn: int = 200000, seed: int = 7) -> None:
    import polars as pl

    rng = np.random.default_rng(seed)
    cust = pl.DataFrame({
        "customer_id": np.arange(n_cust),
        "region": rng.choice(["N", "S", "E", "W"], n_cust).tolist(),
        "age": np.where(rng.random(n_cust) < 0.1, np.nan,
                        rng.uniform(18, 80, n_cust)),
    })
    txn = pl.DataFrame({
        "txn_id": np.arange(n_txn),
        "customer_id": rng.integers(0, n_cust, n_txn),
        "amount": np.where(rng.random(n_txn) < 0.05, np.nan,
                           rng.exponential(100, n_txn)),
        "fee": rng.uniform(0, 5, n_txn),
        "status": rng.choice(["completed", "pending", "failed"], n_txn,
                             p=[0.8, 0.1, 0.1]).tolist(),
    })
    cust.write_parquet(cust_path)
    txn.write_parquet(txn_path)


# -- TensorLang implementation -------------------------------------------------


def tl_features(cust_path: str, txn_path: str):
    """Relational stage: returns a collected feature frame (Lazy)."""
    from tensorlang import T

    cust = T.scan_parquet(cust_path)
    txn = T.scan_parquet(txn_path)
    done = txn[txn.status == "completed"]
    joined = done.join(cust, on="customer_id")
    joined["net"] = joined.amount - joined.fee
    total = joined.net.sum(by=joined.customer_id)
    mean = joined.net.mean(by=joined.customer_id)
    feats = total.join(mean, on="customer_id", suffix="_mean")
    full = cust.join(feats, on="customer_id", how="left")
    # Deterministic row order: join/group output order is not guaranteed
    # across processes; key-sorted frames keep held-out splits comparable.
    return full.sort("customer_id")


def tl_ml_pipeline():
    from tensorlang import impute, pca, standardize

    return impute("mean") >> standardize() >> pca(2)


def tl_split_train_holdout(frame, train_frac: float = 0.8, seed: int = 0):
    import polars as pl

    rng = np.random.default_rng(seed)
    idx = rng.permutation(frame.height)
    cut = int(frame.height * train_frac)
    return frame[idx[:cut]], frame[idx[cut:]]


# -- native Polars + scikit-learn reference --------------------------------------


def np_features(cust_path: str, txn_path: str):
    import polars as pl

    cust = pl.read_parquet(cust_path)
    txn = pl.read_parquet(txn_path)
    done = txn.filter(pl.col("status") == "completed")
    joined = done.join(cust, on="customer_id").with_columns(
        (pl.col("amount") - pl.col("fee")).alias("net"))
    total = joined.group_by("customer_id").agg(
        pl.col("net").sum().alias("total_net"))
    mean = joined.group_by("customer_id").agg(
        pl.col("net").mean().alias("mean_net"))
    feats = total.join(mean, on="customer_id")
    full = cust.join(feats, on="customer_id", how="left")
    return full.sort("customer_id")


TL_FEATURE_COLS = ["net", "net_mean", "age"]
NP_FEATURE_COLS = ["total_net", "mean_net", "age"]
