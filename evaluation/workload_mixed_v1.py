"""M0 workload (v1, frozen): mixed pipeline with explicit conversion.

Original implementation using manual to_numpy()/T() conversion.
Preserved for attribution; see workload_mixed.py (v2, lazy bridge).

Covers: parquet scans, mask filtering, joins, derived columns, grouped
sum/mean/min/max, left-join assembly with nulls, key sorting, the
frame→dense conversion boundary, fitted impute/standardize/PCA with
train/holdout isolation, and dense ridge regression via `solve` on the
converted features.

M0 questions under test (see evaluation/M0_INTEROP.md when written):
P1 lazy-frame input to fitting, P2 re-collect cost, P3 conversion copy
semantics, P4 int-null dtype promotion, P5 stage-wise timing/RSS.
"""
from __future__ import annotations

import numpy as np

FEATURE_COLS = ["net", "net_mean", "net_min", "net_max", "age",
                "tenure", "count"]


def make_data(cust_path: str, txn_path: str, n_cust: int = 100_000,
              n_txn: int = 1_000_000, seed: int = 71) -> None:
    import polars as pl

    rng = np.random.default_rng(seed)
    cust = pl.DataFrame({
        "customer_id": np.arange(n_cust),
        "region": rng.choice(["N", "S", "E", "W"], n_cust).tolist(),
        "segment": rng.choice(["a", "b", "c"], n_cust).tolist(),
        "age": np.where(rng.random(n_cust) < 0.1, np.nan,
                        rng.uniform(18, 80, n_cust)),
        "tenure": pl.Series("tenure", [
            int(x) if rng.random() > 0.1 else None
            for x in rng.integers(0, 120, n_cust)], dtype=pl.Int64),
    })
    txn = pl.DataFrame({
        "txn_id": np.arange(n_txn),
        "customer_id": rng.integers(0, n_cust, n_txn),
        "amount": np.where(rng.random(n_txn) < 0.05, np.nan,
                           rng.exponential(100, n_txn)),
        "fee": rng.uniform(0, 5, n_txn),
        "category": rng.choice(["food", "tech", "fuel", "misc"], n_txn).tolist(),
        "status": rng.choice(["completed", "pending", "failed"], n_txn,
                             p=[0.8, 0.1, 0.1]).tolist(),
    })
    cust.write_parquet(cust_path)
    txn.write_parquet(txn_path)


# -- TensorLang implementation -------------------------------------------------


def tl_features(cust_path: str, txn_path: str):
    """Relational stage; returns a collected feature frame (Lazy)."""
    from tensorlang import T

    cust = T.scan_parquet(cust_path)
    txn = T.scan_parquet(txn_path)
    done = txn[(txn.status == "completed") & (txn.amount > 1.0)]
    joined = done.join(cust, on="customer_id")
    joined["net"] = joined.amount - joined.fee
    total = joined.net.sum(by=joined.customer_id)
    mean = joined.net.mean(by=joined.customer_id)
    lo = joined.net.min(by=joined.customer_id)
    hi = joined.net.max(by=joined.customer_id)
    feats = (total.join(mean, on="customer_id", suffix="_mean")
             .join(lo, on="customer_id", suffix="_min")
             .join(hi, on="customer_id", suffix="_max"))
    cnt = joined.count(by=joined.customer_id)
    feats = feats.join(cnt, on="customer_id")
    full = cust.join(feats, on="customer_id", how="left")
    return full.sort("customer_id")


def tl_convert(frame):
    """The investigated boundary: frame -> numpy -> Tensor (explicit)."""
    from tensorlang import T

    arr = frame.select(FEATURE_COLS).to_numpy()
    assert arr.dtype == np.float64, arr.dtype
    return T(arr), arr


def tl_split_train_holdout(matrix: np.ndarray, train_frac: float = 0.8,
                           seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(matrix.shape[0])
    cut = int(matrix.shape[0] * train_frac)
    return matrix[idx[:cut]], matrix[idx[cut:]]


def tl_ml_pipeline():
    from tensorlang import impute, pca, standardize

    return impute("mean") >> standardize() >> pca(3)


def tl_ridge(components, target, lam: float = 1.0):
    """Ridge regression on PCA components via normal equations + solve.

    The target is mean-imputed with the training mean (same policy as
    the feature pipeline); NaN targets would otherwise poison solve().
    """
    from tensorlang import T, solve

    C = np.asarray(components)
    y = np.asarray(target, dtype=np.float64)
    y = np.where(np.isnan(y), np.nanmean(y), y)
    yc = y - y.mean()
    n = C.shape[1]
    A = C.T @ C + lam * np.eye(n)
    b = C.T @ yc
    return T(A), T(b), float(y.mean())


def tl_ridge_weights(A, b):
    from tensorlang import solve

    return np.asarray(solve(A, b).collect())


# -- native Polars + scikit-learn reference --------------------------------------


def np_features(cust_path: str, txn_path: str):
    import polars as pl

    cust = pl.read_parquet(cust_path)
    txn = pl.read_parquet(txn_path)
    done = txn.filter((pl.col("status") == "completed")
                      & (pl.col("amount") > 1.0))
    joined = done.join(cust, on="customer_id").with_columns(
        (pl.col("amount") - pl.col("fee")).alias("net"))
    grouped = joined.group_by("customer_id").agg(
        pl.col("net").sum().alias("net"),
        pl.col("net").mean().alias("net_mean"),
        pl.col("net").min().alias("net_min"),
        pl.col("net").max().alias("net_max"),
        pl.len().alias("count"))
    full = cust.join(grouped, on="customer_id", how="left")
    return full.sort("customer_id")
