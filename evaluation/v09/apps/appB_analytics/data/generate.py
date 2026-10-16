"""Synthetic customers + transactions (seeded, true nulls, no network)."""
import numpy as np
import polars as pl

def main(outdir, n_cust=2000, seed=23):
    import os
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cid = np.arange(n_cust)
    tenure = rng.integers(1, 60, n_cust).astype(np.float64)
    limit = rng.uniform(1000, 20000, n_cust)
    limit[rng.random(n_cust) < 0.03] = np.nan
    customers = pl.DataFrame({
        "cust_id": cid,
        "tenure": tenure,
        "credit_limit": pl.Series("credit_limit", limit).fill_nan(None),
        "segment": np.where(cid % 3 == 0, "retail",
                     np.where(cid % 3 == 1, "smb", "enterprise")),
    })
    n_txn = 30000
    txn_cust = rng.integers(0, n_cust, n_txn)
    amount = np.abs(rng.normal(120, 80, n_txn)) * (1 + tenure[txn_cust] / 100.0)
    mask = rng.random(n_txn) < 0.04
    transactions = pl.DataFrame({
        "cust_id": txn_cust.astype(np.int64),
        "amount": pl.Series("amount", amount).fill_nan(None),
        "amount2": pl.Series("amount", np.where(mask, np.nan, amount)).fill_nan(None),
        "category": np.where(rng.random(n_txn) < 0.6, "grocery", "travel"),
    })
    customers.write_parquet(f"{outdir}/customers.parquet")
    transactions.write_parquet(f"{outdir}/transactions.parquet")
    return {"n_cust": n_cust, "n_txn": n_txn}
if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/v09b-data"))
