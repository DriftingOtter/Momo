"""Deterministic benchmark datasets (seeded; regenerate with make_data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).resolve().parent / "data"


def make_sales(path: Path, rows: int, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=4.5, sigma=0.8, size=rows).round(2)
    fee = (rng.random(rows) * 20).round(2)
    status = np.where(rng.random(rows) < 0.7, "completed", "pending").astype(object)
    status[rng.random(rows) < 0.02] = None  # realistic missing values
    regions = np.array(["East", "West", "North", "South"])
    region = regions[rng.integers(0, 4, rows)]
    frame = pl.DataFrame({
        "id": np.arange(rows, dtype=np.int64),
        "amount": amount,
        "fee": fee,
        "status": status.tolist(),
        "region": region.tolist(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def make_dense(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    return (
        rng.random(n) + 0.5,
        rng.random(n) + 0.5,
        rng.random(n) + 0.5,
        rng.random(n) + 0.5,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "sales-1M.parquet"))
    args = parser.parse_args()
    make_sales(Path(args.out), args.rows)
    print(f"wrote {args.out} ({args.rows} rows)")
