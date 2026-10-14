"""Workload 3: out-of-core z-score normalization (persistent Zarr).

Covers: lazy source construction, reusable scoped operators, chunk
alignment, memory-aware planning, two-pass reductions, incremental
staged writes, execution introspection. Datasets exceed the working
budget by design; only v0.5.3-supported operators are used (anything
else is recorded as a v0.6 candidate, never worked around unsafely).
"""
from __future__ import annotations

import numpy as np


def make_data(path: str, rows: int, cols: int, chunk_rows: int,
              seed: int = 0) -> None:
    import zarr

    rng = np.random.default_rng(seed)
    arr = zarr.open_array(path, mode="w", shape=(rows, cols),
                          chunks=(chunk_rows, cols), dtype="float64")
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        arr[start:stop] = rng.normal(10.0, 3.0, (stop - start, cols))


# -- TensorLang implementation -------------------------------------------------


def tl_normalize(cust_path: str, out_path: str, budget_bytes: int,
                 workers: int) -> dict:
    """Two-pass z-score with staged streaming write. Returns the admission
    record, timings filled by the driver harness (kept here for LOC)."""
    from tensorlang import P, T, set_chunk_policy

    set_chunk_policy(memory_budget_bytes=budget_bytes, max_workers=workers)
    try:
        X = P("X")
        data = T.scan_zarr(cust_path)
        mu = float(data.mean().collect())
        sigma = float(data.std().collect())
        norm = (X - mu) / (sigma if sigma != 0 else 1.0)
        out = norm(data)
        plan = out.explain()
        out.write_zarr(out_path, overwrite=True)
        check = T.scan_zarr(out_path)
        return {"mu": mu, "sigma": sigma,
                "mean_after": float(check.mean().collect()),
                "std_after": float(check.std().collect()),
                "plan": plan}
    finally:
        from tensorlang import clear_chunk_policy

        clear_chunk_policy()


# -- native Dask/Zarr reference ---------------------------------------------------


def np_normalize(src_path: str, out_path: str, workers: int) -> dict:
    import dask.array as da

    x = da.from_zarr(src_path)
    mu = float(x.mean().compute(num_workers=workers))
    sigma = float(x.std().compute(num_workers=workers))
    normed = ((x - mu) / (sigma if sigma != 0 else 1.0))
    da.to_zarr(normed, out_path, overwrite=True,
               compute=False).compute(num_workers=workers)
    check = da.from_zarr(out_path)
    return {"mu": mu, "sigma": sigma,
            "mean_after": float(check.mean().compute(num_workers=workers)),
            "std_after": float(check.std().compute(num_workers=workers))}
