"""Native Polars/NumPy/SciPy mirror of driver.py (same data, same math)."""
import os
import time

import numpy as np
import polars as pl
from scipy.linalg import solve as _solve
from sklearn.preprocessing import StandardScaler

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(APP, "data")
OUT = os.path.join(APP, "out")
os.makedirs(OUT, exist_ok=True)
_phases = {}


def phase(name, fn):
    t0 = time.perf_counter()
    out = fn()
    _phases[name] = time.perf_counter() - t0
    return out


def main():
    obs = phase("load", lambda: pl.read_parquet(f"{DATA}/observations.parquet"))
    params = pl.read_parquet(f"{DATA}/params.parquet")
    base = phase("filter", lambda: obs.filter(pl.col("meas").is_not_null()))
    joined = phase("join", lambda: base.join(params, on="experiment_id"))
    cal = phase("calibrate", lambda: (
        joined["meas"].to_numpy() * joined["gain"].to_numpy()
        + joined["offset"].to_numpy()))
    enriched = phase("derive", lambda: joined.with_columns(
        pl.Series("cal", cal)))
    grp = phase("group", lambda: enriched.group_by("experiment_id").agg(
        pl.col("cal").mean().alias("m"),
        pl.len().alias("n")))
    sub = phase("convert", lambda: enriched.select(["cal", "temp"]).to_numpy())
    scaler = StandardScaler()
    Z = phase("standardize", lambda: scaler.fit_transform(sub))
    y = Z[:, 0]
    X = np.column_stack([np.ones(Z.shape[0]), Z[:, 1]])
    w = phase("solve", lambda: _solve(X.T @ X + 0.1 * np.eye(2), X.T @ y))
    print("temp coefficient (standardized):", w[1])
    pl.DataFrame({"term": ["intercept", "temp_coef"], "w": w.tolist()}
                 ).write_parquet(f"{OUT}/coefficients_native.parquet")
    return {"coef_temp": float(w[1]), "phases": dict(_phases),
            "rows": enriched.height}


if __name__ == "__main__":
    import json
    result = main()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in result["phases"].items()}, indent=1))
    print("coef_temp =", round(result["coef_temp"], 4), "rows =", result["rows"])
