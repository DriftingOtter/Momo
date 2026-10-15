"""Deterministic synthetic experiment data (no network, seeded)."""
import numpy as np
import polars as pl

def main(outdir: str, n_exp: int = 50, n_meas: int = 2000, seed: int = 7) -> dict:
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    exp_ids = np.repeat(np.arange(n_exp), n_meas)
    t = np.tile(np.linspace(0, 10, n_meas), n_exp)
    temp = 20.0 + 5.0 * np.sin(t / 2.0) + rng.normal(0, 0.5, n_exp * n_meas)
    calib = rng.uniform(0.9, 1.1, n_exp)[exp_ids]
    meas = calib * (100.0 + 3.0 * temp + rng.normal(0, 2.0, n_exp * n_meas))
    mask = rng.random(n_exp * n_meas) < 0.05
    meas = pl.Series("meas", np.where(mask, np.nan, meas)).fill_nan(None)
    obs = pl.DataFrame({
        "experiment_id": exp_ids.astype(np.int64),
        "t": t,
        "meas": meas,
        "temp": temp,
        "category": np.where(exp_ids % 2 == 0, "A", "B"),
    })
    params = pl.DataFrame({
        "experiment_id": np.arange(n_exp),
        "gain": rng.uniform(0.9, 1.1, n_exp),
        "offset": rng.uniform(-1.0, 1.0, n_exp),
    })
    obs.write_parquet(f"{outdir}/observations.parquet")
    params.write_parquet(f"{outdir}/params.parquet")
    return {"n_exp": n_exp, "n_meas": n_meas, "rows": n_exp * n_meas}

if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/m0v8/app/data"))
