"""Structure-B driver: Python-hosted multi-file TensorLang application.

Each .tl stage defines operators/descriptors (declarative); this driver
loads them via run_file, composes across files, and executes with
per-stage timing. The .tl files cannot import each other (no loader);
composition happens here through first-class operator objects.
"""
import os
import sys
import time

try:
    import tensorlang
except ImportError:  # pragma: no cover - source-checkout fallback
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", "..", "src"))

from tensorlang import T, not_null, run_file, solve, standardize

APP = os.path.dirname(os.path.abspath(__file__))
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
    stages = phase("load_stages", lambda: {
        "cal": run_file(os.path.join(APP, "operators/calibration.tl"))["calibrate"],
        "zscore": run_file(os.path.join(APP, "operators/transforms.tl"))["zscore"],
        "src": run_file(os.path.join(APP, "inputs/sources.tl")),
    })
    obs = stages["src"]["obs"]
    params = stages["src"]["params"]
    base = phase("filter", lambda: obs[not_null(obs.meas)])
    joined = phase("join", lambda: base.join(params, on="experiment_id"))
    cal = stages["cal"]
    # Row-level calibration with the cross-file reusable operator,
    # derived INSIDE the lazy graph (positional attachment across
    # separate executions would be unsafe: join order is backend-defined
    # and differs from native polars).
    joined["cal"] = cal(joined.meas, joined.gain, joined.offset)
    enriched = phase("derive", lambda: joined.collect())
    import polars as pl

    assert isinstance(enriched, pl.DataFrame)
    # Per-experiment aggregates of calibrated measurements.
    import tensorlang as _tl

    agg_base = _tl.T(enriched)
    means = phase("group_mean", lambda: agg_base.cal.mean(
        by=agg_base.experiment_id).collect())
    counts = phase("group_count", lambda: agg_base.count(
        by="experiment_id").collect())
    dense = phase("convert", lambda: agg_base.to_dense(
        columns=["cal", "temp"]).collect())
    # Dense modeling input through the v0.7 bridge (lazy, admitted).
    import numpy as np

    sub = np.asarray(dense)
    assert sub.dtype == np.float64 and sub.shape[1] == 2
    fitted = phase("fit", lambda: standardize().fit(T(sub)))
    Z = phase("standardize", lambda: np.asarray(fitted(T(sub)).collect()))
    # Model cal = a + b*temp (truth: meas=gain*(100+3*temp+e), calibrated
    # back with independent gain/offset — slope recovers ~3*scale ratio).
    y = Z[:, 0]
    X = np.column_stack([np.ones(Z.shape[0]), Z[:, 1]])
    w = phase("solve", lambda: np.asarray(
        solve(T(X.T @ X + 0.1 * np.eye(2)), T(X.T @ y)).collect()))
    print("temp coefficient (standardized):", w[1])
    out = pl.DataFrame({"term": ["intercept", "temp_coef"], "w": w.tolist()})
    out.write_parquet(os.path.join(OUT, "coefficients.parquet"))
    phase("explain", lambda: base.meas.mean(
        by=base.experiment_id).explain())
    return {"coef_temp": float(w[1]), "phases": dict(_phases),
            "rows": enriched.height}


if __name__ == "__main__":
    import json

    result = main()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in result["phases"].items()}, indent=1))
    print("coef_temp =", round(result["coef_temp"], 4),
          "rows =", result["rows"])
