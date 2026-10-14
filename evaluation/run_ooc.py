"""Run workload 3 (out-of-core) under the evaluation harness."""
from __future__ import annotations

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness
import workload_ooc as workloads

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
DATA = os.environ.get("TL_EVAL_DATA_DIR", "/tmp/tl-eval-data")

# (rows, cols, chunk_rows, budget_MB, workers)
CONFIGS = [
    (250_000, 32, 10_000, 64, 4),   # 64MB data, 2.56MB chunks
    (1_000_000, 32, 10_000, 16, 4),  # 256MB data, 2.56MB chunks
    (1_000_000, 32, 2_000, 16, 2),   # 256MB data, 0.5MB chunks
]

TL_BODY = """
        import shutil
        from workload_ooc import tl_normalize
        src, out, budget_mb, workers = %r, %r, %d, %d
        shutil.rmtree(out, ignore_errors=True)
        rec = time_phase("normalize", lambda: tl_normalize(
            src, out, budget_mb * 1024 * 1024, workers))
        assert abs(rec["mean_after"]) < 1e-9, rec["mean_after"]
        assert abs(rec["std_after"] - 1.0) < 1e-9, rec["std_after"]
        from tensorlang.runtime import chunk_plan as _cp
        decision = dict(_cp.LAST_DECISION)
        def main():
            return {"phases": _PHASES,
                    "checks": {"mu": rec["mu"], "sigma": rec["sigma"],
                               "outcome": decision.get("outcome"),
                               "workers_used": decision.get("workers_used")}}
"""

NP_BODY = """
        import shutil
        from workload_ooc import np_normalize
        src, out, workers = %r, %r, %d
        shutil.rmtree(out, ignore_errors=True)
        rec = time_phase("normalize", lambda: np_normalize(src, out, workers))
        assert abs(rec["mean_after"]) < 1e-9, rec["mean_after"]
        assert abs(rec["std_after"] - 1.0) < 1e-9, rec["std_after"]
        def main():
            return {"phases": _PHASES,
                    "checks": {"mu": rec["mu"], "sigma": rec["sigma"]}}
"""


def tl_loc() -> int:
    return harness.user_loc(inspect.getsource(workloads.tl_normalize))


def np_loc() -> int:
    return harness.user_loc(inspect.getsource(workloads.np_normalize))


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    all_rows, configs_out = [], []
    for rows, cols, chunk_rows, budget_mb, workers in CONFIGS:
        tag = f"{rows // 1000}k-{cols}c-ch{chunk_rows}-b{budget_mb}MB-w{workers}"
        src = os.path.join(DATA, f"w3-{tag}.zarr")
        if not os.path.exists(src):
            print(f"generating {tag}...")
            workloads.make_data(src, rows, cols, chunk_rows)
        out_tl = os.path.join(DATA, f"w3-{tag}-tl.zarr")
        out_np = os.path.join(DATA, f"w3-{tag}-np.zarr")
        print(f"=== workload 3: {tag} ===")
        tl = harness.run_isolated(harness.driver_template(
            TL_BODY % (src, out_tl, budget_mb, workers)), timeout=2400)
        assert tl["returncode"] == 0, tl["stderr_tail"]
        np_res = harness.run_isolated(harness.driver_template(
            NP_BODY % (src, out_np, workers)), timeout=2400)
        assert np_res["returncode"] == 0, np_res["stderr_tail"]
        assert abs(tl["checks"]["mu"] - np_res["checks"]["mu"]) < 1e-9
        a, b = tl["phases"]["normalize"], np_res["phases"]["normalize"]
        all_rows.append([tag, f"{a['median_s']:.3f}", f"{b['median_s']:.3f}",
                         f"{a['median_s'] / b['median_s']:.2f}x",
                         str(tl.get("peak_rss_mb", float("nan"))),
                         str(np_res.get("peak_rss_mb", float("nan"))),
                         tl["checks"]["outcome"],
                         str(tl["checks"]["workers_used"])])
        configs_out.append({"tag": tag, "tensorlang": tl, "native": np_res})
    print(harness.fmt_table(
        ["config", "tensorlang_s", "native_s", "ratio", "tl_rss",
         "np_rss", "admission", "workers"], all_rows))
    print("user LOC: tensorlang=%d native=%d" % (tl_loc(), np_loc()))
    with open(os.path.join(RESULTS, "out_of_core.json"), "w") as fh:
        json.dump({"configs": configs_out,
                   "loc": {"tensorlang": tl_loc(), "native": np_loc()}},
                  fh, indent=2)
    print("wrote evaluation/results/out_of_core.json")


if __name__ == "__main__":
    main()
