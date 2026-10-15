"""Run workload 2 (data science + ML) under the evaluation harness."""
from __future__ import annotations

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness
import workload_data as workloads

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
DATA = os.environ.get("TL_EVAL_DATA_DIR", "/tmp/tl-eval-data")
N_CUST = int(os.environ.get("TL_EVAL_N_CUST", "20000"))
N_TXN = int(os.environ.get("TL_EVAL_N_TXN", "200000"))

TL_BODY = """
        import numpy as np
        from tensorlang import T as tl_T
        from workload_data import (NP_FEATURE_COLS, TL_FEATURE_COLS,
                                   tl_features, tl_ml_pipeline,
                                   tl_split_train_holdout)
        cust_path, txn_path = %r, %r
        rel = time_phase("relational", lambda: tl_features(
            cust_path, txn_path).collect())
        import polars as pl
        assert isinstance(rel, pl.DataFrame), type(rel)
        feat = rel.select(TL_FEATURE_COLS).to_numpy()
        train, hold = tl_split_train_holdout(
            rel.select(TL_FEATURE_COLS), train_frac=0.8)
        train_np = train.to_numpy()
        hold_np = hold.to_numpy()
        pipe = tl_ml_pipeline()
        fitted = time_phase("fit", lambda: pipe.fit(tl_T(train_np)))
        tout = time_phase("apply_train",
                          lambda: np.asarray(fitted(tl_T(train_np)).collect()))
        hout = time_phase("apply_holdout",
                          lambda: np.asarray(fitted(tl_T(hold_np)).collect()))
        assert tout.shape[1] == 2 and hout.shape[1] == 2, (tout.shape, hout.shape)
        def main():
            dump = os.environ.get("TL_EVAL_DUMP", ".")
            np.save(os.path.join(dump, "tl_train.npy"), np.asarray(tout))
            np.save(os.path.join(dump, "tl_hold.npy"), np.asarray(hout))
            return {"phases": _PHASES,
                    "files": {"train": "tl_train.npy", "hold": "tl_hold.npy"},
                    "checks": {"rows": rel.height,
                               "nulls": int(sum(rel.null_count().row(0)))}}
"""

NP_BODY = """
        import numpy as np
        from sklearn.decomposition import PCA as _PCA
        from sklearn.impute import SimpleImputer as _Imp
        from sklearn.pipeline import make_pipeline as _pipe
        from sklearn.preprocessing import StandardScaler as _Scal
        from workload_data import (NP_FEATURE_COLS, np_features,
                                   tl_split_train_holdout)
        cust_path, txn_path = %r, %r
        rel = time_phase("relational", lambda: np_features(cust_path, txn_path))
        import polars as pl
        train, hold = tl_split_train_holdout(
            rel.select(NP_FEATURE_COLS), train_frac=0.8)
        train_np, hold_np = train.to_numpy(), hold.to_numpy()
        pipe = _pipe(_Imp(strategy="mean"), _Scal(), _PCA(n_components=2))
        time_phase("fit", lambda: pipe.fit(train_np))
        tout = time_phase("apply_train", lambda: pipe.transform(train_np))
        hout = time_phase("apply_holdout", lambda: pipe.transform(hold_np))
        assert tout.shape[1] == 2 and hout.shape[1] == 2
        def main():
            dump = os.environ.get("TL_EVAL_DUMP", ".")
            np.save(os.path.join(dump, "np_train.npy"), np.asarray(tout))
            np.save(os.path.join(dump, "np_hold.npy"), np.asarray(hout))
            return {"phases": _PHASES,
                    "files": {"train": "np_train.npy", "hold": "np_hold.npy"},
                    "checks": {"rows": rel.height,
                               "nulls": int(sum(rel.null_count().row(0)))}}
"""


def tl_loc() -> int:
    # tl_split_train_holdout is shared scaffolding (used by both drivers);
    # counted for neither side.
    src = inspect.getsource(workloads.tl_features)
    src += inspect.getsource(workloads.tl_ml_pipeline)
    return harness.user_loc(src)


def np_loc() -> int:
    return harness.user_loc(inspect.getsource(workloads.np_features))


def _dump_dir(*parts: str) -> str:
    path = os.path.join(DATA, "arrays", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    cust_path = os.path.join(DATA, "w2_cust.parquet")
    txn_path = os.path.join(DATA, "w2_txn.parquet")
    if not (os.path.exists(cust_path) and os.path.exists(txn_path)):
        print("generating data...")
        workloads.make_data(cust_path, txn_path, N_CUST, N_TXN)
    print("=== workload 2: data science (cust=%d txn=%d) ===" % (N_CUST, N_TXN))
    import numpy as _np

    old_dump = os.environ.get("TL_EVAL_DUMP")
    try:
        os.environ["TL_EVAL_DUMP"] = _dump_dir("data-science", "tl")
        tl = harness.run_isolated(harness.driver_template(
            TL_BODY % (cust_path, txn_path)))
        assert tl["returncode"] == 0, tl["stderr_tail"]
        os.environ["TL_EVAL_DUMP"] = _dump_dir("data-science", "np")
        np_res = harness.run_isolated(harness.driver_template(
            NP_BODY % (cust_path, txn_path)))
        assert np_res["returncode"] == 0, np_res["stderr_tail"]
    finally:
        if old_dump is None:
            os.environ.pop("TL_EVAL_DUMP", None)
        else:
            os.environ["TL_EVAL_DUMP"] = old_dump
    for key in ("rows", "nulls"):
        assert tl["checks"][key] == np_res["checks"][key], (key, tl["checks"])
    # PCA sign indeterminacy: compare absolute projections end to end.
    errs = {}
    for split, tkey, nkey in (("train", "tl_train.npy", "np_train.npy"),
                              ("hold", "tl_hold.npy", "np_hold.npy")):
        a = _np.load(os.path.join(DATA, "arrays", "data-science", "tl", tkey))
        b = _np.load(os.path.join(DATA, "arrays", "data-science", "np", nkey))
        assert a.shape == b.shape, (split, a.shape, b.shape)
        errs[split] = float(_np.abs(_np.abs(a) - _np.abs(b)).max())
        assert errs[split] < 1e-9, (split, errs[split])
    summary = {"tensorlang": tl, "native": np_res,
               "projection_err": errs,
               "loc": {"tensorlang": tl_loc(), "native": np_loc() + 3}}
    with open(os.path.join(RESULTS, "data_science.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    rows = []
    for phase in ("relational", "fit", "apply_train", "apply_holdout"):
        a, b = tl["phases"][phase], np_res["phases"][phase]
        rows.append([phase, f"{a['median_s']:.6f}", f"{b['median_s']:.6f}",
                     f"{a['median_s'] / b['median_s']:.2f}x"
                     if b["median_s"] > 0 else "-"])
    print(harness.fmt_table(["phase", "tensorlang_s", "native_s", "ratio"], rows))
    print("peak RSS MB: tensorlang=%.1f native=%.1f" % (
        tl.get("peak_rss_mb", float("nan")),
        np_res.get("peak_rss_mb", float("nan"))))
    print("user LOC: tensorlang=%d native=%d (split helper shared)" % (
        tl_loc(), np_loc()))
    print("wrote evaluation/results/data_science.json")


if __name__ == "__main__":
    main()
