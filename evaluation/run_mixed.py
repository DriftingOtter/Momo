"""Run the M0 mixed workload, v1 (explicit) and v2 (lazy bridge)."""
from __future__ import annotations

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness

VERSIONS = [v for v in os.environ.get("TL_EVAL_VERSIONS", "v2").split(",")
            if v in ("v1", "v2")] or ["v2"]
WL_TL = {"v1": "workload_mixed_v1", "v2": "workload_mixed"}

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
DATA = os.environ.get("TL_EVAL_DATA_DIR", "/tmp/tl-eval-data")
N_CUST = int(os.environ.get("TL_EVAL_N_CUST", "100000"))
N_TXN = int(os.environ.get("TL_EVAL_N_TXN", "1000000"))
LAM = 10.0

V1_BODY = """
        import time as _time
        import numpy as np
        from tensorlang import T as tl_T
        from workload_mixed_v1 import (FEATURE_COLS, tl_convert, tl_features,
                                       tl_ml_pipeline, tl_ridge,
                                       tl_ridge_weights,
                                       tl_split_train_holdout)
        cust_path, txn_path = %r, %r
        rel = time_phase("relational", lambda: tl_features(
            cust_path, txn_path).collect())
        import polars as pl
        assert isinstance(rel, pl.DataFrame), type(rel)
        t0 = _time.perf_counter()
        data_t, arr = tl_convert(rel)
        convert_s = _time.perf_counter() - t0
        train, hold = tl_split_train_holdout(arr)
        y_train = np.where(np.isnan(train[:, 1]),
                           np.nanmean(train[:, 1]), train[:, 1])
        y_hold = hold[:, 1].copy()
        pipe = tl_ml_pipeline()
        fitted = time_phase("fit", lambda: pipe.fit(tl_T(train)))
        comp_train = time_phase(
            "infer_train",
            lambda: np.asarray(fitted(tl_T(train)).collect()))
        comp_hold = time_phase(
            "infer_holdout",
            lambda: np.asarray(fitted(tl_T(hold)).collect()))
        assert comp_train.shape[1] == 3 and comp_hold.shape[1] == 3
        A, b, ymean = tl_ridge(comp_train, y_train, lam=%r)
        w = time_phase("ridge_solve",
                       lambda: tl_ridge_weights(A, b))
        pred = comp_hold @ w + ymean
        def main():
            dump = os.environ.get("TL_EVAL_DUMP", ".")
            np.save(os.path.join(dump, "tl_train.npy"), comp_train)
            np.save(os.path.join(dump, "tl_pred.npy"), np.asarray(pred))
            return {"phases": _PHASES,
                    "convert_s": convert_s, "copied": True,
                    "files": {"train": "tl_train.npy", "pred": "tl_pred.npy"},
                    "ridge_w": np.asarray(w).tolist(),
                    "hold_pred_mean": float(np.asarray(pred).mean()),
                    "executions": 4,
                    "checks": {"rows": rel.height,
                               "n_features": len(FEATURE_COLS)}}
"""

V2_BODY = """
        import time as _time
        import numpy as np
        from tensorlang import T as tl_T
        from workload_mixed import (FEATURE_COLS, tl_features,
                                    tl_ml_pipeline, tl_ridge,
                                    tl_ridge_weights,
                                    tl_split_train_holdout)
        cust_path, txn_path = %r, %r
        # Fresh graphs per repeat (shared Lazy objects would cache and
        # measure cache hits instead of execution).
        ref = time_phase("explicit_convert", lambda: tl_features(
            cust_path, txn_path).collect().select(FEATURE_COLS).to_numpy())
        mat = time_phase("bridge_convert", lambda: tl_features(
            cust_path, txn_path).to_dense(
                columns=FEATURE_COLS).collect())
        assert np.allclose(np.asarray(mat), np.asarray(ref),
                           equal_nan=True), "bridge != explicit reference"
        assert str(np.asarray(mat).dtype) == "float64"
        train_idx, hold_idx = tl_split_train_holdout(
            np.arange(np.asarray(mat).shape[0]))
        y_train = np.where(np.isnan(np.asarray(mat)[train_idx, 1]),
                           np.nanmean(np.asarray(mat)[train_idx, 1]),
                           np.asarray(mat)[train_idx, 1])
        # Recommended pattern: collect the bridge once, reuse materialized
        # arrays downstream (per-object Lazy caching; re-collecting a fresh
        # bridge graph re-executes the relational pipeline).
        mat_once = np.asarray(mat)
        pipe = tl_ml_pipeline()
        fitted = time_phase("fit", lambda: pipe.fit(tl_T(
            mat_once[train_idx])))
        comp_train = time_phase(
            "infer_train",
            lambda: np.asarray(fitted(tl_T(mat_once[train_idx])).collect()))
        comp_hold = time_phase(
            "infer_holdout",
            lambda: np.asarray(fitted(tl_T(mat_once[hold_idx])).collect()))
        assert comp_train.shape[1] == 3 and comp_hold.shape[1] == 3
        # Reuse cost probe (single sample): fitting directly on a fresh
        # lazy bridge re-executes relational+conversion per boundary.
        t_lazy = tl_features(cust_path, txn_path).to_dense(
            columns=FEATURE_COLS)[train_idx]
        _t0 = _time.perf_counter()
        pipe.fit(t_lazy)
        lazy_reuse_s = _time.perf_counter() - _t0
        A, b, ymean = tl_ridge(comp_train, y_train, lam=%r)
        w = time_phase("ridge_solve",
                       lambda: tl_ridge_weights(A, b))
        pred = comp_hold @ w + ymean
        def main():
            dump = os.environ.get("TL_EVAL_DUMP", ".")
            np.save(os.path.join(dump, "tl_train.npy"), comp_train)
            np.save(os.path.join(dump, "tl_pred.npy"), np.asarray(pred))
            return {"phases": _PHASES,
                    "convert_s": None, "copied": True,
                    "lazy_reuse_s": lazy_reuse_s,
                    "files": {"train": "tl_train.npy", "pred": "tl_pred.npy"},
                    "ridge_w": np.asarray(w).tolist(),
                    "hold_pred_mean": float(np.asarray(pred).mean()),
                    "executions": 2,
                    "checks": {"rows": None,
                               "n_features": len(FEATURE_COLS)}}
"""

NP_BODY = """
        import time as _time
        import numpy as np
        from sklearn.decomposition import PCA as _PCA
        from sklearn.impute import SimpleImputer as _Imp
        from sklearn.linear_model import Ridge as _Ridge
        from sklearn.pipeline import make_pipeline as _pipe
        from sklearn.preprocessing import StandardScaler as _Scal
        from workload_mixed_v1 import (FEATURE_COLS, np_features,
                                       tl_split_train_holdout)
        cust_path, txn_path = %r, %r
        rel = time_phase("relational", lambda: np_features(cust_path, txn_path))
        import polars as pl
        t0 = _time.perf_counter()
        arr = rel.select(FEATURE_COLS).to_numpy()
        convert_s = _time.perf_counter() - t0
        train, hold = tl_split_train_holdout(arr)
        y_train = np.where(np.isnan(train[:, 1]),
                           np.nanmean(train[:, 1]), train[:, 1])
        y_hold = hold[:, 1].copy()
        pipe = _pipe(_Imp(strategy="mean"), _Scal(), _PCA(n_components=3))
        time_phase("fit", lambda: pipe.fit(train))
        comp_train = time_phase("infer_train",
                                lambda: pipe.transform(train))
        comp_hold = time_phase("infer_holdout",
                               lambda: pipe.transform(hold))
        model = _Ridge(alpha=%r, solver="cholesky")
        time_phase("ridge_solve", lambda: model.fit(comp_train, y_train))
        pred = model.predict(comp_hold)
        def main():
            dump = os.environ.get("TL_EVAL_DUMP", ".")
            np.save(os.path.join(dump, "np_train.npy"), np.asarray(comp_train))
            np.save(os.path.join(dump, "np_pred.npy"), np.asarray(pred))
            return {"phases": _PHASES,
                    "convert_s": convert_s,
                    "files": {"train": "np_train.npy", "pred": "np_pred.npy"},
                    "ridge_w": np.asarray(model.coef_).tolist(),
                    "hold_pred_mean": float(np.asarray(pred).mean()),
                    "checks": {"rows": rel.height,
                               "n_features": len(FEATURE_COLS)}}
"""


def tl_loc(mod) -> int:
    src = inspect.getsource(mod.tl_features)
    src += inspect.getsource(mod.tl_ml_pipeline)
    src += inspect.getsource(mod.tl_ridge)
    src += inspect.getsource(mod.tl_ridge_weights)
    return harness.user_loc(src)


def np_loc() -> int:
    import workload_mixed_v1 as _v1

    return harness.user_loc(inspect.getsource(_v1.np_features))


def run_version(ver: str, cust_path: str, txn_path: str, tag: str) -> dict:
    body = V1_BODY if ver == "v1" else V2_BODY
    dump_tl = os.path.join(DATA, "arrays", tag, ver, "tl")
    dump_np = os.path.join(DATA, "arrays", tag, ver, "np")
    os.makedirs(dump_tl, exist_ok=True)
    os.makedirs(dump_np, exist_ok=True)
    old_dump = os.environ.get("TL_EVAL_DUMP")
    try:
        os.environ["TL_EVAL_DUMP"] = dump_tl
        tl = harness.run_isolated(harness.driver_template(
            body % (cust_path, txn_path, LAM)), timeout=2400)
        assert tl["returncode"] == 0, tl["stderr_tail"]
        os.environ["TL_EVAL_DUMP"] = dump_np
        np_res = harness.run_isolated(harness.driver_template(
            NP_BODY % (cust_path, txn_path, LAM)), timeout=2400)
        assert np_res["returncode"] == 0, np_res["stderr_tail"]
    finally:
        if old_dump is None:
            os.environ.pop("TL_EVAL_DUMP", None)
        else:
            os.environ["TL_EVAL_DUMP"] = old_dump
    if ver == "v2":
        assert tl["checks"]["n_features"] == np_res["checks"]["n_features"]
    else:
        assert tl["checks"] == np_res["checks"], (tl["checks"],
                                                  np_res["checks"])
    import itertools as _it
    import numpy as _np

    def _load(base, summary, key):
        return _np.load(os.path.join(base, summary["files"][key]))
    train_err = float(_np.abs(_np.abs(_load(dump_tl, tl, "train"))
                              - _np.abs(_load(dump_np, np_res, "train"))).max())
    assert train_err < 1e-9, train_err
    best = min(
        float(_np.abs(_np.asarray(tl["ridge_w"]) * _np.array(signs)
                      - _np.asarray(np_res["ridge_w"])).max())
        for signs in _it.product((-1.0, 1.0), repeat=3))
    # Holdout predictions are sign-invariant (flipped components and
    # weights cancel exactly), so they compare directly.
    perr = float(_np.abs(_load(dump_tl, tl, "pred")
                         - _load(dump_np, np_res, "pred")).max())
    print("%s: ridge sign-pattern diff %.3g; holdout pred diff %.3g"
          % (ver, best, perr))
    assert perr < 1e-6, perr
    import importlib as _importlib

    mod = _importlib.import_module(WL_TL[ver])
    summary = {"version": ver, "tensorlang": tl, "native": np_res,
               "ridge_sign_err": best, "ridge_pred_err": perr,
               "loc": {"tensorlang": tl_loc(mod), "native": np_loc()}}
    with open(os.path.join(RESULTS, f"mixed_{ver}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    rows = []
    for phase in ("relational", "explicit_convert", "bridge_convert", "fit",
                  "infer_train", "infer_holdout", "ridge_solve"):
        if phase not in tl["phases"]:
            continue
        a = tl["phases"][phase]
        b = np_res["phases"].get(
            {"explicit_convert": "relational"}.get(phase, phase), {})
        rows.append([phase, f"{a['median_s']:.6f}",
                     f"{b.get('median_s', float('nan')):.6f}" if b else "-",
                     (f"{a['median_s'] / b['median_s']:.2f}x"
                      if b and b["median_s"] > 0 else "-")])
    print(harness.fmt_table(["phase", "tensorlang_s", "native_s", "ratio"], rows))
    if ver == "v2":
        conv = tl["phases"].get("bridge_convert", {}).get("median_s")
    else:
        conv = tl.get("convert_s")
    print("%s convert: tensorlang=%.4fs native=%.4fs executions=%s" % (
        ver, conv or float("nan"), np_res["convert_s"], tl.get("executions")))
    print("%s peak RSS MB: tensorlang=%.1f native=%.1f" % (
        ver, tl.get("peak_rss_mb", float("nan")),
        np_res.get("peak_rss_mb", float("nan"))))
    print("%s user LOC: tensorlang=%d native=%d" % (
        ver, tl_loc(mod), np_loc()))
    return summary


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    cust_path = os.path.join(DATA, "w0_cust.parquet")
    txn_path = os.path.join(DATA, "w0_txn.parquet")
    if not (os.path.exists(cust_path) and os.path.exists(txn_path)):
        print("generating data...")
        import workload_mixed_v1 as _v1

        _v1.make_data(cust_path, txn_path, N_CUST, N_TXN)
    for ver in VERSIONS:
        print("=== M0 mixed workload %s (cust=%d txn=%d) ==="
              % (ver, N_CUST, N_TXN))
        tag = f"{N_CUST // 1000}k-{N_TXN // 1000}k"
        run_version(ver, cust_path, txn_path, tag)


if __name__ == "__main__":
    main()
