"""Run workload 1 (scientific) under the evaluation harness.

Two isolated subprocesses (TensorLang / native) with identical data
(same seed); equivalence asserted inside the TensorLang driver; timings
are medians of repeats with warmup excluded; RSS is the child process
peak. Results JSON goes to evaluation/results/.
"""
from __future__ import annotations

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness

VERSIONS = [v for v in os.environ.get("TL_EVAL_VERSIONS", "v2").split(",")
            if v in ("v1", "v2")] or ["v2"]
WL_MODULE = {"v1": "workload_scientific_v1", "v2": "workload_scientific"}

N = int(os.environ.get("TL_EVAL_N", "2000"))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

TL_BODY = """
        import numpy as np
        from tensorlang import eigh as tl_eigh, T as tl_T
        from workload_scientific_v1 import (make_positions, np_com,
                                               np_energy, np_gradient,
                                               np_inertia)
        from %s import (tl_build, tl_com, tl_energy, tl_force_law,
                        tl_gradient, tl_inertia)
        N = %d
        pos = make_positions(N)
        ref = {"energy": np_energy(pos), "gradient": np_gradient(pos),
               "inertia": np_inertia(pos), "com": np_com(pos)}
        ops = time_phase("build", lambda: tl_build(N))
        e = time_phase("energy", lambda: tl_energy(ops, pos))
        assert abs(e - ref["energy"]) < 1e-6 * abs(ref["energy"]) + 1e-9, (e, ref["energy"])
        g = time_phase("gradient", lambda: tl_gradient(ops, pos))
        assert np.allclose(g, ref["gradient"], rtol=1e-6, atol=1e-9), np.abs(g - ref["gradient"]).max()
        def _inertia_moments():
            import numpy as _np
            return _np.asarray(tl_eigh(tl_T(tl_inertia(ops, pos))).values.collect())
        m = time_phase("inertia+eigh", _inertia_moments)
        assert np.allclose(m, np.linalg.eigvalsh(ref["inertia"]), rtol=1e-9), m
        c = time_phase("center_of_mass", lambda: tl_com(ops, pos))
        assert np.allclose(c, ref["com"], rtol=1e-9), c
        f = time_phase("force_law_symbolic",
                       lambda: tl_force_law(ops, np.array([0.9, 1.0, 2.0])))
        def _scalar(x):
            return 4.0 * ((1.0 / x) ** 12 - (1.0 / x) ** 6)
        pts = np.array([0.9, 1.0, 2.0])
        fd = (_scalar(pts + 1e-7) - _scalar(pts - 1e-7)) / 2e-7
        assert np.allclose(f, fd, rtol=1e-5), (f, fd)
        def main():
            return {"phases": _PHASES,
                    "checks": {"energy": e,
                               "grad_max_err": float(np.abs(g - ref["gradient"]).max())}}
"""

NP_BODY = """
        import numpy as np
        from workload_scientific_v1 import (make_positions, np_com,
                                               np_energy, np_gradient,
                                               np_inertia)
        N = %d
        pos = make_positions(N)
        e = time_phase("energy", lambda: np_energy(pos))
        assert np.isfinite(e)
        g = time_phase("gradient", lambda: np_gradient(pos))
        assert np.isfinite(g).all()
        m = time_phase("inertia+eigh",
                       lambda: np.linalg.eigvalsh(np_inertia(pos)))
        c = time_phase("center_of_mass", lambda: np_com(pos))
        def main():
            return {"phases": _PHASES, "checks": {"energy": float(e)}}
"""


def tl_loc(workloads) -> int:
    src = inspect.getsource(workloads.tl_build)
    src += inspect.getsource(workloads.tl_energy)
    src += inspect.getsource(workloads.tl_gradient)
    src += inspect.getsource(workloads.tl_inertia)
    src += inspect.getsource(workloads.tl_com)
    src += inspect.getsource(workloads.tl_force_law)
    return harness.user_loc(src)


def np_loc(workloads) -> int:
    import importlib as _importlib

    native = _importlib.import_module("workload_scientific_v1")
    src = inspect.getsource(native.np_energy)
    src += inspect.getsource(native.np_gradient)
    src += inspect.getsource(native.np_inertia)
    src += inspect.getsource(native.np_com)
    return harness.user_loc(src)


def run_conditioned() -> None:
    """Well-conditioned supplement (controlled minimum separation):
    distinguishes AD correctness from large-magnitude effects with a
    tight tolerance both codes must meet."""
    import numpy as _np
    import workload_scientific as _w
    from workload_scientific_v1 import make_positions_conditioned as _cond
    from workload_scientific_v1 import np_gradient as _np_gradient
    n = 1000
    pos = _cond(n)
    ops = _w.tl_build(n)
    g_tl = _w.tl_gradient(ops, pos)
    g_np = _np_gradient(pos)
    abs_err = float(_np.abs(g_tl - g_np).max())
    rel = _np.abs(g_tl - g_np) / (_np.abs(g_np) + 1e-300)
    print(f"conditioned N={n}: max|g|={float(_np.abs(g_np).max()):.4g} "
          f"abs_err={abs_err:.4g} max_rel_err={float(rel.max()):.4g}")
    assert _np.allclose(g_tl, g_np, rtol=1e-9, atol=1e-9), abs_err


def run_version(ver: str) -> dict:
    import importlib as _importlib

    workloads = _importlib.import_module(WL_MODULE[ver])
    mod = WL_MODULE[ver]
    print("=== workload 1 (%s): scientific (N=%d) ===" % (ver, N))
    tl = harness.run_isolated(harness.driver_template(TL_BODY % (mod, N)))
    assert tl["returncode"] == 0, tl["stderr_tail"]
    np_res = harness.run_isolated(harness.driver_template(NP_BODY % N))
    assert np_res["returncode"] == 0, np_res["stderr_tail"]
    summary = {"version": ver, "n": N, "tensorlang": tl, "native": np_res,
               "loc": {"tensorlang": tl_loc(workloads),
                       "native": np_loc(workloads)}}
    with open(os.path.join(RESULTS, f"scientific_{ver}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    rows = []
    phases = sorted(set(tl["phases"]) | set(np_res["phases"]))
    for phase in phases:
        a = tl["phases"].get(phase, {})
        b = np_res["phases"].get(phase, {})
        rows.append([phase,
                     f"{a.get('median_s', float('nan')):.6f}" if a else "-",
                     f"{b.get('median_s', float('nan')):.6f}" if b else "-",
                     (f"{a['median_s'] / b['median_s']:.2f}x"
                      if a and b and b["median_s"] > 0 else "-")])
    print(harness.fmt_table(["phase", "tensorlang_s", "native_s", "ratio"], rows))
    print("peak RSS MB: tensorlang=%.1f native=%.1f" % (
        tl.get("peak_rss_mb", float("nan")),
        np_res.get("peak_rss_mb", float("nan"))))
    print("user LOC: tensorlang=%d native=%d" % (tl_loc(workloads),
                                                 np_loc(workloads)))
    print("wrote evaluation/results/scientific_%s.json" % ver)
    return summary


def main() -> None:
    summaries = {ver: run_version(ver) for ver in VERSIONS}
    if len(summaries) > 1:
        versions = sorted(summaries)
        a, b = (summaries[versions[0]]["tensorlang"]["phases"],
                summaries[versions[1]]["tensorlang"]["phases"])
        print("--- v1 vs v2 TensorLang medians (same engine) ---")
        for phase in sorted(set(a) | set(b)):
            x, y = a.get(phase, {}), b.get(phase, {})
            if x and y and y["median_s"] > 0:
                print(f"{phase}: v1={x['median_s']:.6f}s "
                      f"v2={y['median_s']:.6f}s "
                      f"(v2/v1={y['median_s'] / x['median_s']:.2f}x)")
    run_conditioned()


if __name__ == "__main__":
    main()
