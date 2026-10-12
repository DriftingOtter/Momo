"""Benchmark runner: executes the suite and records results + machine specs."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from harness import machine_specs  # noqa: E402

import bench_dense  # noqa: E402
import bench_memory  # noqa: E402
import bench_pipeline  # noqa: E402
import bench_structured  # noqa: E402
import bench_autodiff  # noqa: E402
from datasets import DATA_DIR, make_sales  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["small", "medium"],
                        choices=["small", "medium", "large"])
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-structured", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-autodiff", action="store_true")
    args = parser.parse_args()

    specs = machine_specs()
    print(f"machine: {specs['platform']} | cpus={specs['cpus']} | "
          f"numpy={specs['numpy']} polars={specs['polars']}")
    results = {"specs": specs, "measurements": []}
    for size in args.sizes:
        rows = bench_structured.SIZES[size]
        data = DATA_DIR / f"sales-{size}.parquet"
        if not data.exists():
            print(f"generating {data} ({rows} rows)...")
            make_sales(data, rows)
        outdir = DATA_DIR / "out"
        outdir.mkdir(exist_ok=True)
        if not args.skip_dense:
            print(f"[{size}] dense...")
            results["measurements"] += [asdict(m) for m in bench_dense.run(size)]
        if not args.skip_structured:
            print(f"[{size}] structured...")
            results["measurements"] += [
                asdict(m) for m in bench_structured.run(size, str(data))]
        if not args.skip_pipeline:
            print(f"[{size}] pipeline...")
            results["measurements"] += [
                asdict(m) for m in bench_pipeline.run(size, str(data), str(outdir))]
        if not args.skip_memory:
            print(f"[{size}] memory...")
            results["measurements"] += [asdict(m) for m in bench_memory.run(size)]
        if not args.skip_autodiff:
            print(f"[{size}] autodiff...")
            results["measurements"] += [asdict(m) for m in bench_autodiff.run(size)]

    results_dir = BENCH_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(results_dir / "RESULTS.md", "w") as fh:
        fh.write(render_markdown(results))
    print("wrote benchmarks/results/results.json + RESULTS.md")
    return 0


def render_markdown(results: dict) -> str:
    specs = results["specs"]
    lines = [
        "# TensorLang v0.2 benchmark results",
        "",
        "## Machine",
        "",
        *[f"- {k}: {v}" for k, v in specs.items()],
        "",
        "## Method",
        "",
        "- Wall-clock: best of 3 after 1 warmup (`time.perf_counter`).",
        "- `peak_python_alloc_mb`: tracemalloc peak around the measured call.",
        "- `peak_rss_mb`: process high-water mark (absolute; never decreases).",
        "- Correctness: every TensorLang result asserted vs native output.",
        "- `backend`/`materializations`: from `profile()`/`explain()`.",
        "",
        "## Results",
        "",
        "| workload | size | impl | seconds | py_alloc_MB | rss_MB | backend | scans |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in results["measurements"]:
        lines.append(
            f"| {m['workload']} | {m['size']} | {m['impl']} | {m['seconds']:.4f} | "
            f"{m['peak_python_alloc_mb']:.2f} | {m['peak_rss_mb']:.1f} | "
            f"{m['backend']} | {m['materializations']} |"
        )
        if m.get("note"):
            lines.append(f"|  |  |  | {m['note']} |  |  |  |  |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
