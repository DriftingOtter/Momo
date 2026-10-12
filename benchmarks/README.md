# TensorLang benchmarks (v0.2)

Reproducible performance suite comparing TensorLang against equivalent
native NumPy / Polars code.

## Workloads

| File | Coverage |
|---|---|
| `bench_dense.py` | elementwise chains, operator application, matmul+reduction (1M–10M values) |
| `bench_structured.py` | 2-predicate filter, filter+derive (lazy scan), grouped sum (100K–5M rows) |
| `bench_pipeline.py` | scan → filter → derive → group → write Parquet, output equality |
| `bench_memory.py` | snapshot cost (copy-on-write views), branch collection, cache release |
| `bench_autodiff.py` | gradient graphs of increasing depth/size vs autograd or finite differences, forward/backward split |

## Method

- Wall-clock: best of 3 after 1 warmup (`time.perf_counter`).
- `peak_python_alloc_mb`: tracemalloc peak around the measured call
  (misses off-heap NumPy/Polars buffers — reported alongside RSS, not
  instead of it).
- `peak_rss_mb`: process high-water mark via `resource.getrusage`
  (absolute; maxrss never decreases, so compare across rows, not to zero).
- Correctness: asserted on every run (`allclose` / frame `equals`).
- Backend/materializations: from `profile()` and `explain()`, never guessed.

## Run

```bash
.venv/bin/python benchmarks/datasets.py --rows 1000000   # (optional precache)
.venv/bin/python benchmarks/run_all.py --sizes small medium
```

Note: run on a quiet machine — the reference numbers were taken on a
shared laptop where native baselines themselves swing ~2x between runs.
Best-of-3 minimizes, but does not eliminate, that noise.

Sizes: `small` (100K rows / 100K values / 500² matmul),
`medium` (1M / 1M / 1000²), `large` (5M–50M — needs a big machine).
Results land in `results/results.json` and `results/RESULTS.md`.

## Reading the numbers

- The v0.2 objective is **near-native backend performance without
  unnecessary materialization**: structured pipelines should lower to one
  Polars query (`backend=polars-lazy`, `scans=1`).
- Dense tensor overhead is the per-operator graph-building cost; it should
  stay a small constant factor over raw NumPy, shrinking with data size.
- `snapshot-10-branches` proves copy-on-write snapshots: building ten
  graphs over an 80MB array should add ~0MB RSS.
- Lazy/streaming plans preserve no incidental row order; benchmarks compare
  order-insensitively where no explicit `sort` is applied.
