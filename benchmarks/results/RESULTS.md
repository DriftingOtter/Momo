# TensorLang v0.2 benchmark results

## Machine

- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- python: 3.14.7
- cpus: 8
- tensorlang: 0.4.0
- numpy: 2.5.4
- polars: 2.0.0
- sympy: 1.14.0

## Method

- Wall-clock: best of 3 after 1 warmup (`time.perf_counter`).
- `peak_python_alloc_mb`: tracemalloc peak around the measured call.
- `peak_rss_mb`: process high-water mark (absolute; never decreases).
- Correctness: every TensorLang result asserted vs native output.
- `backend`/`materializations`: from `profile()`/`explain()`.

## Results

| workload | size | impl | seconds | py_alloc_MB | rss_MB | backend | scans |
|---|---|---|---|---|---|---|---|
| dense-chain | small | tensorlang | 0.0001 | 1.60 | 118.4 | materialized | 1 |
| dense-chain | small | native | 0.0001 | 0.80 | 118.4 | numpy | 0 |
| dense-operator | small | tensorlang | 0.0001 | 1.60 | 119.5 | materialized | 1 |
| dense-operator | small | native | 0.0000 | 0.80 | 119.5 | numpy | 0 |
| matmul-500x500 | small | tensorlang | 0.0008 | 2.00 | 129.7 | materialized | 1 |
| matmul-500x500 | small | native | 0.0007 | 2.00 | 129.7 | numpy | 0 |
| filter-2-predicates | small | tensorlang | 0.0004 | 0.01 | 165.6 | materialized | 1 |
| filter-2-predicates | small | native | 0.0003 | 0.00 | 168.6 | polars | 0 |
| filter-derive | small | tensorlang | 0.0025 | 0.01 | 212.2 | polars-lazy | 1 |
| filter-derive | small | native | 0.0021 | 0.00 | 218.3 | polars-lazy | 0 |
| grouped-sum | small | tensorlang | 0.0030 | 0.02 | 223.2 | polars-lazy | 1 |
| grouped-sum | small | native | 0.0025 | 0.00 | 223.9 | polars-lazy | 0 |
| pipeline-write | small | tensorlang | 0.0031 | 0.01 | 226.0 | polars-lazy | 1 |
| pipeline-write | small | native | 0.0028 | 0.00 | 226.3 | polars-lazy | 0 |
| snapshot-10-branches | small | tensorlang | 0.0000 | 0.00 | 242.4 | cow-views | 0 |
|  |  |  | tensor=8.0MB graphs=0.0MB for 8MB data |  |  |  |  |
| branch-collect | small | tensorlang | 0.0000 | 0.00 | 258.5 | materialized | 0 |
|  |  |  | collect added 16.1MB |  |  |  |  |
| cache-release | small | tensorlang | 0.0000 | 0.00 | 258.5 | n/a | 0 |
|  |  |  | release() clears retained collect() results |  |  |  |  |
| external-copy-once | small | tensorlang | 0.0000 | 0.00 | 266.5 | n/a | 0 |
|  |  |  | caller-side mutation cannot leak into graphs |  |  |  |  |
| grad-depth-1 | small | tensorlang | 0.0003 | 0.07 | 268.6 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.12ms backward=0.23ms |  |  |  |  |
| value-and-grad-depth-1 | small | tensorlang | 0.0003 | 0.07 | 268.7 | autodiff-shared | 1 |
|  |  |  | forward=0.11ms backward=0.22ms |  |  |  |  |
| grad-depth-8 | small | tensorlang | 0.0007 | 0.25 | 269.0 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.31ms backward=0.62ms |  |  |  |  |
| value-and-grad-depth-8 | small | tensorlang | 0.0007 | 0.25 | 269.0 | autodiff-shared | 1 |
|  |  |  | forward=0.30ms backward=0.60ms |  |  |  |  |
| dense-chain | medium | tensorlang | 0.0010 | 16.00 | 325.2 | materialized | 1 |
| dense-chain | medium | native | 0.0009 | 8.00 | 325.2 | numpy | 0 |
| dense-operator | medium | tensorlang | 0.0005 | 16.00 | 341.3 | materialized | 1 |
| dense-operator | medium | native | 0.0003 | 8.00 | 341.3 | numpy | 0 |
| matmul-1000x1000 | medium | tensorlang | 0.0060 | 8.00 | 365.7 | materialized | 1 |
| matmul-1000x1000 | medium | native | 0.0059 | 8.00 | 365.7 | numpy | 0 |
| filter-2-predicates | medium | tensorlang | 0.0020 | 0.01 | 469.1 | materialized | 1 |
| filter-2-predicates | medium | native | 0.0019 | 0.00 | 486.0 | polars | 0 |
| filter-derive | medium | tensorlang | 0.0076 | 0.01 | 568.1 | polars-lazy | 1 |
| filter-derive | medium | native | 0.0071 | 0.00 | 576.5 | polars-lazy | 0 |
| grouped-sum | medium | tensorlang | 0.0071 | 0.01 | 587.1 | polars-lazy | 1 |
| grouped-sum | medium | native | 0.0067 | 0.00 | 587.8 | polars-lazy | 0 |
| pipeline-write | medium | tensorlang | 0.0076 | 0.01 | 587.9 | polars-lazy | 1 |
| pipeline-write | medium | native | 0.0070 | 0.00 | 587.9 | polars-lazy | 0 |
| snapshot-10-branches | medium | tensorlang | 0.0000 | 0.00 | 743.5 | cow-views | 0 |
|  |  |  | tensor=80.0MB graphs=0.0MB for 80MB data |  |  |  |  |
| branch-collect | medium | tensorlang | 0.0000 | 0.00 | 903.5 | materialized | 0 |
|  |  |  | collect added 160.0MB |  |  |  |  |
| cache-release | medium | tensorlang | 0.0000 | 0.00 | 903.5 | n/a | 0 |
|  |  |  | release() clears retained collect() results |  |  |  |  |
| external-copy-once | medium | tensorlang | 0.0000 | 0.00 | 983.5 | n/a | 0 |
|  |  |  | caller-side mutation cannot leak into graphs |  |  |  |  |
| grad-depth-1 | medium | tensorlang | 0.0012 | 6.41 | 983.5 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.21ms backward=1.18ms |  |  |  |  |
| value-and-grad-depth-1 | medium | tensorlang | 0.0013 | 6.41 | 983.5 | autodiff-shared | 1 |
|  |  |  | forward=0.19ms backward=1.19ms |  |  |  |  |
| grad-depth-8 | medium | tensorlang | 0.0026 | 23.21 | 990.4 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.82ms backward=2.63ms |  |  |  |  |
| value-and-grad-depth-8 | medium | tensorlang | 0.0025 | 23.21 | 990.4 | autodiff-shared | 1 |
|  |  |  | forward=0.69ms backward=2.42ms |  |  |  |  |
