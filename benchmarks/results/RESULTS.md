# TensorLang v0.2 benchmark results

## Machine

- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- python: 3.14.7
- cpus: 8
- tensorlang: 0.4.2
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
| dense-chain | small | tensorlang | 0.0002 | 1.60 | 119.6 | materialized | 1 |
| dense-chain | small | native | 0.0001 | 0.80 | 119.6 | numpy | 0 |
| dense-operator | small | tensorlang | 0.0001 | 1.60 | 120.7 | materialized | 1 |
| dense-operator | small | native | 0.0001 | 0.80 | 120.7 | numpy | 0 |
| matmul-500x500 | small | tensorlang | 0.0013 | 2.00 | 130.9 | materialized | 1 |
| matmul-500x500 | small | native | 0.0012 | 2.00 | 130.9 | numpy | 0 |
| filter-2-predicates | small | tensorlang | 0.0007 | 0.01 | 167.0 | materialized | 1 |
| filter-2-predicates | small | native | 0.0005 | 0.00 | 169.9 | polars | 0 |
| filter-derive | small | tensorlang | 0.0043 | 0.01 | 211.9 | polars-lazy | 1 |
| filter-derive | small | native | 0.0034 | 0.00 | 220.8 | polars-lazy | 0 |
| grouped-sum | small | tensorlang | 0.0047 | 0.02 | 226.3 | polars-lazy | 1 |
| grouped-sum | small | native | 0.0042 | 0.00 | 226.7 | polars-lazy | 0 |
| pipeline-write | small | tensorlang | 0.0049 | 0.02 | 229.9 | polars-lazy | 1 |
| pipeline-write | small | native | 0.0044 | 0.00 | 230.2 | polars-lazy | 0 |
| snapshot-10-branches | small | tensorlang | 0.0000 | 0.00 | 246.2 | cow-views | 0 |
|  |  |  | tensor=8.0MB graphs=0.0MB for 8MB data |  |  |  |  |
| branch-collect | small | tensorlang | 0.0000 | 0.00 | 262.3 | materialized | 0 |
|  |  |  | collect added 16.1MB |  |  |  |  |
| cache-release | small | tensorlang | 0.0000 | 0.00 | 262.3 | n/a | 0 |
|  |  |  | release() clears retained collect() results |  |  |  |  |
| external-copy-once | small | tensorlang | 0.0000 | 0.00 | 270.3 | n/a | 0 |
|  |  |  | caller-side mutation cannot leak into graphs |  |  |  |  |
| grad-depth-1 | small | tensorlang | 0.0005 | 0.07 | 272.5 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.20ms backward=0.41ms |  |  |  |  |
| value-and-grad-depth-1 | small | tensorlang | 0.0005 | 0.07 | 272.6 | autodiff-shared | 1 |
|  |  |  | forward=0.19ms backward=0.39ms |  |  |  |  |
| grad-depth-8 | small | tensorlang | 0.0012 | 0.25 | 272.9 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.59ms backward=1.13ms |  |  |  |  |
| value-and-grad-depth-8 | small | tensorlang | 0.0012 | 0.25 | 272.9 | autodiff-shared | 1 |
|  |  |  | forward=0.55ms backward=1.09ms |  |  |  |  |
| dense-chain | medium | tensorlang | 0.0020 | 16.00 | 329.1 | materialized | 1 |
| dense-chain | medium | native | 0.0012 | 8.00 | 329.1 | numpy | 0 |
| dense-operator | medium | tensorlang | 0.0007 | 16.00 | 345.2 | materialized | 1 |
| dense-operator | medium | native | 0.0005 | 8.00 | 345.2 | numpy | 0 |
| matmul-1000x1000 | medium | tensorlang | 0.0094 | 8.00 | 369.6 | materialized | 1 |
| matmul-1000x1000 | medium | native | 0.0097 | 8.00 | 369.6 | numpy | 0 |
| filter-2-predicates | medium | tensorlang | 0.0022 | 0.01 | 470.3 | materialized | 1 |
| filter-2-predicates | medium | native | 0.0020 | 0.00 | 490.6 | polars | 0 |
| filter-derive | medium | tensorlang | 0.0105 | 0.01 | 572.2 | polars-lazy | 1 |
| filter-derive | medium | native | 0.0094 | 0.00 | 589.4 | polars-lazy | 0 |
| grouped-sum | medium | tensorlang | 0.0169 | 0.01 | 600.6 | polars-lazy | 1 |
| grouped-sum | medium | native | 0.0194 | 0.00 | 600.7 | polars-lazy | 0 |
| pipeline-write | medium | tensorlang | 0.0145 | 0.01 | 600.7 | polars-lazy | 1 |
| pipeline-write | medium | native | 0.0154 | 0.00 | 600.7 | polars-lazy | 0 |
| snapshot-10-branches | medium | tensorlang | 0.0000 | 0.00 | 735.5 | cow-views | 0 |
|  |  |  | tensor=80.0MB graphs=0.0MB for 80MB data |  |  |  |  |
| branch-collect | medium | tensorlang | 0.0000 | 0.00 | 895.5 | materialized | 0 |
|  |  |  | collect added 160.0MB |  |  |  |  |
| cache-release | medium | tensorlang | 0.0000 | 0.00 | 895.5 | n/a | 0 |
|  |  |  | release() clears retained collect() results |  |  |  |  |
| external-copy-once | medium | tensorlang | 0.0000 | 0.00 | 975.2 | n/a | 0 |
|  |  |  | caller-side mutation cannot leak into graphs |  |  |  |  |
| grad-depth-1 | medium | tensorlang | 0.0026 | 6.41 | 975.2 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.39ms backward=2.31ms |  |  |  |  |
| value-and-grad-depth-1 | medium | tensorlang | 0.0024 | 6.41 | 975.2 | autodiff-shared | 1 |
|  |  |  | forward=0.36ms backward=2.27ms |  |  |  |  |
| grad-depth-8 | medium | tensorlang | 0.0050 | 23.21 | 975.2 | autodiff-vs-autograd | 1 |
|  |  |  | forward=1.41ms backward=4.48ms |  |  |  |  |
| value-and-grad-depth-8 | medium | tensorlang | 0.0042 | 23.21 | 975.2 | autodiff-shared | 1 |
|  |  |  | forward=1.15ms backward=4.08ms |  |  |  |  |
