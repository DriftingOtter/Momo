# TensorLang benchmark results

## Machine

- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- python: 3.14.7
- cpus: 8
- tensorlang: 0.5.3
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
| dense-chain | small | tensorlang | 0.0003 | 1.60 | 138.6 | materialized | 1 |
| dense-chain | small | native | 0.0001 | 0.80 | 138.6 | numpy | 0 |
| dense-operator | small | tensorlang | 0.0002 | 1.60 | 139.4 | materialized | 1 |
| dense-operator | small | native | 0.0001 | 0.80 | 139.4 | numpy | 0 |
| matmul-500x500 | small | tensorlang | 0.0013 | 2.00 | 149.6 | materialized | 1 |
| matmul-500x500 | small | native | 0.0012 | 2.00 | 149.6 | numpy | 0 |
| filter-2-predicates | small | tensorlang | 0.0008 | 0.01 | 186.3 | materialized | 1 |
| filter-2-predicates | small | native | 0.0005 | 0.00 | 189.2 | polars | 0 |
| filter-derive | small | tensorlang | 0.0043 | 0.01 | 230.5 | polars-lazy | 1 |
| filter-derive | small | native | 0.0035 | 0.00 | 236.7 | polars-lazy | 0 |
| grouped-sum | small | tensorlang | 0.0048 | 0.02 | 247.0 | polars-lazy | 1 |
| grouped-sum | small | native | 0.0040 | 0.00 | 247.5 | polars-lazy | 0 |
| pipeline-write | small | tensorlang | 0.0049 | 0.02 | 250.0 | polars-lazy | 1 |
| pipeline-write | small | native | 0.0043 | 0.00 | 250.1 | polars-lazy | 0 |
| snapshot-10-branches | small | tensorlang | 0.0000 | 0.00 | 266.1 | cow-views | 0 |
|  |  |  | tensor=8.0MB graphs=0.0MB for 8MB data |  |  |  |  |
| branch-collect | small | tensorlang | 0.0000 | 0.00 | 282.2 | materialized | 0 |
|  |  |  | collect added 16.1MB |  |  |  |  |
| cache-release | small | tensorlang | 0.0000 | 0.00 | 282.2 | n/a | 0 |
|  |  |  | release() clears retained collect() results |  |  |  |  |
| external-copy-once | small | tensorlang | 0.0000 | 0.00 | 290.2 | n/a | 0 |
|  |  |  | caller-side mutation cannot leak into graphs |  |  |  |  |
| grad-depth-1 | small | tensorlang | 0.0006 | 0.07 | 292.3 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.25ms backward=0.54ms |  |  |  |  |
| value-and-grad-depth-1 | small | tensorlang | 0.0006 | 0.07 | 292.4 | autodiff-shared | 1 |
|  |  |  | forward=0.20ms backward=0.48ms |  |  |  |  |
| grad-depth-8 | small | tensorlang | 0.0015 | 0.25 | 292.9 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.57ms backward=1.43ms |  |  |  |  |
| value-and-grad-depth-8 | small | tensorlang | 0.0016 | 0.25 | 292.9 | autodiff-shared | 1 |
|  |  |  | forward=0.58ms backward=1.41ms |  |  |  |  |
| chunked-apply | small | tensorlang | 0.0832 | 26.06 | 403.6 | dask-chunked | 1 |
|  |  |  | input=12.8MB |  |  |  |  |
| chunked-apply | small | native | 0.0293 | 38.41 | 421.9 | numpy-direct-read | 0 |
|  |  |  | native reads the whole store (unbounded) |  |  |  |  |
| chunked-reduce | small | tensorlang | 0.0989 | 15.67 | 422.2 | dask-chunked | 1 |
| chunked-write | small | tensorlang | 0.1331 | 16.21 | 435.6 | dask-chunked | 1 |
| chunked-aligned | small | tensorlang | 0.0336 | 37.79 | 492.3 | dask-chunked | 1 |
|  |  |  | tasks=9 |  |  |  |  |
| chunked-misaligned | small | tensorlang | 0.0814 | 53.14 | 518.9 | dask-chunked | 1 |
|  |  |  | tasks=68 throughput=157.2MB/s |  |  |  |  |
| chunked-misaligned | small | native-dask | 0.0715 | 53.18 | 523.5 | dask | 0 |
|  |  |  | raw dask baseline |  |  |  |  |
| chunked-transpose | small | tensorlang | 0.0355 | 37.77 | 536.4 | dask-chunked | 1 |
| chunked-reshape | small | tensorlang | 0.0329 | 37.78 | 536.4 | dask-chunked | 1 |
| chunked-matmul | small | tensorlang | 0.0447 | 0.69 | 537.0 | dask-chunked | 1 |
|  |  |  | throughput=0.7MB/s |  |  |  |  |
| chunked-matmul | small | native-dask | 0.0348 | 0.67 | 537.1 | dask | 0 |
| chunked-budget | small | tensorlang | 0.0863 | 26.03 | 538.2 | dask-chunked | 1 |
|  |  |  | c10000-w4-b64MB tasks=82 outcome=admitted est=20.50MB workers=4 throughput=148.3MB/s |  |  |  |  |
| chunked-budget | small | native-dask | 0.0831 | 26.07 | 538.3 | dask | 0 |
|  |  |  | c10000-w4-b64MB raw dask, same worker cap |  |  |  |  |
| chunked-budget-reduce | small | tensorlang | 0.1023 | 9.30 | 538.4 | dask-chunked | 1 |
|  |  |  | c10000-w4-b64MB reduce 12.8MB under 64MB budget |  |  |  |  |
| chunked-budget | small | tensorlang | 0.3093 | 26.57 | 539.3 | dask-chunked | 1 |
|  |  |  | c2000-w4-b16MB tasks=402 outcome=admitted est=14.44MB workers=4 throughput=41.4MB/s |  |  |  |  |
| chunked-budget | small | native-dask | 0.2748 | 26.30 | 539.3 | dask | 0 |
|  |  |  | c2000-w4-b16MB raw dask, same worker cap |  |  |  |  |
| chunked-budget-reduce | small | tensorlang | 0.3751 | 3.11 | 540.3 | dask-chunked | 1 |
|  |  |  | c2000-w4-b16MB reduce 12.8MB under 16MB budget |  |  |  |  |
| chunked-budget | small | tensorlang | 0.3602 | 26.51 | 540.4 | dask-chunked | 1 |
|  |  |  | c2000-w1-b16MB tasks=402 outcome=admitted est=13.29MB workers=1 throughput=35.5MB/s |  |  |  |  |
| chunked-budget | small | native-dask | 0.3221 | 26.20 | 540.4 | dask | 0 |
|  |  |  | c2000-w1-b16MB raw dask, same worker cap |  |  |  |  |
| chunked-budget-reduce | small | tensorlang | 0.4214 | 1.83 | 540.4 | dask-chunked | 1 |
|  |  |  | c2000-w1-b16MB reduce 12.8MB under 16MB budget |  |  |  |  |
| chunked-over-budget | small | tensorlang | 0.3920 | 2.23 | 540.4 | dask-chunked | 1 |
|  |  |  | reduce 12.8MB under 2MB budget |  |  |  |  |
| chunked-over-budget | small | tensorlang | 0.5338 | 2.07 | 540.4 | dask-chunked | 1 |
|  |  |  | stream 12.8MB output under 2MB budget |  |  |  |  |
| chunked-write-mode | small | tensorlang | 0.1320 | 16.21 | 540.4 | dask-chunked | 1 |
|  |  |  | staged publish=4.7ms validate=4.4ms final=12.8MB transient~25.6MB throughput=96.9MB/s |  |  |  |  |
| chunked-write-mode | small | tensorlang | 0.1289 | 16.88 | 540.4 | dask-chunked | 1 |
|  |  |  | direct publish=0.0ms validate=0.0ms final=12.8MB transient~0MB throughput=99.3MB/s |  |  |  |  |
| chunked-write-cleanup | small | tensorlang | 0.0203 | 0.00 | 540.4 | dask-chunked | 0 |
|  |  |  | fail-after-1-partition abort+cleanup, destination absent, no staging leaked |  |  |  |  |
| dense-chain | medium | tensorlang | 0.0014 | 16.00 | 540.4 | materialized | 1 |
| dense-chain | medium | native | 0.0012 | 8.00 | 540.4 | numpy | 0 |
| dense-operator | medium | tensorlang | 0.0007 | 16.00 | 555.9 | materialized | 1 |
| dense-operator | medium | native | 0.0005 | 8.00 | 555.9 | numpy | 0 |
| matmul-1000x1000 | medium | tensorlang | 0.0096 | 8.00 | 580.3 | materialized | 1 |
| matmul-1000x1000 | medium | native | 0.0096 | 8.00 | 580.3 | numpy | 0 |
| filter-2-predicates | medium | tensorlang | 0.0024 | 0.01 | 679.3 | materialized | 1 |
| filter-2-predicates | medium | native | 0.0020 | 0.00 | 690.4 | polars | 0 |
| filter-derive | medium | tensorlang | 0.0108 | 0.01 | 753.6 | polars-lazy | 1 |
| filter-derive | medium | native | 0.0096 | 0.00 | 768.3 | polars-lazy | 0 |
| grouped-sum | medium | tensorlang | 0.0109 | 0.01 | 778.2 | polars-lazy | 1 |
| grouped-sum | medium | native | 0.0091 | 0.00 | 779.8 | polars-lazy | 0 |
| pipeline-write | medium | tensorlang | 0.0096 | 0.01 | 779.8 | polars-lazy | 1 |
| pipeline-write | medium | native | 0.0095 | 0.00 | 779.8 | polars-lazy | 0 |
| snapshot-10-branches | medium | tensorlang | 0.0000 | 0.00 | 911.9 | cow-views | 0 |
|  |  |  | tensor=75.3MB graphs=0.0MB for 80MB data |  |  |  |  |
| branch-collect | medium | tensorlang | 0.0000 | 0.00 | 978.5 | materialized | 0 |
|  |  |  | collect added 66.6MB |  |  |  |  |
| cache-release | medium | tensorlang | 0.0000 | 0.00 | 978.5 | n/a | 0 |
|  |  |  | release() clears retained collect() results |  |  |  |  |
| external-copy-once | medium | tensorlang | 0.0000 | 0.00 | 993.7 | n/a | 0 |
|  |  |  | caller-side mutation cannot leak into graphs |  |  |  |  |
| grad-depth-1 | medium | tensorlang | 0.0022 | 6.41 | 994.9 | autodiff-vs-autograd | 1 |
|  |  |  | forward=0.33ms backward=2.11ms |  |  |  |  |
| value-and-grad-depth-1 | medium | tensorlang | 0.0022 | 6.41 | 994.9 | autodiff-shared | 1 |
|  |  |  | forward=0.32ms backward=2.11ms |  |  |  |  |
| grad-depth-8 | medium | tensorlang | 0.0045 | 23.21 | 994.9 | autodiff-vs-autograd | 1 |
|  |  |  | forward=1.18ms backward=4.42ms |  |  |  |  |
| value-and-grad-depth-8 | medium | tensorlang | 0.0046 | 23.21 | 994.9 | autodiff-shared | 1 |
|  |  |  | forward=1.17ms backward=4.34ms |  |  |  |  |
| chunked-apply | medium | tensorlang | 0.3422 | 256.99 | 1694.0 | dask-chunked | 1 |
|  |  |  | input=128.0MB |  |  |  |  |
| chunked-apply | medium | native | 0.2218 | 384.02 | 1694.0 | numpy-direct-read | 0 |
|  |  |  | native reads the whole store (unbounded) |  |  |  |  |
| chunked-reduce | medium | tensorlang | 0.4040 | 34.12 | 1694.0 | dask-chunked | 1 |
| chunked-write | medium | tensorlang | 0.5429 | 32.92 | 1694.0 | dask-chunked | 1 |
| chunked-aligned | medium | tensorlang | 0.0900 | 375.81 | 1694.0 | dask-chunked | 1 |
|  |  |  | tasks=9 |  |  |  |  |
| chunked-misaligned | medium | tensorlang | 0.1502 | 577.58 | 1742.4 | dask-chunked | 1 |
|  |  |  | tasks=68 throughput=852.3MB/s |  |  |  |  |
| chunked-misaligned | medium | native-dask | 0.1403 | 577.56 | 1767.1 | dask | 0 |
|  |  |  | raw dask baseline |  |  |  |  |
| chunked-transpose | medium | tensorlang | 0.1298 | 375.80 | 1829.7 | dask-chunked | 1 |
| chunked-reshape | medium | tensorlang | 0.0846 | 375.80 | 1829.7 | dask-chunked | 1 |
| chunked-matmul | medium | tensorlang | 0.0448 | 0.69 | 1829.7 | dask-chunked | 1 |
|  |  |  | throughput=0.7MB/s |  |  |  |  |
| chunked-matmul | medium | native-dask | 0.0347 | 0.67 | 1829.7 | dask | 0 |
| chunked-budget | medium | tensorlang | 0.0000 | 0.00 | 1829.7 | dask-chunked | 0 |
|  |  |  | c10000-w4-b64MB tasks=402 rejected TL1024: 128.0MB output vs 64MB budget |  |  |  |  |
| chunked-budget-reduce | medium | tensorlang | 0.4259 | 19.15 | 1829.7 | dask-chunked | 1 |
|  |  |  | c10000-w4-b64MB reduce 128.0MB under 64MB budget |  |  |  |  |
| chunked-budget | medium | tensorlang | 0.0000 | 0.00 | 1829.7 | dask-chunked | 0 |
|  |  |  | c2000-w4-b16MB tasks=2002 rejected TL1024: 128.0MB output vs 16MB budget |  |  |  |  |
| chunked-budget-reduce | medium | tensorlang | 1.8723 | 10.10 | 1829.7 | dask-chunked | 1 |
|  |  |  | c2000-w4-b16MB reduce 128.0MB under 16MB budget |  |  |  |  |
| chunked-budget | medium | tensorlang | 0.0000 | 0.00 | 1829.7 | dask-chunked | 0 |
|  |  |  | c2000-w1-b16MB tasks=2002 rejected TL1024: 128.0MB output vs 16MB budget |  |  |  |  |
| chunked-budget-reduce | medium | tensorlang | 2.2867 | 7.22 | 1829.7 | dask-chunked | 1 |
|  |  |  | c2000-w1-b16MB reduce 128.0MB under 16MB budget |  |  |  |  |
| chunked-over-budget | medium | tensorlang | 2.2741 | 7.50 | 1829.7 | dask-chunked | 1 |
|  |  |  | reduce 128.0MB under 2MB budget |  |  |  |  |
| chunked-over-budget | medium | tensorlang | 3.3233 | 5.03 | 1829.7 | dask-chunked | 1 |
|  |  |  | stream 128.0MB output under 2MB budget |  |  |  |  |
| chunked-write-mode | medium | tensorlang | 0.5388 | 32.74 | 1829.7 | dask-chunked | 1 |
|  |  |  | staged publish=23.6ms validate=13.4ms final=128.0MB transient~256.0MB throughput=237.6MB/s |  |  |  |  |
| chunked-write-mode | medium | tensorlang | 0.5516 | 33.86 | 1829.7 | dask-chunked | 1 |
|  |  |  | direct publish=0.0ms validate=0.0ms final=128.0MB transient~0MB throughput=232.1MB/s |  |  |  |  |
| chunked-write-cleanup | medium | tensorlang | 0.0347 | 0.00 | 1829.7 | dask-chunked | 0 |
|  |  |  | fail-after-1-partition abort+cleanup, destination absent, no staging leaked |  |  |  |  |
