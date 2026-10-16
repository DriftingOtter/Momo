# v0.9 M0 — performance

macOS arm64, 8 CPUs, 8 GB RAM, py3.14, installed wheel. Medians where
repeated; import overhead separated from execution.

| App | Cold `tensorlang run` | Peak RSS | Entry LOC | Native ref |
|---|---|---|---|---|
| A (GD fit, 400 obs, 1500 iters) | 2.1 s | 108 MB | 31 | SciPy: params agree 1e-9 |
| B (30k rows, ridge) | 0.9 s | 169 MB | 72 | sklearn: mse 1e-12 |
| C (512 MB zarr) | ~3 s run* | 697 MB vs 750 MB dask | 18 | dask: mean 16 digits |

\* C cold-start dominated by interpreter + first zarr read; steady
 re-runs faster (page cache). Module loading ~2 ms/file cold,
 0.1 ms cached (v0.8 baselines hold; not re-timed precisely here).

No unexplained regressions: costs sit in data/BLAS/polars/scipy, not
in validation or module loading. RSS interpretation follows v0.5
rules (estimates are planning heuristics, never RSS caps).
