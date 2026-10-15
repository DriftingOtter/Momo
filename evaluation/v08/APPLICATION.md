# v0.8 M0 application — scientific experiment analysis

`evaluation/v08/app/` (driver `driver.py`, stages under `io/`,
`operators/`; single-file baseline `main.tl`; native reference
`reference/native_reference.py`; data script `data/generate_data.py`;
tests `tests/test_application.py`).

## Data (deterministic, seeded, no network)

50 experiments × 2000 measurements (100K rows): `experiment_id`,
`t`, `meas` (5% true nulls), `temp`, `category`, plus per-experiment
`gain`/`offset` calibration parameters.

## Pipeline

1. Load scans (`io/sources.tl`, paths relative to `__file__`).
   (M1: path joins are precomputed with `=` — arbitrary calls cannot
   nest under `:=`, see `M1_EFFECTS.md`.)
2. Filter valid rows (`not_null` — NaN/`==`-self filtering does NOT
   work: Polars `NaN == NaN` is true, unlike NumPy).
3. Join observations with calibration parameters.
4. Row-level calibration via the cross-file reusable operator
   (`calibrate := {M} * {G} + {O}`), derived **inside** the lazy graph
   — positional attachment across executions is unsafe because join
   order is backend-defined and differs from native polars.
5. Per-experiment `mean`/`count` aggregates.
6. `to_dense(["cal", "temp"])` conversion.
7. `standardize` fitting; ridge solve for the temperature coefficient.
8. Parquet output + `explain()` diagnostics.

## Correctness

Temperature coefficient 0.6311 (TensorLang) vs 0.6311 (native); all
intermediate aggregates verified stage-by-stage during development.
95K rows after null filtering (deterministic).

## Performance (100K rows, medians where repeated)

| Stage | TL | Native |
|---|---|---|
| stage loading (3 `.tl` files) | 1.4 ms | — (import) |
| filter / join / derive | 17–31 ms total | 24 ms total |
| grouped aggs | ~16 ms total | ~12 ms |
| convert | 3–5 ms | 1–3 ms |
| standardize fit/apply | 1–4 ms | 5 ms |
| solve (2×2) | ~126 ms first call (SciPy init), <1 ms after | <1 ms |
| cold app (`tensorlang run main.tl`) | 0.33 s wall, 147 MB peak | — |
| driver peak RSS | 179 MB | — |

Stage files: 17 non-comment lines of `.tl` total; driver 71 vs native
47 lines (stage loading, timing, and diagnostics scaffolding account
for the difference).
