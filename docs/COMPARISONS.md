# Missing values: NaN vs null across representations

NaN and null are **different spellings of missingness** with
representation-dependent comparison behavior. They are never
interchangeable; normalize at ingestion (the M0 application converts
ingested NaN to null with `fill_nan(None)`) and filter with null
predicates, never `==` self-comparisons.

Backend versions pinned: NumPy 2.5.4 (IEEE equality), Polars 2.0.0.

## Contract (tested in `tests/test_missing_values.py`)

| Case | Dense (NumPy) | Structured (Polars) |
|---|---|---|
| `NaN == NaN` | `False` (IEEE) | `True` (total-order `==`) |
| `null == x` | n/a (no null bit) | `null` (3-valued propagate) |
| `== None` literal | `TL0499` | `TL0499` |
| `is_null(NaN)` | `True` | `False` |
| `is_null(null)` | n/a | `True` |
| `x[x == x]` filter | drops NaN | keeps NaN, drops null row |
| `not_null()` filter | drops NaN | keeps NaN, drops null |
| `to_dense(nulls="nan")` | identity | null → NaN (nullable int → float64) |
| `to_dense(nulls="error")` | passes clean data | `TL0418` on any missing value |
| grouped `mean` with nulls | n/a | skips nulls |

## Rules

1. Filter missing values with `not_null(x)` / `is_null(x)`.
2. Never use `x == x` or `x == float("nan")` as a filter — it drops NaN
   in dense code but keeps NaN rows in structured code.
3. `is_null()` diverges on NaN by design (it reports the
   representation's missing bit; dense NumPy has only NaN). Normalize
   first when crossing representations.
4. `fill_null(x, v)` / `drop_null(x)` make handling explicit;
   `nulls="error"` makes conversion strict.
