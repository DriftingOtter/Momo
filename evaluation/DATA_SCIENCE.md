# Workload 2 — customer/transaction pipeline (data science + ML)

`evaluation/workload_data.py`, driver `evaluation/run_data.py`, results
`evaluation/results/data_science.json`. 20K customers / 200K
transactions (Parquet); medians of 5; per-process RSS; equivalence
asserted in-run (row counts, null counts, sign-invariant PCA
projections to 1e-9).

## What was built

Mask filter → join → derived `net` column → grouped sums/means →
left-join feature assembly (nulls for customer without transactions) →
key-sorted frames → explicit `collect()` to dense → shared
train/held-out split → `impute >> standardize >> pca(2)` fit on train,
applied to both splits. Native mirror in Polars + scikit-learn.

## Correctness — all equivalent

| Check | Result |
|---|---|
| Row counts / null counts TL vs native | identical |
| Imputed arrays TL vs `SimpleImputer` | exact |
| Standardized arrays TL vs `StandardScaler` | exact |
| PCA eigenvalues / components vs sklearn | exact (abs) |
| Train + holdout projections vs sklearn | 1e-9 (sign-invariant) |
| Train-stats reuse on held-out | exact (existing suite property) |

Two methodology traps found and fixed during evaluation (both
documented in the workload): join/group output order differs across
processes, so positional splits diverged until both sides key-sort;
an early sign error in the evaluator's own analytic reference (workload
1), caught by cross-checking.

## Performance (20K/200K, medians)

| Phase | TensorLang | Polars/sklearn | Ratio |
|---|---|---|---|
| relational | 0.0052 s | 0.0042 s | 1.22x |
| fit | 0.0005 s | 0.0020 s | **0.26x** |
| apply train | 0.0004 s | 0.0011 s | **0.34x** |
| apply held-out | 0.0002 s | 0.0010 s | **0.20x** |

Peak RSS: TensorLang 167 MB vs native 204 MB (parity).

## Expressiveness (16 LOC TL vs 14 native)

Essentially at parity — the strongest expressiveness result of the
evaluation. The reusable pipeline object (`impute >> standardize >>
pca`) with train/test discipline reads as cleanly as the sklearn
equivalent. Costs recorded: one explicit `collect()` boundary between
the relational and ML stages (no lazy frame→dense bridge exists), and
no lazy count aggregation (features use sum/mean only).

## Verdict

This workload is a genuine success for the v0.5 architecture: correct,
concise, and faster than the native stack on fitting/inference at this
scale (exact solvers beat sklearn's overhead-bound paths here; expect
convergence at larger scales). v0.6 candidates: lazy frame→dense bridge
(§7-F5), count aggregation (§7-F6). Neither blocks real work — the
`collect()` boundary is explicit and honest.
