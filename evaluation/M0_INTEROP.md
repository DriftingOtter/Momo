# M0 — Mixed structured/dense evaluation (1M transactions)

v0.7 M1 update: v1 (explicit conversion) vs v2 (lazy bridge) comparison
below; bridge implementation landed per this report's scoping.

`evaluation/workload_mixed.py`, driver `evaluation/run_mixed.py`,
results `evaluation/results/mixed.json`. 100K customers / 1M
transactions (numeric, nullable-int, categorical, identifier columns);
warmup + median-of-5; per-process RSS; equivalence asserted in-run
(sign-invariant projections, ridge predictions to 1.7e-13).

## Stage measurements (100K/1M medians)

| Phase | TensorLang | Native | Ratio |
|---|---|---|---|
| relational | 0.053 s | 0.047 s | 1.12x |
| convert (frame→numpy→Tensor) | 0.0031 s | 0.0012 s | — |
| fit (impute≫standardize≫pca) | 0.028 s | 0.020 s | 1.39x |
| infer train / holdout | 0.015 / 0.004 s | 0.009 / 0.003 s | 1.62x / 1.18x |
| ridge solve (3×3) | 0.0001 s | 0.0039 s | µs-noise |

Peak RSS: 608 vs 632 MB (parity). User LOC: 43 vs 15.

Scale note: at 2K/20K rows TL fit/infer was 0.2–0.4x native; at
80K×6 rows it is 1.2–1.7x. Crossover comes from fixed per-operator
costs against tiny absolute times (all stages < 30 ms) — not a
scaling wall, but worth re-measuring at 10M+ rows before claiming
parity holds generally.

## Boundary answers (P1–P5)

- **P1 — lazy input to fitting**: `pca().fit(lazy_frame)` fails with
  `'Lazy' object is not callable` (no code, no guidance). Fitting a
  materialized `Tensor` frame works. The boundary is enforced but
  undiscoverable — diagnostic deficiency, not just a missing feature.
- **P2 — re-execution**: collected results cache; re-collect ≈0.1 ms.
  No accidental recomputation.
- **P3 — copy semantics**: `to_numpy()` on the heterogeneous,
  null-bearing frame copies (proven: mutating the array leaves the
  frame intact); `T()` copies again per the documented once-at-boundary
  rule. The ~2 ms convert delta *is* those copies. Zero-copy is
  structurally unavailable here (mixed dtypes + nulls), so the honest
  bridge goal is *lazy, planned* copying — not zero-copy.
- **P4 — null/dtype policy**: nullable Int64 tenure → float64 NaN on
  `to_numpy()`; NaN ages/amounts flow through; mean-imputation matches
  sklearn exactly. Promotion is automatic and correct, but implicit —
  a lazy bridge must make it explicit and inspectable.
- **P5 — costs**: relational 1.12x (planning amortized); ML stages
  1.2–1.7x at this scale; ridge/solve exact; verbosity 43 vs 15 LOC
  (stage separation + explicit conversion/split/solve scaffolding).

## v2 bridge results (same data/splits/tolerances)

| Phase | v2 bridge | v1 explicit | Native |
|---|---|---|---|
| convert (relational+conv) | 0.041–0.052 s | 0.039+0.002 s | 0.034–0.045 s |
| fit / infer | same graphs as v1 | 0.019/0.010 s | 0.013/0.006 s |
| ridge | same | same | same |
| equivalence | bridge ≡ explicit bit-for-bit; projections 1e-9; ridge 7e-14 | — | — |
| peak RSS | 565–630 MB | 609–610 MB | 634–643 MB |

Lazy-reuse probe: fitting directly on a fresh lazy bridge costs 84 ms
vs 31 ms on materialized input (full relational re-execution per
boundary). Per-object Lazy caching is the documented mechanism —
collect once and reuse arrays for hot paths; each `.collect()` is an
explicit, admitted boundary, never hidden work. Module LOC unchanged
at 37 (conversion helper replaced by method calls; split/ridge
scaffolding dominates both versions).

## Correctness (both versions)

Row/null counts identical; imputed/standardized arrays exact; PCA
eigenvalues exact; projections to 1e-9 (sign-invariant); ridge weights
to 6.7e-14 (sign-adjusted), holdout predictions to 1.7e-13. One
workload bug of mine (NaN regression target poisoning `solve`) was
caught by the harness and fixed identically on both sides
(mean-imputation of the target).

## What the M1 bridge must do (scoped from evidence)

1. Accept a lazy frame + column list and produce a lazy dense Tensor
   without executing (`explain()`-visible, no data read).
2. Resolve output schema/shape/dtype *from metadata* (nullable-int →
   float64 rules stated, not discovered).
3. Attach an explicit null policy (reject unknown policies; never
   silently drop rows).
4. Admit against memory budgets (estimated dense bytes vs policy) or
   raise a coded error — never silently materialize.
5. Feed fitted operators directly (fix the P1 diagnostic regardless).
6. Preserve pushdown before the boundary (the 1.12x relational number
   must not regress).

## v0.7 M2 update: count features integrated

Grouped row counts (`frame.count(by=)`) and non-null field counts
(`series.count(by=)`) landed with native Polars parity (null keys form
their own group; u32 dtypes preserved; row-vs-value never conflated).
The mixed workload now engineers 7 features (count included), flowing
through the v2 bridge into PCA/ridge with identical equivalence
(projections 1e-9, ridge 7e-14). Null keys do not join (polars
semantics, verified identical natively).

## Limitations carried forward

F5 (bridge — done), F6 (count done; multi-key grouping still rejected
as before), F7 (verbosity — unchanged). No new operator families were
needed; nothing in this workload requires partitioned contractions or
chunked AD.
