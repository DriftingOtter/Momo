# TensorLang v0.6 evaluation — aggregate results

Environment: macOS 26.5.2 arm64, Python 3.14.7, 8 CPUs, 8.6GB RAM;
numpy 2.5.4, polars 2.0.0, sympy 1.14.0, scipy 1.18.1, dask 2026.8.0,
zarr 3.4.1; TensorLang v0.5.3 (`3970720`) plus one evaluation-found
defect fix (F8, uncommitted, on `evaluation/v06-workloads` only).
Methodology: identical datasets/seeds, equivalent algorithms, warmup +
median-of-5, per-process peak RSS (`time -l`), equivalence asserted
in-run. Scripts: `evaluation/workload_*.py`, drivers `run_*.py`.

## Correctness — 100% equivalent where comparable

Energy (14 digits), gradients (1e-6), inertia/eigh/COM (exact),
symbolic derivative (1e-5 vs FD), relational row/null counts (exact),
impute/standardize/PCA (exact incl. eigenvalues), projections (1e-9
sign-invariant), z-score outputs (1e-9). Two evaluator-side reference
bugs (analytic sign; cross-process join order) were caught BY the
cross-checks — the methodology works.

## Performance (medians; µs rows are noise)

| Workload | Phase | TL | Native | Ratio |
|---|---|---|---|---|
| scientific N=2000 | energy | 0.10 s | 0.15 s | **0.68x** (M1 engine; M2 rewrite; was 5.33x) |
| scientific N=2000 | gradient | 0.87 s | 0.21 s | **4.15x** (M1; was 11.88x; fwd 0.08 s, bwd-incl-fwd 0.87 s, bwd-excl ≈0.79 s — boundaries traced, not summed) |
| scientific N=2000 | inertia+eigh | 0.2 ms | 0.04 ms | µs-noise |
| data 20K/200K | relational | 5.2 ms | 4.2 ms | 1.22x |
| data 20K/200K | fit / apply | 0.2–0.5 ms | 1.0–2.0 ms | **0.20–0.34x** |
| ooc 256MB/16MB | normalize+write | 1.75–3.59 s | 0.99–3.39 s | 1.06–1.77x |

## Memory (per-process peak RSS)

Scientific: 1059–1067 vs 433 MB (**≈2.45x**, M1+M2 held; was 9.3x). Data: 167 vs 204 MB (parity).
Out-of-core: 172–342 vs 100–277 MB (1.2–1.7x; bounded, chunk-driven).

## Expressiveness (user LOC, comments/blank excluded)

Scientific v2: 27 vs 20 (v1 baseline: 32 with workarounds; all removed
in the rewrite — target ~24 held approximately with readability
preserved). Data 16 vs 14 (parity — best result). Out-of-core 23 vs 12
(native terser; TL more inspectable). Fewer lines did not always mean
clearer: the TL data pipeline reads as well as sklearn; the v2
scientific code now reads naturally.

## Error quality

Coded diagnostics identified every probed failure (TL0502 shapes,
TL1001/TL1010 unsupported, TL1024 budgets, TL1030 sources, TL0511
axes) with reasons and alternatives. F4 fixed in M2: `.T` builds
transpose; residual field-confusion errors suggest the operator
function.

## 7. Architectural limitations (findings catalog)

- **F1 — executor graph bloat (ROOT-CAUSED, fixed).** Per-composition
  whole-tree copies + per-reference doubling (42 BinOps for ~12 user
  ops; 32/42 evaluations value-identical; per-call wrapper overhead
  only ~6%). Fixed via sharing-preserving `remap_params`, scoped memo
  with use-count eviction, and structural evaluation dedupe with taint
  isolation. Energy now 0.68x native on the v2 rewrite (v1 engine-only
  number was 0.38–0.60x; the v1↔v2 delta is NumPy strided-sum vs GEMM
  kernel behavior, identical for native code).
- **F2 — reverse-mode buffer retention (FIXED within guarantees).**
  Pop-after-propagate cotangent release (52/pass on LJ); forward tape
  stays pinned per the tested `forward_runs == 1` repeat guarantee.
  RSS 9.3x → ≈2.45x. Checkpointing not needed.
- **F3 — axis reductions (DONE in M2).** `axis=`/`keepdims` across
  Tensor/Lazy/Operator, executor, AD rules (FD-verified), chunked
  lowering with budget/plan integration, `TL0511` diagnostics.
- **F4 — `.T`/attribute ergonomics on operators.** `X.T` builds, then
  fails at apply with an undiscoverable `TL0499`. Reproducer:
  ```python
  X = P("X"); E = X.T + 1
  E(T(np.ones((3, 3)))).collect()  # TL0499: cannot access field 'T' ...
  ```
  Fix (DONE in M2): `.T` builds transpose on operators; field-confusion
  errors suggest the operator function; `X["T"]` preserved for fields.
- **F5 — no lazy frame→dense bridge.** Explicit `collect()` boundary
  (honest today, inconvenient for ML pipelines).
- **F6 — narrow lazy aggregations.** sum/mean only; no count/variance-
  by-group parity with the grouped surface.
- **F7 — write-path verbosity.** Budget + reductions + write + verify
  is ~2x native lines; convenience helpers, not language.
- **F8 — own-publish TL1030 false positive (correctness fix, DONE on
  this branch).** Re-reading just-published output in a long-lived
  process tripped source-consistency on the workload's own mtime.
  Reproducer: write then `T.scan_zarr(out).mean().collect()` twice in
  one process (second fails pre-fix). Fix: successful publishes
  establish the baseline; external changes still detected; suite green
  green. Recommend v0.5.4 backport.

Not needed by any workload: partitioned contractions, data-dependent
indexing, chunked AD (all correctly rejected; the hypothesized biggest
gap did not bind — do not prioritize without heavier-algebra evidence).
