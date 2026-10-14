# TensorLang v0.6 roadmap (workload-driven, evidence-ranked)

Ranked by (1) workflow impact, (2) ontology fit, (3) complexity,
(4) perf/memory leverage, (5) compatibility, (6) testability.
Correctness fixes first, then enhancements; multi-workload wins first.

## Backport to v0.5.4 — DONE (committed `5b50c9b`, tagged `v0.5.4` separately)

- **F8** (own-publish baseline + 2 regression tests; 7 files).
  Zero semantics change for all other paths; suite green; clean-install
  smoke verified.

## v0.6 milestone 1 — scale the numerics (F1 + F2) — DONE this phase

- Executor: sharing-preserving `remap_params` (identical output, no
  uid-policy change), scoped evaluation memo with use-count eviction,
  structural-key dedupe with taint isolation (proven by induced
  gradient-descent divergence when disabled). Energy 5.33x → 0.68x
  on the rewritten workload (v1 engine-only number: 0.38–0.60x; delta is
  NumPy strided-sum vs GEMM kernel behavior — target ≤2x exceeded
  either way).
- AD: pop-after-propagate cotangent release + structural-aliasing
  forward tape (complete per-node values preserved). RSS 9.3x → ≈2.45x
  (target ≤2.5x met); gradient 11.88x → 4.15x with traced attribution
  (total ≈ backward-inclusive ⊃ forward; exclusive ≈ 0.79 s).
  Checkpointing not needed.
- 355 → 428 tests (6 M1 + 67 M2 new: dedupe/taint/release/scan-count
  guards; axis/keepdims/transpose differential suites);
  native parity held on data + out-of-core regressions.

## v0.6 milestone 2 — close the expressiveness gaps (F3 + F4) — DONE this phase

- Axis reductions (`axis=None/int/negative/tuple`, `keepdims`) across
  Tensor/Lazy/Operator, dense executor, AD rules (FD-verified 30/30),
  and chunked lowering with plan/budget integration; `TL0511` axis
  diagnostics; lazy shape inference without materialization.
- Operator `.T` builds transpose (field `"T"` still via `X["T"]`);
  Tensor/Lazy `.T` already existed; field-confusion errors now suggest
  the operator function.
- Workload rewritten: 32 → 27 LOC, zero workarounds (target ~24 held
  approximately with readability preserved); v1 frozen for attribution.
- Backward follow-up: bottleneck persists as genuine numpy arithmetic
  (0.37 s reductions + 0.34 s cotangent ops of 0.88 s); no separate
  optimizer recommended — evidence in `SCIENTIFIC.md`.

## Explicitly deferred (evidence: not needed by any workload)

- Partitioned contractions / data-dependent indexing / chunked AD:
  correctly rejected today; revisit only with a heavier-algebra
  workload that demonstrably needs them.
- F5 (frame→dense bridge), F6 (more lazy aggregations), F7 (write
  helpers): conveniences; revisit if ML-pipeline evaluation demands.

## Progression criteria

Each milestone re-runs `evaluation/run_*.py` and must improve the
target numbers without regressing equivalence, the 428-test suite, or
native-parity benchmarks. No new operator families beyond F3's axis
parameter.
