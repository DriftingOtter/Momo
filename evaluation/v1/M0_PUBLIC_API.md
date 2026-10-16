# M0 — Public API audit

Baseline v0.9.0 (`bbbef00`). `len(tensorlang.__all__) == 82` verified (81 functional + `__version__ == "0.9.0"`); zero duplicates/missing; `git diff v0.8.0..HEAD -- src/tensorlang/__init__.py` shows version bump only — no export added/removed/renamed in v0.9.

## 1. Observable behavior by group

- Core `T/Tensor/Lazy/Operator/P`: scalar/seq/mapping/ndarray/frame accepted; `T` copies at boundary (input mutation doesn't leak); rebinding never mutates; only `.inplace()` mutates. Errors `TL0305`/`TL0499`/`TL0201-203`.
- Math `sin..sqrt/diff/integrate/simplify`: pure; `TL07xx`/`TL0806`.
- Shape `concat/stack/reshape/flatten/transpose`: pure; `TL05xx`.
- Logic `logical_and/or/not`, `equal/not_equal`: `TL0512` fail-closed verified (`&` across domains raises); dense-vs-structured NaN divergence by design (B).
- Relational `join/sort/unique`: explicit keys; join order backend-defined (documented); `TL0430/431/0510`, chunked global-op `TL1013`.
- `where/clip/over/rolling`: `TL0504/0505`.
- Fitted `standardize/minmax/normalize/impute/pca/FittableOperator/PipelineOperator`: factories → fittable; `fit` returns new; `TL0419` at apply time verified (incl. pipeline stages); `TL0410-416/0508/0509`.
- AD `grad/value_and_grad/vjp/...`: first-order only; `TL0801-809` (incl. 2nd-order + chunked-deferred rejections).
- Linalg `svd/eigh/solve/norm`: `TL0506/0507`; scipy-backed.
- `cov/corr/quantile`, `einsum/tensordot/moveaxis`, `onehot/polynomial`, null ops + `to_dense(nulls=)` (`TL0418`).
- Entry `run_source/run_file/check_source/diagnose_*`: isolated sessions; `check_*` pure; `TL01xx-03xx` + `TL0310-315` chains.
- Chunk policy `set/clear/current_policy`: process-global policy mutation (explicit); `TL10xx` admission family.
- 10 error classes with `(message, loc, code, details)`; 63+ codes incl. new `TL0419`/`TL0512`.

Doc-vs-impl notes: no per-symbol 82-row table exists (both `docs/API_STABILITY.md` and `evaluation/v09/M0_PUBLIC_API.md` are area-based) — count claim holds, not row-diffable. `check_source(..., trusted_names, injected_names)` still exposes plumbing (B/D, still open). Aggregated column names, grouped-float ~1e-12 order noise, `==`-of-masks unfused, method-named-field `TypeError`, REPL `:reset` retention, `extra=` `=`-override — all B, impl matches docs.

## 2. A/B/C/D retention + v1.0 candidates

Retain A per `docs/API_STABILITY.md` (language core, data+math, modules+sessions, project workflow, C-resolved, D-experimental).
Genuine v1.0-stable candidates (narrow A): `T` shapes; `:=`/`=`/`{X}`/`P`; def/apply/`>>`; `collect()` + snapshot; `run_source`/`run_file` (minus `extra=` quirk); `diagnose_*` shapes; code identities `TL0101–TL1035` (codes, not wording); `to_dense(nulls=)`; `fit`-returns-new; `TL0419`/`TL0512` verdicts; `(path,name)` identity + `TL0310–315`; `tl run/check` exit contract.
Risk register — B (aggregate names, float order, NaN contract, `check_source` params, REPL retention, `extra=` override, penalty convention, `to_dense` default); C (F-B4/TL0512, F-A2 identity, TL0419 — present, lock-in against regression); D (trusted/injected plumbing, chunked-AD deferrals, REPL history, benchmarks, `tl build`/`publish` nonexistent-by-design) — excluded from v1.0.

## 3. M0 verdict

82-export inventory reconciles; v0.9 added no exports, two fail-closed codes. M1: freeze narrow-A list, clarify/deprecate B items (esp. `check_source` params), keep D out.
