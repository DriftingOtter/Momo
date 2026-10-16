# v0.9 M0 — public API inventory and consistency

82 names in `tensorlang.__all__` (v0.8.0). Coverage from application use:

## Findings by area

- **Construction**: `T(...)` accepts scalars/sequences/mappings/arrays/
  frames; rank-0 stays a collectable scalar. `P("X")` placeholder stable.
- **Operators**: definitions, `>>` composition, cross-file reuse all
  worked. `Operator.__call__` rejects Operator args (`TL0499`) — by
  design, but forces flat composites (F-A1).
- **AD**: `value_and_grad` multi-`wrt` verified vs finite differences
  (9 digits). Unfitted-estimator differentiation deferred (documented).
- **Fitting**: `standardize`/`fit` reuse correct. Unfitted application
  defers its error to `collect()` with generic `TL0499` (B: should be
  raisable earlier/coded precisely).
- **Structured**: scans/joins/groups/counts/`to_dense`/predicates all
  worked; grouped-sum float order varies at 1e-12 (parallelism noise,
  benign). Aggregated columns keep backend names (`amount`, `count` —
  B: document explicitly).
- **Comparisons**: `==` diverges dense-vs-structured by design
  (NaN/None); `is_null`/`not_null` correct; `["..."]` required for
  method-named fields (B: collision error is a raw `TypeError` — should
  suggest indexing).
- **Lazy**: graphs lazy until `.collect()`; separate collects are NOT
  row-aligned (F-B3); `&` of masks misaligns silently (F7, see below).
- **Chunked**: scan/transform/reduce/write/admission/diagnostics all
  worked at 512 MB.
- **Modules**: sibling/dotted/alias/re-export/namespace/caching all
  worked single-spelling; dual-spelling fails (F-A2); `extra=` rules
  held (`TL0203` never triggered — no collisions needed).
- **Errors**: coded, located, chained; `diagnose`/hints effective.
- **Leaked internals**: `check_source()` exposes `trusted_names` /
  `injected_names` (B: internal machinery in a public signature).

## Stability classes (preliminary)

- **A candidate**: `T`, `P`, operator defs/`>>`/application, `collect`,
  scans, joins, groups, `to_dense` policies, `standardize`/`fit`,
  `grad`/`value_and_grad`, `run_file`/`run_source`, CLI run/check/
  diagnose, error codes `TL03xx`–`TL05xx` semantics.
- **B needs clarification**: unfitted-apply timing/code; aggregated
  column naming; `check_source` extra params; grouped-sum float-order
  noise; intercept-penalty convention (app chose sklearn's).
- **C needs correction**: F-B4 (`&` mask misalignment — silent wrong
  results); F-A2 (dual-spelling failure + failure-path poisoning).
- **D experimental**: `trusted_names`/`injected_names` params, REPL
  session retention semantics, chunked-AD/diff deferrals (documented).
