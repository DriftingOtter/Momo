# v0.9 M1 — correctness report

## F-B4 root cause

`and`/`or` had no frame-level lowering (`'BinOp' has no lazy-frame
lowering`), so each side executed as an independent query (joins and
group-bys yield backend-defined row orders) and combined positionally
(`pl.concat(how="horizontal")` / Series `&` check length only).
Fix (`runtime/polars_plan.py`, `runtime/executor.py`):

- `_row_roots` extracts frame-root IR per side (frozen-dataclass
  structural equality; opaque values unify by identity only).
- One distinct root → fuse into a single Polars query
  (`SELECT [(expr & expr)]`); column naming preserves the Series
  contract via `collect_lowered` unwrap.
- Multiple distinct roots → hard `TL0512` (propagates through
  `_try_lowered`, which otherwise swallows everything).
- Unanalyzable shapes → historical path (documented residual).
- `GetItem(obj, Const(str))` added as a row expression (also fixes
  filter fusion for `["..."]` field access).
- Top-level `BinOp(and/or)` evaluation tries lowering first (only
  `and`/`or`, so dense hot paths are untouched — verified 0.1 ms
  parity on a matmul microbench).

Verification: 6-row reproducer exact records, 12/12 runs count 1;
`|`/`~`/nesting/reversed/reordered/grouped/scalar-bool/lazy cases;
App B big-ticket stable at true 1774 (was 1715–1721); model parity
with sklearn unchanged (1e-12). 10 new tests.

## F-A2 root cause and fix

The session cache was keyed by path only, so a second spelling
aliased the first spelling's module object while the machinery built
a fresh empty module around it — `IMPORT_FROM` then failed on the
empty object, which also lingered in `sys.modules` and poisoned later
sessions. Fix (`frontend/modules.py`): cache key `(path, name)` —
each spelling initializes independently (CPython semantics), same
spelling reuses. The expected-failure pin is now a passing test;
dual probe + leak probe + full suite green.

## Unfitted diagnostic

`FittableOperator.__call__` and `PipelineOperator.__call__` (with
nested-pipeline walk) reject unfitted application at application time
with `TL0419` (estimator named, `.fit` directed). `fit` paths use the
executor directly and are unaffected. Existing tests (which expected
`TLTypeError` at collect) still pass; new test pins `TL0419`.

## Compatibility notes

- Previously silent cross-domain `&` now errors (`TL0512`) — intended
  correctness fix.
- Dual spellings now yield independent objects (was: failure) —
  matches CPython; uid scoping unchanged.
- Unfitted errors surface earlier with a precise code (was: generic
  `TL0499` at collect).
- No grammar/ontology/engine changes; 561 core tests green
  (549 + 8 mask + 1 unfitted + dual un-xfailed + …), 15 app tests
  green, benchmarks ran without regression.

## Residuals

Frame-vs-frame arithmetic/comparison stays positional by design;
`==` of masks unaligned is not fused; unanalyzable `and`/`or`
shapes keep the historical path; dynamic-`__import__` cycles rely on
the runtime backstop.

## Review-gate evidence (unanalyzable shapes)

Per review: Where-built masks over differently-ordered frames (row
analysis impossible) now raise `TL0512` instead of zipping — verified.
Dense-only `&` keeps positional semantics; chunked combinations keep
the chunked path (explicit bypasses, both tested). Scalar reductions
contribute no domain. Structural root equality is sound here because
roots are either identical objects (shared substitution) or
value-equal scans of file-addressed sources; materialized snapshots
unify by object/value identity only.

## Compatibility notes (addendum)

- `GetItem(obj, Const(str))` is now a plannable row expression (also
  fixes filter fusion for `["..."]` field access).
- Fused single-column boolean results unwrap to Series, preserving
  the historical `collect()` contract.
- Previously silent cross-domain `&` now errors (`TL0512`); dual
  spellings initialize independently; unfitted errors surface at
  application (`TL0419`).
