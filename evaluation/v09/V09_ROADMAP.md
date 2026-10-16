# v0.9 roadmap (evidence-ranked, post-M0)

M0 answers: the engine is sound and apps ship, but two correctness
defects (F-B4, F-A2) plus documentation/consistency gaps block any
stability commitment.

## M1 — correctness (required)

- F-B4: align `&`-combined masks or reject unalignable combinations
  with a coded diagnostic. Acceptance: 6-row reproducer stable at 2;
  App B big-ticket == 1774 across runs.
- F-A2: per-spelling module independence (validated approach:
  `(path, name)` cache keys); no post-failure poisoning; regression
  tests un-`xfail`ed. Acceptance: App A/B probe patterns green;
  failed session followed by fresh `run_file` green in-process.
- F-D1: precise unfitted-application error.

## M2 — distribution and workflow (small)

- Document single-spelling layouts, chained re-exports,
  single-materialization, sequential filters, `["..."]` indexing,
  aggregated naming, `extra=` rules, subprocess test isolation.
- Minimal `pyproject` dependency convention (evaluated adequate).
- F-B1 coded hint; F-D2 signature cleanup; F-B2 naming docs.

## M3 — compatibility and stabilization (small)

- Formalize A/B/C/D classifications from M0_COMPATIBILITY.md.
- Deprecation/versioning policy proposal; diagnostic-code stability
  statement; dependency-floor policy.
- Full CI + packaging + app regressions.

## Deferred

New operators/grammar/backends/packages/hot-reload/global caches —
none required by the evidence. No v1.0 contract yet.

## Milestone acceptance

M1 done when the F-B4 reproducer is exact, dual-spelling works or is
rejected loudly without poisoning, and 549 + app suites stay green.

## Status after M1

- F-B4 fixed (fused same-domain `and`/`or`, `TL0512` otherwise);
  F-A2 fixed (`(path, name)` cache identity, no poisoning);
  unfitted `TL0419` at application time. Suite 561 core + 15 app
  green; benchmarks without regression. Proceed to M2 (workflow).

## Status after M1 (revised scope per review)

- F-B4 fixed incl. review gate (unanalyzable structured `&` → `TL0512`;
  dense/chunked paths preserved; Series contract preserved).
- F-A2 fixed (`(path, name)` cache identity; no poisoning).
- Unfitted `TL0419` at application (incl. nested pipelines).
- Suite 563 core + 15 app green; benchmarks without regression.
- Proceed to M2 (project workflow) per revised priorities.

## Status after M2

- `tl` project manager implemented per plan (uv-backed, no solver);
  acceptance flow green incl. copy-reproduce; Apps A/B/C migrated
  with identical numerics; 27 new tests; 590 core green.
- Proceed to M3 (compatibility stabilization) per revised priorities.

## Status after M3

- Contracts (API inventory, `tl` semantics, freshness, forwarding,
  isolation) documented in `docs/API_STABILITY.md` + `docs/PROJECTS.md`.
- 7 M3 tests; managed app re-runs green; no open P0/P1.
- Recommendation: v0.9.0 release validation.
