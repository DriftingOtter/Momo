# v0.8 roadmap (evidence-ranked, post-M0; M1+M2 status appended)

M0 answers to the three decisive questions:

M0 answers to the three decisive questions:

1. **Cross-file operator reuse**: SAFE — uid-scoped params, correct
   composition, no snapshot/lookup hazards. A loader does not risk the
   scoping model.
2. **Python as integration layer**: WORKS — `run_file` namespaces +
   first-class operators compose in plain Python; no compiler-internals
   knowledge required. Do not reproduce Python packaging inside TL.
3. **Test/diagnose/distribute independently**: YES for test and
   diagnose (plain pytest, coded errors, installed CLI); PARTIAL for
   distribute (ad-hoc file shipping works, no convention).

## Proposed milestones

### P0 — Close the effect-gate bypass (correctness)
`x := evil.grab()` executing I/O is the only correctness hole found.
Smallest fix preserving semantics: runtime provenance tracking or an
allowlist for `:=` right-hand sides, plus a regression test. No syntax
change. Do first, before any loader widens the attack surface.

### P1 — Narrow `.tl` loader (the v0.8 feature)
File-based loader only: `import` a `.tl` file by path, executing its
top-level declarations once into a cached, isolated namespace
(isolation semantics already verified). Explicitly out of scope:
relative-import packages, circular imports (reject with a coded
diagnostic), parameterization (`extra=` stays a Python-side mechanism
with documented shadowing). Include chained "imported from" error
attribution from day one (F6). Acceptance: the evaluation app's driver
shrinks to stage imports + execution, with identical numerics.

### P2 — Semantics documentation + lints (cheap, high-value)
Document `==`/NaN divergence (F4), join/group order (F5), `extra=`
shadowing (F2), and the `not_null()`/in-graph-derivation patterns.
Add `check`-time lints where cheap (e.g. warn on `==`-self float
comparison in `.tl`). No engine changes.

### P3 — Packaging convention + plan summary (deferrable)
Document shipping `.tl` stages + driver (entry points,
installed-data layout); optionally add a driver-level multi-binding
`explain` summary (F8). Defer if P0/P1 fill the cycle.

## Explicitly deferred

- Full Python-style package system (relative imports, circular
  support, import hooks) — evidence shows a narrow loader suffices.
- Custom package manager, compiler backend, GPU/distributed, chunked
  AD, partitioned contractions, new operator families, global lazy
  caching, arbitrary Python↔operator auto-conversion — all out of scope
  per the M0 brief; none blocked the application.
- v1.0 compatibility contract — premature; instead, P2 documentation
  marks `run_file`/`run_source`/`T`/operators/`to_dense`/diagnostics as
  "stable-candidate" and `extra=` injection as "evolving".

## Acceptance for v0.8

- P0 regression test green; P1 loader with cache + isolation +
  chained errors + circular rejection; app driver re-expressed with
  loader imports producing identical 0.6311 coefficient; full 463-test
  suite green; evaluation docs updated.

## Status after M1+M2

- M1 (P0) done: fail-closed `TL0305`, 20 tests, spec + `docs/EFFECTS.md`.
  Suite then 483 green.
- M2 (P1) done: narrow loader (`docs/MODULES.md`), per-session cache,
  `TL0310`–`TL0314`, transitive CLI preflight, 31 tests, native app at
  identical 0.6311. Suite 514 green. See `M2_LOADER.md`.
- Remaining for a small M3: `extra=` shadowing semantics, NaN
  comparison documentation, import-error polish. No package manager.
  Then release validation.

## Status after M3

- M3 done: `TL0203` injection precedence, NaN/null contract
  (`docs/COMPARISONS.md`, no semantic change), dotted-import
  `sys.path` independence, loader hygiene audit, documented
  thread-unsafety. Suite 544+ green (core), 11 app green.
- App equivalence re-verified: 0.6311 × 3 implementations, 94976 rows.
- Recommendation: proceed to v0.8.0 release validation (CI matrix,
  packaging, install checks). No v1.0 contract yet.
