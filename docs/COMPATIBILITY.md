# TensorLang compatibility and deprecation policy (v1.0 proposal)

**Status: proposed (M4-finalized wording).** This policy is the
intended v1.0 contract. It is not active until the v1.0 release is
approved and tagged. Pre-1.0 releases (including v0.9.0) carry no
long-term guarantee beyond the migration notes in
`evaluation/v1/M0_COMPATIBILITY.md`.

## 1. Semantic versioning

- Major (`X.0.0`): may break promised contracts listed in §2.
- Minor (`1.x.0`): additive public APIs, new diagnostics codes,
  broader platform/dependency support; no promised-contract breakage.
- Patch (`1.x.y`): correctness fixes preserving promised semantics;
  behavioral corrections ship with the diagnostic code and a
  migration note.
- Pre-1.0 (`0.y.z`): no compatibility promise; intentional
  correctness fixes (e.g. `TL0512` fail-closed alignment,
  `TL0419` apply-time rejection, dual-spelling module independence)
  may change behavior with migration notes.

## 2. Promised contracts (breaking change ⇒ major)

- `.tl` syntax and declaration semantics: `:=`/`=`, `{X}` scoping,
  operator definition/application/`>>`, explicit `.inplace()`,
  resolving import spellings, `TL0305` fail-closed verdicts.
- Operator and execution semantics: broadcast rules, `collect()`
  boundary + snapshot/ownership, `to_dense(nulls=)` policies,
  join-key contract, no-order-guarantee for relational results,
  `fit`-returns-new + `TL0419`, first-order AD rule set + `TL08xx`
  rejections, `TL0512` fail-closed verdict.
- Stable Python signatures in `docs/API_STABILITY.md` (parameter
  names, defaults, arity, return kinds) and `run_source` /
  `run_file` / `diagnose_*` result shapes. `check_source(source,
  filename)` is stable; its `trusted_names=`/`injected_names=`
  plumbing is deprecated (warns) and excluded from the promise.
- Diagnostic **code identities and triggering conditions**
  (`TLxxxx`); message wording, hint text, and traceback formatting
  are explicitly not frozen.
- Manifest: `[project]` single truth, additive-only evolution,
  `uv.lock` honored by `tl sync --locked`; `tl
  new/init/add/remove/sync/run/check/test/doctor` exit-status
  contract. `uv.lock` internals and `[tool.uv.sources]` encodings
  are uv's domain, not TensorLang's.
- Import behavior: file-relative resolution, Python-first
  precedence, per-session `(path, name)` identity, `TL0310`–`TL0315`
  conditions.
- Persistence: no silent store-format change; any format change
  ships with round-trip tests and a migration tool.

## 3. Internal details (may change in minor/patch)

IR node shapes, backend query plans and lowering choices, Polars/Dask
version floors within declared ranges, task counts, RSS/admission
estimates, message wording, REPL history, benchmark harness,
chunked-AD deferral set, performance characteristics, temporary
filenames.

## 4. Numerical behavior

Documented mathematical outcomes hold within stated tolerances:
exact for integer/shape paths; `allclose`-style agreement for
floating-point and AD (reference comparisons use `rtol=1e-4`,
`atol=1e-6`; application parities per `evaluation/v09`
1e-9/1e-12/16-digit references). Bitwise identity across BLAS
implementations, architectures, or dependency versions is not
promised. Grouped floating-point last-ulp ordering noise (~1e-12)
is documented, not frozen.

## 5. Correctness-driven behavior changes

A demonstrated silent-wrong behavior may be corrected in a minor or
patch release even when observable behavior changes, provided the
correction ships with (a) a stable diagnostic code, (b) a migration
note, and (c) a major-version reservation if the old behavior was
ever promised in §2. v0.9 precedents: `TL0512`, `TL0419`,
dual-spelling independence.

## 6. Deprecation process

1. Announce in minor release notes + `DeprecationWarning` at use
   (example: `check_source` plumbing in M1).
2. Document the migration path and the removal target (no earlier
   than the next major).
3. Keep the old behavior working through the deprecation window;
   never silently change semantics under a deprecated flag.

## 7. Platforms and dependencies

Python support follows the CI matrix (currently 3.12–3.14 on
Ubuntu/macOS); raising the floor requires a minor release with
notice. Dependency floors are pinned per release and validated
against minimum versions at release-candidate time (M4 evidence:
`polars>=1.21` for rolling `min_samples`, `scipy>=1.11` for
Python 3.12 wheels, `zarr>=3.0.8` excluding yanked data-loss
releases). Windows is Tier-2 (best-effort, no CI) until evidence
says otherwise. `tl` requires `uv` on `PATH` (no pip/global
fallback); authenticated package URLs are redacted in diagnostics.

## 8. Activation

This policy takes effect at the v1.0.0 tag. Until then it is a
review draft guiding M1–M4 stabilization work.
