# TensorLang changelog

## v1.0.0

- Corrupt manifests reported as invalid (never as a missing project).
- `tl add` accepts local `.whl` dependencies via uv path sources.
- uv failures preserve diagnostic context (credentials redacted).
- Missing/unreadable `.tl` inputs produce structured `TL0103`
  diagnostics (no raw tracebacks).
- `to_dense()` contract states the post-upstream admission boundary;
  persistence docs state the recoverable multi-step replacement.
- `check_source()` stable 2-arg form; plumbing deprecated.
- Join row ordering documented as backend-defined with safe patterns.
- Proposed v1.0 contract: `docs/API_STABILITY.md` stable subset (66),
  `docs/COMPATIBILITY.md` policy (pending release approval).
- M3: `docs/GETTING_STARTED.md` (validated end-to-end in a clean
  environment), `docs/TROUBLESHOOTING.md`, library-consumer guide;
  M2 skipped (standard packaging + `tl add <wheel>` sufficient).
- M4: dependency floors corrected from evidence (`polars>=1.21` for
  rolling `min_samples`; `scipy>=1.11` for Python 3.12 wheels;
  `zarr>=3.0.8` excluding yanked data-loss releases); portability
  lint (`scripts/check_portability.py`, CI-gated, allowlisted
  fixtures); machine-readable API inventory
  (`tests/test_public_api.py`); release validation script
  (`scripts/release_check.py`, verify-only); example tolerance
  aligned with the numerical policy.

## v0.9.0

- Structured mask composition fused per row domain; incompatible
  domains rejected (`TL0512`).
- Canonical per-spelling module identity; failure cleanup without
  cross-session poisoning.
- Unfitted-estimator application rejected at application time
  (`TL0419`).
- `tl` project manager (uv-backed): `new`/`init`/`add`/`remove`/
  `sync`/`run`/`check`/`test`/`doctor`; `pyproject.toml` manifest
  with `[tool.tensorlang]` entry; additive `tensorlang run` argument
  forwarding.

## v0.8.0 — modular applications

- Native file-based `.tl` imports (`import m`, `from m import x`,
  aliases, dotted paths): resolved relative to the importing file,
  Python-first precedence, per-session caching, isolated namespaces,
  static circular-import rejection (`TL0312`), chained diagnostics
  (`TL0310`–`TL0314`).
- Fail-closed declarative effect enforcement: unclassified Python calls
  under `:=` rejected as `TL0305` before execution (no grammar change).
- Deterministic `extra=` injection precedence: `:=` over an injected
  binding is `TL0203`; `=` rebinding keeps module precedence; dunder
  injection rejected; injected modules never inherit `extra`.
- Documented NaN-vs-null comparison contract across dense/structured
  representations (`docs/COMPARISONS.md`); no semantic changes.
- CLI transitive effect preflight for `graph`/`explain`/`profile`;
  REPL module sessions with `:reset` cache clearing.
- No new operators, backends, or execution-engine changes.

## v0.7.0 — unified analytical pipelines

- `frame.to_dense(columns=[...], nulls="error"|"nan")`: lazy,
  schema-aware conversion to dense tensors (no data read at
  construction; row count unknown until execution).
- Deterministic dtype promotion (nullable-int to float64, explicit
  Boolean-with-null cast) validated against NumPy conversion.
- Memory admission at the execution boundary (`TL1020`) with
  documented post-upstream guarantee scope; unknown cardinality
  labeled, never zeroed.
- Grouped `count`: row counts (`frame.count(by)`) and non-null field
  counts (`series.count(by)`), native Polars parity, u32 preserved.
- `fit()` accepts lazy dense inputs; structured Lazy inputs get a
  coded `TL0418` diagnostic recommending the bridge.
- Differentiation through conversion deferred with a diagnostic.
- `abstract()`/`explain()`/`profile()` expose bridge semantics,
  estimates, and observed timings.
- 463 tests; evaluation reports under `evaluation/`.

## v0.6.0 — workload-driven numerics and natural tensor ops

- Dense execution: sharing-preserving parameter remapping, scoped
  evaluation memo with use-count eviction, structural deduplication
  with taint isolation (no language or semantics changes).
- Reverse-mode AD: pop-after-propagate cotangent release with
  complete forward tapes; `last_released_cotangents` introspection.
- Axis-aware reductions (`axis=`/`keepdims`) across Tensor/Lazy/
  Operator, executor, AD rules, and chunked lowering; `TL0511`.
- Operator `.T` transpose; field-confusion diagnostics suggest the
  operator function (`X["T"]` preserved for fields).
- F8 source-baseline fix included (as released in v0.5.4).
- 428 tests; benchmarks at native parity or better on ML/energy.

## v0.5.3 — out-of-core chunked execution

Chunking is a physical execution property, not a new operator kind:
all existing syntax, operators, and in-memory semantics are unchanged.

- **M1 — chunked foundation**: `T.scan_zarr` lazy persistent sources,
  Dask-backed lowering (elementwise math, shape ops, whole-array
  reductions), streaming `write_zarr`, chunk metadata in `abstract()`/
  `explain()`/`profile()`, `TL1001`–`TL1003`.
- **M2 — chunk-aware planning**: operation capability classes, union
  subdivision alignment with recorded movement, task/block policy gates,
  bounded rank-2 matmul (unchunked contract axis), reshape/transpose
  planning, `TL1010`–`TL1013`.
- **M3 — memory budgeting**: `set_chunk_policy(memory_budget_bytes=...,
  max_workers=...)`, per-task resource models, admit /
  admit-reduced-concurrency / reject decisions per purpose (collect vs
  stream), per-call worker caps, baseline-vs-estimate monitoring,
  outputs larger than the budget via streaming, `TL1020`–`TL1024`.
- **M4 — execution reliability**: per-run lifecycle
  (planned/admitted/running/committing/completed/failed/cancelled),
  staged publication with atomic local rename (recoverable — not
  continuously atomic — overwrites), metadata-only output validation,
  source-consistency checks (`TL1030`, report-once), interruption
  handling (`TL1032`), stale-staging recovery, `TL1031`/`TL1033`–`TL1035`,
  `write_zarr` classified effectful.
- **M5 — release validation**: recovery state table (`recover()`),
  hook isolation (`injected_failures`), dependency floors
  (`dask>=2025.1`, `zarr>=3.0.1`; zarr 3.0.0 excluded — upstream yank),
  wheel+sdist with clean-install smoke tests, merged benchmark results,
  capability matrix, recovery guide, CI.

## Guarantees (and their limits)

- Estimates are planning heuristics; only the OS enforces hard process
  containment. Budgets exclude baseline RSS, fragmentation, running
  native tasks.
- Overwrites are recoverable staged replacements, not continuously
  atomic; power-loss durability is not claimed.
- Source checks are metadata-only and best-effort.
- Windows is Tier-2 (best-effort, untested); remote stores are direct
  and explicitly non-transactional.

## Upgrade notes

- `dask` and `zarr` are now required runtime dependencies.
- `write_zarr()` defaults to staged publication on local paths
  (same arrays out; ~1–7% throughput overhead, 2× transient disk).
- `write_zarr` under `:=` is now rejected as effectful (previously
  bypassed I/O protection).
- `tensorlang bench` requires a source checkout (benchmarks are not
  shipped in the wheel).
