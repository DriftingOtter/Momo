# M0 — Runtime & persistence

Spec refs are `LANGUAGE_SPEC.md` v0.7.0 (§§7–9, 13, 16, 18). No code changes.

## 1. Lazy execution & caching

- Construction lazy (`core/tensor.py:299-303`, `core/operator.py:266-282`); triggers explicit only: `collect/inplace/write_*/profile/fit/_material` — pass. `explain/describe_write/abstract` never execute — pass.
- `.collect()` per-object cache (repeat returns same cached object; `.release()` drops) — matches spec §7 "may cache". Wording gap: aliasing consequence unstated (mutating a collected array corrupts cache) — M1 doc fix.
- Cross-boundary recompute by design: memo scoped to one `evaluate()` (`runtime/executor.py:295-327`); measured 84 ms fresh vs 31 ms materialized — intended, document.
- Snapshots: read-only views (dense) / clones (frames) at construction; `Lazy` immutable post-construction; `apply_inplace` evaluates-then-touches + restores on failure — pass.
- Fitted exec re-evaluates `arg` each run; `fit()` returns new, receiver unchanged — pass.
- Module-init vs result caching separate (`LoaderSession.modules (path,name)` vs per-`Lazy` cache; `docs/MODULES.md:40-42`) — pass.
- Source invalidation: `verify_sources` (spec+baseline+mtime) → `TL1030` + resync — pass.
- Process-global state: scoped caches are `ContextVar`s (clean). Globals that exist: `dask_backend.describe` LRU-128 (cleared on write/publish + TL1030 resync — spec undisclosed, rely on verify); `_SOURCE_BASELINES` bounded dict; `_TEST_HOOKS`/`LAST_*` observation state (gated); loader `_SESSION_STACK`/meta_path hooks/`_TRUSTED_IDS` (documented). No process-global result cache — pass with wording fix for describe-cache.

## 2. Memory / out-of-core

- Dask/Zarr paths: capability classes, union-subdivision (split-only), tree reductions w/ partial-byte modeling, explicit rechunk, matmul contract gate; unsupported → coded `TL1001/1010-1013`, never silent materialize — pass.
- Budget contract correct everywhere checked: budget = estimated incremental working memory only (RSS baseline/BLAS/frag excluded; "only the OS enforces hard containment"); outcomes admit / admit-reduced-concurrency (workers→1 only) / reject `TL1020/21/23/24`; RSS `TL1022` observational only; completed runs never rebranded; interruption `TL1032` honest — pass. Do NOT claim budget=hard cap (already correct).
- `to_dense` wording gap: `bridge.evaluate()` materializes upstream first, then admits conversion (3× heuristic); `bridge.describe()` states scope honestly but spec §9 ("conversion is memory-admitted") reads as whole-pipeline bound — M1 spec fix.
- Streaming/recovery/retry match `docs/RECOVERY.md` + tests: staged-default vs direct-explicit (partial-output warning), precheck `TL1002`, disk-vs-RAM separate, self-overwrite refused, census+metadata validation, backup round-trip overwrite, `TL1031/1034`, concurrent-winner `TL1002`, TL1030 resync retry, `TL1035` manual path — pass.
- Persistence scope gaps: spec §16 M4 "atomic rename" needs `publish_staged` qualifier (each rename POSIX-atomic; backup overwrite = recoverable multi-step, not continuously atomic); census detects missing chunks, not garbage bytes (correctly unclaimed) — M1 wording.

## 3. M0 verdict

Guarantees match reality except four wording/scope fixes (to_dense scope, staged-atomic qualifier, describe-cache disclosure, collect-aliasing). No new algorithms; no planner changes.
