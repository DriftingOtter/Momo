# M0 — Modules & interop

No code changes.

## 1. Native module system (v0.8/v0.9 loader)

Verified in `frontend/modules.py`, `docs/MODULES.md`, `test_modules.py`, `test_loader_audit.py`, `test_injection.py`:

- Local/dotted imports, file-relative (never sys.path/CWD), Python-first (finder last), inert namespace parents (`__init__` never runs), relative-import `TL0314`, `TL0310/311` chains, static pre-scan cycle `TL0312` pre-execution — pass.
- `(path, name)` identity: dual spellings initialize independently (CPython-like); per-session cache + cross-session isolation; failure never cached + retry works (sys.modules entry lingers until session close sweeps — documented level holds) — pass.
- REPL `:reset` drops session; installed-path independence (entry sys.path push/pop; module-dir push/pop); third-party imports survive teardown by design — pass.
- TL0315: overlapping cross-thread sessions fail fast both directions (exactly-one-winner, never silent share); same-thread nesting joins; sequential fine; subprocess parallelism documented — pass.

## 2. Python interop

TL→Python: `run_file/run_source` isolated sessions; `extra=` entry-only (`=` keeps module precedence, `:=` over injected → `TL0203`, dunders rejected, injected callables untrusted under `:=` unless identity-verified); operator retrieval/application with UID scoping; `TL0419` at apply; repeat re-initializes; coded chains + `diagnose_*` — pass.
Ownership gap: `T(ext)` copies once; `to_numpy/to_polars` copy/clone; but repeated `Lazy.collect()` returns the same cached object — caller mutation corrupts cache (`.release()` escape). Spec §7/docstrings don't state aliasing — M1 wording fix.
Python→TL: arrays, scipy (LAPACK linalg; imports preserved by design), Polars (lazy single-query + fallback preserving results) — pass. Runtime calls effectful by design (`T.read` IO-provenance `TL0304` under `:=`; `fit/write/inplace/collect` execute; `scan_*` declarative; `to_dense` bridge `TL0418/TL1020`) — pass, never described as pure.
Stale eval note: `evaluation/M0_INTEROP.md:32-35` P1 ("Lazy not callable, no code/guidance") outdated — current raises coded `TL0418` + `to_dense(columns=...)` suggestion — M1 doc refresh.

## 3. M0 verdict

Loader + interop contracts hold incl. TL0315; no concurrent-loader work in M0. M1: collect-aliasing + outdated-note fixes only.
