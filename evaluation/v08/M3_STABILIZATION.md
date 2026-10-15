# v0.8 M3 — developer-experience stabilization

## 1. `extra=` precedence (root cause + fix)

Root cause: `run_source` did `namespace.update(extra)` before `exec`,
so any module-level binding (`=` or `:=`) silently overwrote injected
values after they were trusted for M1 seeding. No override contract
existed.

Deterministic policy (implemented, tested in
`tests/test_injection.py`):

- `=` over injected: allowed, module wins (assignment is explicit
  override intent; the only intentional-override path).
- `:=` over injected: `TL0203` conflict, raised at compile time with
  zero execution (sentinel-verified).
- Dunder/interpreter names rejected at injection (`TL0203`).
- Imported `.tl` modules never inherit `extra` (fresh namespaces).
- Smuggling blocked: foreign, aliased, and operator-valued injected
  callables under `:=` all stay `TL0305`; only identity-verified
  TensorLang public objects seed trust.
- REPL per-push rebinding unchanged (no `extra` there).

## 2. NaN/null contract (no semantic change)

Measured on NumPy 2.5.4 / Polars 2.0.0; pinned in
`tests/test_missing_values.py`, documented in `docs/COMPARISONS.md`:

- Dense `NaN == NaN` is False; structured is True; null propagates.
- `== None` is `TL0499` everywhere.
- `is_null(NaN)` diverges by design (dense True, structured False):
  it reports the representation's missing bit.
- Rule: `not_null()`/`is_null()` predicates only; `to_dense` policies
  explicit (`nan` promotes, `error` → `TL0418`).

## 3. Import diagnostics audit

- Chained format verified end-to-end (missing/symbol/cycle/init).
- `diagnose` hints added for `TL0310`/`TL0312`; `TL0203` registered.
- Fixed during M3: dotted imports depended on `run_file`'s `sys.path`
  insertion, so `diagnose`/REPL/`run_source` failed them (`TL0310` for
  the parent). The loader now synthesizes inert package shells for
  bare directories — no `sys.path` cooperation needed, `__init__`
  never executed, sibling `.tl` file takes precedence.

## 4. Loader integration audit

- `sys.path`: pushed/popped symmetrically per module (`test: restored`).
- `sys.meta_path`: exactly one finder, idempotent install.
- `sys.modules`: owned `.tl` modules + namespace parents swept;
  third-party imports (scipy) preserved by identity across sessions.
- Nested `run_file` joins the active session (single init).
- REPL `:reset` drops the module cache.
- **Concurrency**: simultaneous independent sessions in threads are
  rejected fail-fast (`TL0315` session-ownership guard — exactly one
  proceeds, never silent contamination). Parallelism must use
  subprocesses. Documented in spec + `MODULES.md`.

## 5. Application verification

Driver 0.6311 / 94976 rows; native `.tl` 0.6311; NumPy reference
0.6311. CLI matrix (`run/check/diagnose/graph/explain/profile/doctor`)
verified against `main_native.tl` from an unrelated CWD; preflight
refusal correctly attributes the entry's `write_parquet`.

## 6. Performance

No engine changes. Import baselines hold (~2 ms/file cold,
0.1 ms cached). Full suite + app green (see below).

## 7. Limitations remaining

- Operator dunders on foreign objects, comprehension iteration, lambda
  defaults (documented in `EFFECTS.md`).
- Dynamic-`__import__` cycles, `.tl` imports from plain Python,
  function-level `.tl` imports resolving against the caller.
- `=`-over-`extra` is silent-but-documented (override intent).

## Release-readiness recommendation

Ship v0.8.0 after release validation: six-cell CI matrix,
installed-distribution loading, source-path independence, session
lifetime, transitive enforcement, and the M0 numerical equivalence
above — all green locally. No v1.0 compatibility contract yet.
