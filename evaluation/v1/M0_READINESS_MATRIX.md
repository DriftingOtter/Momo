# M0 — Readiness matrix (consolidated)

Baseline: v0.9.0 `bbbef00`; macOS arm64 8 GB; py3.14.7; numpy 2.5.4 / scipy 1.18.1 / polars 2.0.0 / dask 2026.8.0 / zarr 3.4.1 / uv 0.9.26.
Suites: 563 passed (excl. `test_tl`) + 33/33 `test_tl` executed = 596 core; 15/15 apps; 10/10 examples. Clean-venv 0.9.0 install verified (`tensorlang` + `tl`; falling-body project green).

## 1. Category readiness (12 areas)

| # | Area | Evidence | Readiness |
|---|------|----------|-----------|
| 1 | Syntax/semantics | `:=`/`=`/`{X}`/`>>`/apply verified; 8 classified ambiguities (loop, trust-flat, attr rule, literal-exempt, plain-values, open-set, aliasing, dunder) | Provisional — needs M1 wording |
| 2 | Public APIs | 82 exports reconcile; narrow-A candidate list; B/C risk register; D excluded | Provisional — needs freeze list |
| 3 | Numerical | Exact arith; FD sweep; AD/VJP; TL0419/TL0512; AppB 1774 + rel 1e-9 | Ready (lock-in tests) |
| 4 | Data reprs | Kleene/NaN-vs-null/`to_dense` TL0418/grouped-nulls verified | Ready (keep `is_null` guidance) |
| 5 | Exec/memory | Lazy/cache/snapshot verified; budget≠hard-cap correct; to_dense-scope + atomic-rename + describe-cache wording gaps | Provisional — 3 wording fixes |
| 6 | Modules | (path,name) identity; TL0310-315; TL0315 fail-fast; isolation | Ready |
| 7 | Interop | run_*/extra=/UID/TL0419 verified; collect-aliasing + stale-note gaps | Provisional — 2 doc fixes |
| 8 | Projects | Full `tl` flow + locked strict + isolation verified; 33/33 executed | Provisional — P0/P1 fixes open |
| 9 | Reproducibility | pyproject truth; uv.lock honored; stale-warn; doctor read-only; copy-reproduce byte-identical | Ready (code); CLI disk bug open |
| 10 | Diagnostics | Codes+chains+hints+exits verified; 1 uncoded escape (nonexistent file → raw traceback exit 2) | Provisional — 1 fix |
| 11 | Distribution | Std wheel installs + imports + applies `.tl` libs; `tl add` path-spec blocked; no build/publish (by design) | Gap — conditional M2 |
| 12 | Compatibility | 10/10 examples; 15/15 apps; no accidental regressions; migration notes drafted; policy proposed (not enacted) | Ready (adopt policy) |

## 2. Findings by severity

### P0 — must fix before any freeze
- Corrupt `pyproject.toml` via CLI misreported as "no project found" (`find_root` swallows `TOMLDecodeError`; `load_manifest` unreachable). Wrong error + wrong remediation on a supported op.

### P1 — release blockers
- `tl add` rejects local paths/wheels (index-only `_PKG_RE`) — no `tl`-native lib-install path (workaround: `uv pip`/hand-edit).
- Uv failures truncated to last stderr line (offline shows "cache.") — context-poor on supported ops; no `--verbose`.
- `tensorlang run /nonexistent.tl` → exit 2 raw `FileNotFoundError` traceback, not a coded `TensorLangError`.
- Spec §9 `to_dense`: reads as whole-pipeline memory bound; code gates conversion only (upstream materializes first).
- Spec §16 M4 "atomic rename": backup-overwrite is recoverable multi-step, not continuously atomic (code docstring correct; spec overclaims).
- `check_source(..., trusted_names, injected_names)` plumbing in public signature — freeze-as-D or deprecate before v1.0.
- Join row order backend-defined (varies run-to-run); appB F-B3 materialize-together workaround load-bearing — v1.0 must not promise stable order.

### P2 — deferrable/document
Loop/`:=` static semantics; file-flat trust; attr rule; literal-receiver exemption; plain-value derivations; open declarative set; `x = f` aliasing; dunder/format limits (disclosed); aggregate names; grouped-float ~1e-12; NaN contract; REPL retention; `extra=` override; `to_dense` default; stale `M0_INTEROP` P1 note; `collect()` aliasing doc; describe-cache disclosure; `new`vs`init` unicode rules; venv-without-pip; `diagnose` JSON schema; `needs_uv` dead marker; `[tool.uv.sources]` URI escaping; manual release steps (bump/changelog/tag); Windows Tier-2 correctly labeled (no CI — do not claim parity).

## 3. v1.0 entry criteria status

- [~] Semantics documented (8 ambiguities worded, not fixed) → M1
- [~] Stable-candidate APIs reviewed (narrow-A proposed, not frozen) → M1
- [x] Correctness defects: no silent-math/mutation issues found
- [x] Intentional incompatibilities documented (migration notes)
- [~] Execution guarantees tested (3 wording fixes) → M1
- [x] Project creation/reproduction verified (clean-env green)
- [x] Module/import behavior consistent
- [~] Dependency compat specified (floors declared; minimum-version install untested) → M4
- [~] Diagnostics reliable (1 escape) → M1
- [~] Release reproducible (manual steps; no script) → M4
- [ ] No open P0/P1 → blocked (1 P0 + 7 P1)
- [~] Maintenance/deprecation policy proposed, not adopted → M1
- [x] Unsupported functionality identified (deferred list in roadmap)

## 4. Recommendation

**NO-GO for declaring v1.0-ready now; GO for entering v1.0 stabilization.** No architectural risks found; smallest path = fix 1 P0 + 7 P1 + wording/policy freeze. No new language features, no ontology/compiler changes, no `build`/`publish`/registry in M1.
