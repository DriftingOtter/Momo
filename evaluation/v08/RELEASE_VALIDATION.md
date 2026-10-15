# TensorLang v0.8.0 — release validation

Branch `evaluation/v08-applications` vs immutable baseline `v0.7.0`
(`80bde0d`). All work uncommitted; no tag/merge/push performed.

## 1. Inventory

In-scope changes only — frontend effect/module systems
(`effects.py`, `parser.py`, `compiler.py`, `repl.py`, `cli.py`,
`diagnose.py`), error registry (`errors.py`: `TL0203`, `TL0305`,
`TL0310`–`TL0315`), four example path-precompute migrations,
new `frontend/modules.py`, five new test files, docs
(`EFFECTS`/`MODULES`/`COMPARISONS`), spec/README/changelog updates,
`evaluation/v08/` evidence. **Zero changes** to the numerical backend,
operators, runtime, Tensor/Operator/Lazy ontology, or persistence.
Generated data, benchmark outputs, `dist/`, caches excluded
(`dist/` + benchmark results verified ignored/reverted).

## 2. Test totals

- Core: **549 passed** (463 baseline + 86 new), Python 3.14/macOS.
- Core on Python 3.12/macOS: **549 passed**.
- Application: **11 passed**.
- No-skips gate (per-file collect-only): clean.
- Catalogue freshness gate: green (in-suite).

## 3. CI matrix (honest status)

| Cell | Status |
|---|---|
| macOS × 3.14 | 549 green (local) |
| macOS × 3.12 | 549 green (local venv) |
| macOS × 3.13 | not installed locally — CI on push |
| Ubuntu × 3.12/3.13/3.14 | cannot run locally — CI on push |
| Packaging (ubuntu/py3.14) | mirrored locally (below) |

## 4. Packaging

`tensorlang-0.8.0` wheel + sdist built locally. Clean venv install
(py3.14, outside repo): CLI, `run`/`check`/`repl`, sibling + dotted
`.tl` imports (incl. spaces-in-path), precedence, nested modules,
effect/import diagnostics, `run_file`/`run_source`, structured→dense,
grouped counts — all green from an unrelated CWD. Not published.

## 5. Concurrency gate (was P1, now closed)

Reproduced cross-talk (wrong-module binding / `ModuleNotFoundError`
races). Fix: thread-ownership on sessions + locked check-and-push +
requester verification → overlapping cross-thread sessions fail fast
with `TL0315` (5/5 trials: exactly one winner with correct values,
loser `TL0315`, zero contamination). Same-thread nesting and
sequential cross-thread sessions verified working. Tested in
`test_loader_audit.py`; documented (spec + `MODULES.md`).

## 6. Effect enforcement (M1 preserved)

`TL0305` across `.tl`/Python-hosted/imports/REPL/CLI; nested/aliased/
injected/smuggled forms blocked pre-effect (20 M1 + 12 injection
tests). `--allow-effects` verified not to bypass validation.
`TL0301/302/303` precedence intact. Trust stays identity-based.

## 7. Injection (M3)

`:=`-over-`extra` → `TL0203` (no execution); `=` keeps module
precedence; dunder rejected; no inheritance by imports.

## 8. Modules (M2)

Python-first, file-relative, isolated namespaces, per-session cache,
`TL0310`–`TL0315` chains, static cycle rejection (+ runtime backstop),
transitive CLI preflight, phantom-cycle + nested-anchor scan fixes,
synthetic package shells (no `sys.path` dependence), third-party
preservation (scipy identity), `sys.path`/`sys.modules` hygiene.

## 9. Missing values (M3)

Contract tested (`test_missing_values.py`) and documented
(`COMPARISONS.md`); nullable boolean → float64/NaN verified; no
semantic changes.

## 10. Application equivalence

94,976 rows; coefficients agree to 1e-12 across native `.tl`
(0.6310866430046405), driver (…352), NumPy reference (…388).
Native entry 56 LOC vs 94 driver; cold ~166 ms / 106 MB peak;
stage import ~2 ms cold, 0.1 ms cached. Small benchmark workloads
ran (dense/structured/pipeline/autodiff); outputs reverted.

## 11. Limitations (all P2, documented)

Foreign-object dunders, comprehension iteration, lambda defaults;
dynamic-`__import__` cycles; `.tl` imports from plain Python;
function-level `.tl` imports resolve against the caller;
`=`-over-`extra` silent-but-documented; no hot reload / package
manager / v1.0 contract.

## 12. Defects

P0: none. P1: none open (concurrency closed by `TL0315` guard).

## Recommendation

**Ready to release v0.8.0**: commit, run CI (ubuntu cells + 3.13),
verify green, tag annotated `v0.8.0`. Do not publish the package
without owner approval.
