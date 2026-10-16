# TensorLang v0.9.0 — release validation

Branch `evaluation/v09-api-readiness` vs immutable `v0.8.0` (`c3615e2`).
Uncommitted; no tag/merge/push performed.

## 1. Inventory

Modified: CHANGELOG, LANGUAGE_SPEC, README, ERROR_CODES, MODULES,
pyproject (`tl` entry, version 0.9.0), cli.py (additive run-args),
errors.py (`TL0419`, `TL0512`, `TL0315` from v0.8 cycle),
frontend/{modules,compiler,diagnose,effects,parser,repl}.py,
operators/ml.py, runtime/{executor,polars_plan}.py, 4 examples
(path precompute), CI workflow (+project gate), tests.
New: `frontend/project.py` (`tl`), docs
(API_STABILITY/COMPARISONS/EFFECTS/MODULES/PROJECTS),
tests (mask_domains, injection, loader_audit, missing_values,
modules, tl), `evaluation/v09/` (8 reports + app sources, no
generated data). No numerical-backend/ontology changes beyond
documented M1 corrections.

## 2. Tests

- Core: **596 passed** (local macOS py3.14; 3.12 subset green).
- `tl` suite: **33 passed, 0 skipped** (uv + network present).
- Apps: 15 green (4 + 7 + 4), incl. managed `tl test` runs.
- Catalogue freshness + no-skips gates green.

## 3. CI matrix (honest)

Local: macOS × 3.12/3.14 green. Ubuntu × 3, macOS × 3.13, packaging,
new project gate: **pending push** (workflow updated, not yet run).

## 4. Packaging

0.9.0 wheel+sdist built. Clean venv: `tensorlang` + `tl` entry
points, template project flow (`new/sync/run/test/check/locked`),
sibling+dotted imports, precedence, diagnostics, REPL, dense/grouped
ops — green from unrelated CWD (incl. spaces). Default template
needs a published index **or** `--tensorlang PATH` (documented;
tensorlang unpublished pre-release).

## 5. uv integration

`add/remove/sync(+locked)/run/check/test/doctor` exercised for
real; stale-lock rejection, offline failure, missing-uv diagnostic,
lockfile regen, copy-reproduce byte-identical. Isolation: overlapping
module names + distinct deps never mix; global packages don't leak
in (verified with pytest).

## 6. Freshness

Hash-record states; `run` warns on stale, never auto-upgrades;
`doctor` read-only; wording explicitly disclaims installed-content
verification.

## 7. Numerics

App A 1e-9, App B 1774 + 1e-12, App C 16 digits — all preserved via
managed workflows. Benchmarks ran; outputs reverted; no regressions.

## 8. API/docs

82 exports reconciled in `docs/API_STABILITY.md` (A/B/C/D, no v1.0
contract). Mask/module/freshness/argv contracts documented.
Known P2 residuals listed; no P0/P1 open.

## Recommendation

**Ready to commit; CI must still run** (6 cells + packaging +
project gate) before merge/tag. Requires network + uv in CI
(present by default).
