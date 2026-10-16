# FINAL — v1.0.0 release validation (local preflight)

Branch `evaluation/v1-readiness` vs immutable v0.9.0 (`bbbef00`).
No commits, merges, tags, pushes, publishes, or wiki modifications
performed. Owner-controlled stages A–F remain for the owner.

## 1. Baseline

- Branch: `evaluation/v1-readiness`; prior release `v0.9.0`/`bbbef00`.
- All 20 M0–M4 reports present (`evaluation/v1/`).
- 639 core tests passing, 0 skipped (network present, macOS py3.14).
- 15/15 app tests; 10/10 examples from clean state;
  13/13 local release checks.
- No open P0/P1. Work uncommitted (27 paths + excluded session note).

## 2. API inventory reconciliation (discrepancy resolved)

Truth: **82 = 65 functional stable + 16 provisional + `__version__`**
(stable release metadata). M1's "66 stable" counted `__version__`
inside the 66; M1's report formula "`66 + 16 + __version__`" was
arithmetically sloppy (would be 83). No export was reclassified,
added, or dropped: `test_public_api.py` enforces the exact
65/16/version partition and `docs/API_STABILITY.md` now states the
convention explicitly. No guarantee was changed to fix arithmetic.

## 3. Final classification

Stable (65 + version): core/math/shapes/logic/relational/fitted/
first-order AD/linalg/stats/nulls/entry/policy/10 error classes —
each with pinned signature and tested contract. Provisional (16):
`equal`, `not_equal`, `unique`, `over`, `rolling`, `einsum`,
`tensordot`, `moveaxis`, `onehot`, `polynomial`,
`Grad/ValueGrad/VjpOperator`, `ValueGradResult`, `SVDResult`,
`EighResult`. Internal: plumbing (deprecated, warns), chunked-AD
deferrals, harness, nonexistent `build`/`publish`.

## 4. Compatibility policy

`docs/COMPATIBILITY.md` M4-finalized: stable scope, provisional
boundary, semver, deprecation windows, codes-frozen-not-wording,
numerics within tolerances (no bitwise promise), backend freedom,
manifest additive-only, py3.12–3.14, Tier-2 Windows, `uv`
required. Applies **from the v1.0.0 tag only**; v0.9.0
retroactivity explicitly disclaimed. `check_source(source,
filename)` stable; plumbing deprecated.

## 5. Test totals (local, macOS)

- Core: **639 passed, 0 skipped** (py3.14, current deps).
- Floor (py3.12, oldest non-yanked minimums): **540 passed**,
  4 pre-existing skips. Excluded from the floor run: `test_tl.py`
  (33, needs uv+index; covered on main env) — floor coverage is
  the numerical/structural/persistence surface, not the full 639.
- Apps: 15/15 (A: SciPy agreement; B: exactly 1774, 1e-12;
  C: Dask 16 digits, streaming behavior).
- Examples: 10/10 from clean generated inputs (writers correctly
  refuse re-runs via TL1002).
- Portability: script + 3 tests green; allowlist justified.

## 6. Floors (validated)

`numpy>=1.26`, `polars>=1.21` (rolling `min_samples` absent
through 1.20.x), `sympy>=1.12`, `scipy>=1.11` (1.10.x has no
cp312 wheels), `dask>=2025.1`, `zarr>=3.0.8` (3.0.0–3.0.7 yanked
incl. data-loss). Reflected in `pyproject.toml` and wheel
METADATA. Evidence in M4 report.

## 7. Distribution

Wheel (56 files) + sdist inspected: both console scripts,
corrected floors, `src/` in sdist. Clean py3.12 install smoke
green. 1.0.0 bump mechanics proven in a throwaway copy (2-line
bump → build → install → 82 exports); repo stays 0.9.0 until the
owner's Stage-A commit.

## 8. Workflows (final tree)

- `tl` lifecycle in spaced+Unicode path: new/sync/run/check/
  test/doctor/add/locked/relocate/argv/exit codes green;
  TL0103 (rc 1) and `invalid manifest` (rc 1) verified; stale
  warning advisory; refresh-then-lock rule for moved projects
  documented (uv correctly refuses stale locks on changed
  relative geometry).
- Library loop: build → `tl add` wheel → import/apply/compose
  (`[6.]`) → clean-location reproduce → missing-artifact
  actionable failure.
- Getting-started guide re-executed (`[17. 29.]`, tests green).

## 9. CI status (honest)

Configured: correctness Ubuntu/macOS × 3.12–3.14 (6),
packaging (+portability), project (real uv, `-rs`), benchmarks
(manual). Locally executed: macOS × 3.12/3.14, floor py3.12,
packaging-equivalent build+smoke, project-equivalent e2e.
**Pending push: all 8 remote checks.** Nothing remote claimed.

## 10. Issues

P0/P1: none. P2: prior residuals (M1 §7) + moved-project
refresh note (documented). No new defects in final validation.

## 11. Wiki readiness

7 pages mirror the candidate docs (re-synced after M4/M4-final
edits); publication checklist in M4_RELEASE_CHECKLIST.md.
Uncommitted, unpublished — separate post-tag step.

## 12. Owner runbook

- **A.** Review `git status`/`git diff` (sections in M4 report
  §1); set version 0.9.0 → 1.0.0 in `pyproject.toml` +
  `src/tensorlang/__init__.py`; finalize CHANGELOG `## v1.0.0`
  (replace the `in development` header); commit; push
  `evaluation/v1-readiness`.
- **B.** Require 8/8 branch CI green (incl. real uv gate).
- **C.** Merge to `main` via normal workflow.
- **D.** Require 8/8 final-tree CI green.
- **E.** `git tag -a v1.0.0 -m "..."`; push; verify resolution.
  Optional: publish the wheel/sdist to an index (separate action).
- **F.** Flip COMPATIBILITY/API_STABILITY status to active,
  verify links/commands against the tag, then commit + push the
  wiki independently.

## 13. Recommendation

**GO for owner-controlled release** (Stages A–F). Preflight is
complete and internally consistent; the only remaining gate is
actual remote CI execution, which cannot be produced from here.
