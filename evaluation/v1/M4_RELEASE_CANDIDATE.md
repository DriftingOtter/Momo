# M4 — Release candidate hardening

Branch `evaluation/v1-readiness`, baseline v0.9.0 (`bbbef00`).
No production-code changes except three evidence-backed floor/test
corrections (below); no new subsystems. All work uncommitted;
wiki untouched (reviewed, not published).

## 1. Working-tree inventory (vs v0.9.0)

- **M0 evaluation**: `evaluation/v1/M0_*.md`, `V1_ROADMAP.md`
  (read-only findings; app data/benchmark outputs excluded).
- **M1 corrections**: `src/tensorlang/project.py` (manifest
  discovery, wheel `add`, uv context), `cli.py` + `compiler.py` +
  `diagnose.py` + `errors.py` (`TL0103`, `check_source`
  deprecation), `LANGUAGE_SPEC.md` (§3, §9, join order, M4 wording),
  `docs/` (PROJECTS, EFFECTS, ERROR_CODES regen, API_STABILITY,
  COMPATIBILITY, COMPARISONS), `tests/test_m1_stabilization.py`,
  `tests/test_diagnose.py` (contract update).
- **M2 decision**: `evaluation/v1/M2_DISTRIBUTION_DECISION.md`
  (no code).
- **M3 docs**: `docs/GETTING_STARTED.md`,
  `docs/TROUBLESHOOTING.md`, README links, CHANGELOG notes,
  `M3_*.md` reports.
- **M4 hardening** (this milestone):
  - `pyproject.toml` floors: `polars>=1.21`, `scipy>=1.11`,
    `zarr>=3.0.8` (evidence below; numpy/sympy/dask floors stand).
  - `tests/test_public_api.py` (82-name + signature + partition
    gate), `tests/test_portability.py` + `scripts/check_portability.py`
    (+ CI packaging step), `scripts/release_check.py`.
  - `tests/test_integration.py`: parquet round-trip tolerance
    (policy-aligned, see §4).
  - `examples/release_check.tl`: bitwise assert → `allclose`
    1e-12 (policy-aligned, see §4).
  - `docs/CAPABILITY_MATRIX.md` (to_dense scope),
    `docs/COMPATIBILITY.md` (M4-finalized wording + floor evidence),
    CHANGELOG (M4 entry).
  - `evaluation/v1/M4_*.md` (5 reports).

## 2. Floor corrections (P1 class, evidence-backed)

1. **zarr>=3.0.1 → >=3.0.8.** The declared floor permitted
   yanked releases including 3.0.1 (`zarr.load() deletes data`;
   pip installs it with only a warning). 3.0.0–3.0.7 are
   yanked/hidden; 3.0.8 is the oldest available non-yanked
   release and the version the v0.5 comment already validated.
2. **polars>=1.0 → >=1.21.** `rolling_*` with `min_samples`
   (which TensorLang's lowering passes) does not exist through
   1.20.x (bisected 1.10/1.15/1.18/1.19/1.20 lack it, 1.21.0 has
   it) — the old floor broke `rolling(..., min_periods=...)`
   with `TL0499`.
3. **scipy>=1.10 → >=1.11.** 1.10.x ships no Python 3.12 wheels
   (supported floor is ≥3.12), forcing a source build; 1.11.4
   verified working.
4. **Sales round-trip tolerance.** `test_integration.py` file
   comparison now uses column equality + `allclose` 1e-9: the
   in-memory check stays exact; only the parquet round-trip
   absorbs documented last-ulp group-sum noise.

Floor venv (py3.12.14, numpy 1.26.4 / scipy 1.11.4 / polars 1.21.0 /
sympy 1.12 / dask 2025.1.0 / zarr 3.0.8): **540 passed, 4
pre-existing skips**. Untestable as declared: exact historical
minimums where wheels don't exist for supported Pythons
(recorded above, not silently substituted).

## 3. Verification summary (final tree, local)

- Core **639 passed, 0 skipped** (633 + API inventory + portability).
- Apps **15/15** (SciPy 1e-9; 1774 + 1e-12; Dask 16 digits).
- Examples **10/10** from clean state (inputs generated, each run
  once — re-runs of writers correctly hit TL1002 by design).
- `tl` lifecycle (new/sync/run/check/test/doctor/add/locked/
  relocate/argv/exit codes) green in spaced+Unicode path.
- Local-wheel loop green (build → add → import/apply/compose →
  locked clean-location reproduce → missing-artifact failure).
- Getting-started guide re-executed on the final tree (M4 shop
  project: `[17. 29.]`, 1 passed, locked + doctor healthy).
- Wheel/sdist inspected: 56 files, both console scripts,
  corrected floors in metadata; clean py3.12 install smoke green.
- Platform coverage: **macOS × 3.12/3.14 executed** (main + floor
  envs). Ubuntu × 3, macOS × 3.13, packaging, project gate:
  configured, pending push — **not claimed**.

## 4. Findings

- P0: none. P1: the three floor corrections above (fixed in M4).
- P2: prior residuals stand (M1 report §7); no new blockers.
- No silent numerics, mutation, alignment, or session defects.

## 5. Recommendation

M4 acceptance criteria met. Proceed to v1.0.0 final release
validation (push → 8 remote checks → merge → final-tree CI →
tag → optional index publication → wiki publication as a
separate step). No further milestones needed before the gate.
