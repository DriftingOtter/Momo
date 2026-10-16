# M4 — Compatibility freeze (wording final, activation deferred)

## Document status

`docs/COMPATIBILITY.md` is the **intended** v1.0 contract with
M4-finalized wording. It does **not** take effect for v0.9.0 and
becomes binding only when the v1.0.0 tag is approved and published
(§8 Activation). `docs/API_STABILITY.md` carries the same header:
"No v1.0 guarantee is active."

## What is frozen (wording)

- Stable scope: 66 exports (§2 + API_STABILITY stable table).
- Provisional boundary: 16 exports, explicitly not frozen.
- Semantic versioning incl. pre-1.0 no-promise rule and v0.9
  correctness precedents (`TL0512`, `TL0419`, dual-spelling).
- Compatible additions, deprecation windows (announce +
  `DeprecationWarning` + removal ≥ next major; `check_source`
  plumbing is the live example), breaking-change rule.
- Codes frozen, wording not; numerics within stated tolerances
  (exact integer paths, `rtol=1e-4`/`atol=1e-6` float/AD refs,
  app parities, no bitwise promise, ~1e-12 group-noise documented);
  backend freedom; manifest additive-only; Python 3.12–3.14;
  Tier-2 Windows; `uv` required with URL redaction.
- M4 floor evidence recorded in §7 (`polars>=1.21`,
  `scipy>=1.11`, `zarr>=3.0.8`).

## Verification (all green, local)

- 639 core tests (incl. API inventory, portability, catalogue
  freshness gates), 0 skipped with network present.
- 15/15 app tests (parities: SciPy 1e-9, analytics 1774 + 1e-12,
  out-of-core 16 digits).
- 10/10 examples from clean state (one policy-aligned example
  fix: `release_check.tl` bitwise assert → `allclose` 1e-12,
  matching §4; plus one policy-aligned test fix:
  `test_integration.py` parquet round-trip tolerance).
- Floor venv (py3.12, oldest non-yanked minimums): 540 passed,
  4 pre-existing skips; 3 floor defects found and corrected
  (zarr + polars + scipy floors; sales round-trip tolerance).
- `tl` lifecycle, local-wheel consumer loop, and onboarding guide
  re-validated on the final tree (see M4_RELEASE_CANDIDATE.md).

## Pending (not claimed)

Remote CI cells (Ubuntu × 3, macOS × 3.13, packaging, project
gate) require a pushed branch/final commit — configured, not
executed. Windows stays Tier-2. Wiki publication is a separate
post-approval step (checklist in M4_RELEASE_CANDIDATE.md).
