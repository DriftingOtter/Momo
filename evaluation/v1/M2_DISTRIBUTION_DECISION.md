# M2 — Distribution decision (skipped as implementation)

Status: **no additional distribution implementation for v1.0**.
M1's local-wheel support plus standard Python packaging already close
the functional loop. `tl build` / `tl publish` / registry remain
explicitly out of scope (post-v1.0 developer-experience candidates).

## Evidence (M0 experiment + M1 implementation)

M0 built a minimal `mylib-tl` wheel with hatchling (`.tl` files
included in wheel + sdist by default) and verified, in `/tmp` only:

- `import mylib_tl` from `.tl` (Python-first precedence) works;
- `from mylib_tl.operators.calibration import calibrate`, then
  application and composition, produce correct numerics;
- `run_file()` against the installed path works;
- a vendored `src/vendor/` copy works as fallback.

M1 closed the workflow gap: `tl add ./dist/<wheel>` normalizes the
path (project-relative inside the root, absolute outside) and
delegates to `uv add` via argument arrays. `uv` records
`[tool.uv.sources]` path entries and honors them under
`tl sync --locked`. The guarded end-to-end test
(`test_local_wheel_end_to_end`) proves: add → sync → import →
apply → compose → `tl run` → locked copy-reproduction with the
wheel at the same relative location → clean failure when the
artifact is absent.

## What this establishes (and what it does not)

- **Installing libraries**: supported (`tl add` wheel/dir path or
  index requirement; `uv pip --project` for out-of-band cases).
- **Building artifacts**: supported by standard Python tooling
  (`python -m build`, hatchling/setuptools); `.tl` data files are
  included by default under hatchling, but setuptools
  `packages.find` configurations can silently drop non-`.py` data —
  library authors must verify wheel contents (M3 documents how).
- **Publishing artifacts**: out of scope; `uv publish`/twine to any
  Python index work unchanged. No TensorLang-specific step exists.
- **Registry**: none; explicitly not required for v1.0.

The remaining gap is usability (discoverability, error messages,
prose), not mechanism. M3 owns the library-consumer guide and
troubleshooting; `tl build` stays a post-v1.0 convenience proposal.

## Reopen condition

Reopen distribution work only on a reproducible blocker that
documented tooling cannot address (e.g. a supported `tl add` form
that cannot install a valid artifact, or a locked-reproduction
failure with artifacts present). M3 found none (see
`M3_DEVELOPER_EXPERIENCE.md`).
