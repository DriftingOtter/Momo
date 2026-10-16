# v0.9 M2 — project manager report

## Implementation

`src/tensorlang/project.py` (~600 lines) + `tl` console entry;
no compiler/runtime changes except additive `tensorlang run`
argument forwarding (`sys.argv` visible to programs).

- Manifest: `[project]` + `[tool.tensorlang]` (entry, source-dir);
  discovery walks up; validation errors name file/field/hint.
- `new`/`init` scaffold without overwriting; names validated;
  templates contain no checkout paths.
- `add`/`remove`/`sync` delegate to uv subprocess arrays;
  `--dev`, `--locked` supported; missing uv → actionable error.
- `run` auto-syncs missing envs, executes via `uv run --no-sync`
  in the project env; `check`/`test`/`doctor` likewise managed.
- 27 regression tests (16 hermetic + 11 integration with explicit
  uv/network skip).

## Validation

- Acceptance flow green: `new` → `add` → `sync` → `run` → `test`;
  copy-to-new-location reproduces byte-identical output.
- Apps A (params 1e-9), B (1774, mse 1e-12), C (mean 16 digits)
  migrated without algorithm changes; managed runs match
  unmanaged numerics.
- Isolation: identical module names across projects never mix
  (separate processes); nested CWDs and spaced paths work.
- Overhead: `tl run` ≈ 1.6 s vs 0.6 s direct (uv startup + env
  check); no numerical cost.
- Full suite 590 green (563 + 27) on 3.14.

## Limitations

- `tl test` needs pytest (`tl add --dev pytest` hinted).
- Migrating non-template layouts needs manual file arrangement
  (`init` never overwrites; entry path is manually editable).
- `tl run` uses the env as-is (staleness enforced via
  `tl sync --locked`, not implicitly).
- Threads inherit `TL0315`; one program per process.
- No registry/solver/publish/build (explicitly deferred).
