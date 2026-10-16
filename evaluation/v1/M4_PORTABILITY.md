# M4 — Portability gate

## Mechanism

`scripts/check_portability.py` (stdlib only, no network) scans
`tests/`, `examples/`, `src/tensorlang/`, `.github/`, `docs/` for
eight developer-assumption patterns (macOS/Linux home paths,
`$HOME`, `~/`, Windows profiles, `/private/`, absolute `/tmp/`,
`PYTHONPATH`). Findings print as `file:line [rule] reason`.
Reviewed exemptions live in `ALLOWLIST` with reasons — never a
broad waiver.

CI: `packaging` job runs the script before the clean-install smoke
test (single cell, no network beyond setup). Local gate:
`tests/test_portability.py` unit-tests each rule and fixture,
requires every allowlist entry to be live (path exists + rule
fires), and asserts a clean self-scan.

## Adjudicated findings (all allowlisted, none a real defect)

- `tests/test_tl.py:54` — asserts the project template contains no
  `/Users/` paths (the test's subject, not an assumption).
- Six `.tl`-string fixtures with `/tmp/...` paths
  (`test_declaration_effects`, `test_effects`, `test_modules`,
  `test_reliability`) — paths are rejected statically
  (`TL0302`/`TL0305`) and never touch the filesystem.
- `src/tensorlang/frontend/repl.py:330` — `~/.tensorlang_history`
  via `expanduser` is user-scoped product behavior, not a test
  assumption.
- `.github/workflows/ci.yml` — `/tmp/smoke`, `/tmp/tlci` are
  CI-only scratch locations.
- `docs/GETTING_STARTED.md` — instructional `/tmp/repro` copy
  target for the reader's own machine.

## Verification

- `python scripts/check_portability.py` → clean (rc 0).
- `tests/test_portability.py` → 3 passed.
- Full suite green with the gate included (639 passed).

## Standing rule

Any new hardcoded checkout path fails CI. Additions to ALLOWLIST
require a reason and must keep the dead-entry test green (every
entry re-verified on each run).
