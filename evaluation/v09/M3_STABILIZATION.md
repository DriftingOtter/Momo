# v0.9 M3 — stabilization report

## Contracts established

- **Public API inventory** reconciled (82 exports) with A/B/C/D
  classifications in `docs/API_STABILITY.md`; no v1.0 contract declared.
- **`tl` semantics**: discovery (nearest manifest with
  `[tool.tensorlang]`), manifest validation (file/field/hint),
  freshness states (`fresh`/`stale:*`/`no-state`/`no-environment`)
  via advisory hash record — `run`/`check`/`test` warn on staleness
  without auto-upgrading; `doctor` reports without mutating;
  `sync --locked` is the strict path.
- **Arg forwarding**: flags/spaces/`--`/unicode/exit-status/exceptions
  verified; app args are `sys.argv[3:]`; no shell interpretation.
- **Isolation**: project envs never satisfy missing deps from outer
  environments (verified with pytest present outside, absent inside).
- **Mask/module/effect guarantees** preserved from M1 (563 → 590 with
  M2; all green).

## Evidence

- 7 new M3 tests (freshness, staleness warning, lockfile regen,
  leak isolation, exit status, unicode args) + full `tl` matrix.
- Managed re-runs: App A params 1e-9, App B 1774 + mse 1e-12, App C
  mean 16 digits (prior runs; unchanged code paths).
- Skip accounting: integration tests skip explicitly only when uv or
  the index is unreachable (`pytest -rs` shows reasons); hermetic
  tests need neither.

## Residual P2 items (documented, not blocking)

- `tl test` requires pytest (hinted); non-template layouts need
  manual file arrangement; `run` uses env as-is by design.
- Frame arithmetic stays positional by design; `==` of masks unfused;
  REPL retention subtle; `extra=` `=`-override silent-but-documented.
- No registry/solver/publish/build; no v1.0 contract.

## Recommendation

Begin v0.9.0 release validation (clean install, uv paths,
six-cell CI, packaging, app reproduction). No open P0/P1.
