# M0 — Projects & distribution (`tl` + reusable libs + `tl build` need)

Experiments in `/tmp` only; nothing published. No code changes.

## 1. `tl` verification (v0.9 interface)

`new/init/add/remove/sync/run/check/test/doctor` all present; help exit 0; unknown exit 2. Template has no abs paths (only explicit `--tensorlang` dev source).
Verified: clean create/deps/locked sync; `pyproject` truth + `uv.lock` record (arg arrays only; `tl` never parses lock internals; freshness = SHA256 files-vs-record); `run` warns on stale, never auto-upgrades (`uv run --no-sync`; missing `.venv` auto-created); `--locked` strict (stale → exit 1); hashes ≠ integrity (doctor states "installed contents not independently verified"); `doctor` read-only (mtimes unchanged); isolation (separate `.venv`s; global pytest doesn't leak); nested-CWD discovery; spaces ok; unicode dir ok for `init`; argv verbatim incl. empty/flags/`--`/spaces/unicode/quoting, `sys.argv[3:]`, exit-code passthrough (7→7); missing uv → actionable, no pip/global fallback; offline → exit 1; relocation copy-reproduces (absolute dev source must persist — documented dev-only).
Full `test_tl.py`: 33 passed, 0 skipped (network+uv present); CI `-rs` + no-silent-skips gate covers offline case.

## 2. Gaps found (evidence-backed)

1. **Corrupt manifest misdiagnosed (P0):** `find_root` swallows `TOMLDecodeError` → "no project found" instead of "invalid manifest". `load_manifest` has the right error but unreachable via CLI.
2. **`tl add` rejects local paths/wheels (P1):** `_PKG_RE` index-only; lib wheel enters only via `uv pip`/hand-edit — blocks the lib workflow; untested-as-feature.
3. **Uv error detail truncated (P1):** `run_uv` keeps last stderr line only (offline shows "cache."); exits correct, context poor; no `--verbose`.
4. **`tl new` rejects unicode names, `tl init` allows unicode dirs (P2):** inconsistent, undocumented ASCII rule.
5. `needs_uv` marker defined but never applied (dead; runtime `_networked()` skip used) (P2).
6. Managed venv has no `pip` — must use `uv pip --project`; undocumented (P2).
7. `[tool.uv.sources] path=` raw string (no URI escaping); spaces ok macOS; non-ASCII/Windows-drive untested (P2).

## 3. Library distribution experiment

Minimal `mylib-tl` wheel (hatchling, std packaging): `.tl` included in wheel+sdist by default. Installed into separate TL project via `uv pip install`:
- `import mylib_tl` from `.tl` — works (Python-first).
- `from mylib_tl.operators.calibration import calibrate`, apply/compose → correct numerics — works.
- `run_file(site-packages abs path)` — works. Vendored `src/vendor/` copy — works.
- `tl add <wheel path>` — fails (gap 2).

## 4. Does v1.0 need `tl build`?

Evidence: zero `tl` support needed for source inclusion/wheel metadata/installability today. A thin `python -m build` delegate would add: (a) guaranteed `.tl` inclusion across backends (setuptools `packages.find` can drop non-`.py` data — template+doctor check prevents); (b) metadata/entry conventions (no `.tl` entry-point concept today); (c) `tl add` path-spec acceptance. Registry/versioning/`publish` stay `uv`/twine (out of scope).
Recommendation: **conditional M2** — implement only `tl package --check` (wheel contains `.tl` + fresh-venv import probe) + path-spec `tl add` IF distribution evidence demands; do NOT build a custom packager/registry/`publish` for v1.0.

## 5. M0 verdict

Project workflow reproduces across clean envs; locked path strict. M1 must fix corrupt-manifest (P0) + decide `tl add` paths/truncated errors (P1) before any v1.0 freeze.
