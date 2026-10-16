# M4 — Release checklist (repeatable, verify-only)

Mechanism: `scripts/release_check.py` (stdlib + pytest + build;
temp dirs under `/tmp` only). It never commits, merges, rebases,
tags, pushes, publishes, touches the wiki, or deletes user files,
and it reports executed/skipped/failed honestly.

## Gate inventory and local status

| Gate | Local status |
|---|---|
| git working-tree state | reported (25 uncommitted paths; no commit performed) |
| version consistency (`pyproject`/`__init__`/CHANGELOG) | PASS (0.9.0) |
| error-catalogue freshness | PASS (render == `docs/ERROR_CODES.md`) |
| API inventory (`test_public_api`) | PASS (3) |
| portability (`test_portability` + script) | PASS (3 + clean scan) |
| error codes / diagnose / M1 suites | PASS (8 + 9 + 37) |
| core suite `--full` | PASS when run (639, 0 skipped) |
| app suite `--apps` | PASS when run (15/15; regenerates gitignored data) |
| dist build + entry_points | PASS (56-file wheel, both console scripts) |
| CI gates | configured (correctness ×6, packaging, project, benchmarks-manual); remote execution pending push — not claimed |

Quick mode (`scripts/release_check.py`): 13/13 green in ~40s.
Full mode adds `--full` (core, ~3 min) and `--apps` (data + refs,
~2 min, ~600 MB gitignored scratch).

## Pre-tag manual steps (not automated by design)

1. Review `git status`/`git diff` (only approved stabilization).
2. Run quick mode, then `--full --apps`, on the final tree.
3. Push the release branch; verify all 8 remote checks green.
4. Merge to `main`; verify final-tree CI (merge commit differs).
5. `git tag -a v1.0.0`; `git push origin v1.0.0`; verify resolution.
6. Publish the wiki (checklist below) only after the tag verifies.

## Wiki publication checklist (separate step, not in M4)

- [ ] Re-sync copies: GETTING_STARTED, TROUBLESHOOTING, PROJECTS,
  API_STABILITY, COMPATIBILITY (+ Home/Sidebar links).
- [ ] Verify no claim that v1.0 is already released; policy marked
  proposed until the tag exists, then flip to active.
- [ ] Verify links, commands, and stable/provisional tables match
  the tagged tree.
- [ ] Confirm no unrelated wiki changes present (`git status`).
- [ ] Commit + push the wiki as its own step (never part of the
  source release commit).
