# v0.9 M0 — distribution

Verified with the released `tensorlang-0.8.0` wheel (not the source tree):

- Clean venv (`pip install tensorlang-0.8.0-py3-none-any.whl` + deps).
- App directory copied to a path containing spaces, outside the repo.
- `tensorlang run app/main.tl` and `pytest app/tests/` green
  (App A: run + 4 tests).
- `__file__`-relative data discovery works from any CWD.
- Deterministic installs via pinned versions (venv record in report).

No package manager, format, or build step was needed: a `.tl` project
is shippable as plain files + a dependency list. A minimal
`pyproject.toml` declaring dependencies is sufficient convention;
evaluated as adequate, no new format proposed.

Limitation: test suites that deliberately trigger import failures
should run in fresh processes (F-A2 poisoning note), which is a test
hygiene rule, not a distribution blocker.
