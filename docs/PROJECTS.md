# TensorLang projects (`tl`)

`tl` is a Cargo-inspired project manager over uv + the TensorLang
runtime. It orchestrates environments and execution; it does not
compile, resolve, or execute anything itself.

## Workflow

```bash
tl new my-app --tensorlang /path/to/TensorLang   # or plain index dep
cd my-app
tl sync        # create .venv, install deps, write uv.lock
tl run         # execute [tool.tensorlang] entry in project env
tl check       # validate without executing
tl test        # pytest in project env (needs pytest dev dep)
tl doctor      # manifest/env/runtime diagnostics
tl add numpy 'scipy>=1.14' [--dev] / tl remove pkg / tl sync --locked
tl add ./dist/mylib-0.1.0-py3-none-any.whl  # local wheel (M1)
```

## Manifest (`pyproject.toml`)

`[project]` (name/version/requires-python/dependencies) is the single
dependency source of truth. `[project] name`/`version` must be strings.
Unknown fields are ignored (forward-compatible), but a manifest that
cannot be parsed as TOML, or whose `[tool.tensorlang]` table is
malformed, is an **invalid manifest** — reported as such, never as a
missing project (see Rules). `[tool.tensorlang]` configures:

```toml
[tool.tensorlang]
entry = "src/main.tl"
source-dir = "src"
```

`[tool.uv.sources]` (standard uv) expresses local TensorLang checkouts
for development; generated templates stay portable (no embedded paths
except an explicit `--tensorlang` flag). Until a TensorLang release is
published to an index, default templates require `--tensorlang PATH`
(or equivalent `[tool.uv.sources]`) to synchronize.

Local wheels (`tl add ./dist/pkg-0.1.0-py3-none-any.whl`) are recorded
through uv's path-source mechanism (`[tool.uv.sources]`, project-
relative when the wheel lives inside the project). A project that
depends on a local wheel reproduces only together with the wheel file:
copy the wheel to the same project-relative location, then
`tl sync` / `tl sync --locked`. A missing wheel artifact fails with an
actionable diagnostic at synchronization time. `tl` never edits
`uv.lock` internals and never claims portability for projects whose
local artifacts were not copied along.

## Environment freshness

Three distinct states — declared (`pyproject.toml`), resolved
(`uv.lock`), installed (`.venv`) — are never conflated. `tl` records
manifest/lockfile hashes at each successful sync
(`.venv/tl-sync-state.json`, advisory only):

- `tl run`/`check`/`test` create a missing environment, then use the
  existing one as-is — with a stderr warning when hashes show staleness.
- `tl sync --locked` fails on missing/stale lockfiles (CI path).
- Freshness states (`fresh`/`stale:*`/`no-state`/`no-environment`)
  compare manifest/lockfile hashes against the last sync record only;
  installed package contents are never independently verified.
- `tl doctor` reports `fresh` / `STALE ...` / `no-state` /
  `no-environment` without altering anything.

## Rules

- Project root: nearest `pyproject.toml` containing `[tool.tensorlang]`.
  Discovery distinguishes three states: no manifest (search parents),
  valid manifest (project root), invalid manifest (actionable
  `invalid manifest <path>: <toml location>` diagnostic — discovery
  stops and never silently selects an outer project). `tl doctor`
  reports `project: INVALID ...` for malformed manifests without
  modifying files or environments.
- uv failures preserve actionable context: up to eight trailing
  diagnostic lines plus the subprocess exit status (credentials in
  URLs redacted). `tl` never invokes uv through a shell.
- `tl run` auto-creates a missing environment (`uv sync`), then runs
  `python -m tensorlang run <entry> [args...]` via `uv run --no-sync`
  (no implicit upgrades). Trailing args reach the program through
  `sys.argv` (after the runner's own entries: with
  `tl run -- args...`, the application arguments are `sys.argv[3:]`).
- `tl test` requires pytest (`tl add --dev pytest`); runs `pytest tests`.
  Project environments contain no `pip` binary: out-of-band installs
  use `uv pip --project <root>`.
- Project names for `tl new` are ASCII (`[A-Za-z0-9][A-Za-z0-9_-]*`);
  `tl init` adopts the existing directory name instead (unicode working
  directories are supported; unicode project *names* are not).
- `tensorlang doctor` reports the runtime environment; `tl doctor`
  diagnoses a project (manifest, lockfile, freshness, environment).
  Similar names, different scopes.
- `tl sync --locked` fails on missing/stale lockfiles (CI use).
- Missing uv → actionable error, never a pip/global fallback.
- Each project gets an isolated `.venv`; same-named modules in
  different projects never mix (separate processes).
- `tl init` never overwrites existing files; existing manifests are
  left untouched (add `[tool.tensorlang]` manually).
- Threads: TensorLang loader sessions remain single-threaded
  (`TL0315`); `tl` runs one program per process.

## Architecture

Responsibilities: TensorLang owns language execution/diagnostics;
uv owns versions/venvs/resolution/lockfiles; Python packaging owns
metadata; `tl` owns scaffolding, discovery, and subprocess
orchestration (argument arrays only — no shell execution, no custom
solver, no sandbox claims).
