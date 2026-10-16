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
```

## Manifest (`pyproject.toml`)

`[project]` (name/version/requires-python/dependencies) is the single
dependency source of truth. `[tool.tensorlang]` configures:

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
- `tl run` auto-creates a missing environment (`uv sync`), then runs
  `python -m tensorlang run <entry> [args...]` via `uv run --no-sync`
  (no implicit upgrades). Trailing args reach the program through
  `sys.argv` (after the runner's own entries: with
  `tl run -- args...`, the application arguments are `sys.argv[3:]`).
- `tl test` requires pytest (`tl add --dev pytest`); runs `pytest tests`.
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
