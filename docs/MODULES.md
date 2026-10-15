# TensorLang modules (v0.8 M2)

`.tl` files import each other with existing Python syntax. No new
grammar, no package manager, no `__init__` execution.

## Supported syntax

```python
import calibration
from calibration import calibrate
from calibration import calibrate as apply_cal
from operators.calibration import calibrate   # dotted → relative path
```

## Resolution

- Dotted names map to relative paths: `a.b` → `a/b.tl`.
- Resolved relative to the **importing file**, never `sys.path` or CWD.
  (`run_source` uses its filename's directory; the REPL uses CWD.)
- **Python wins**: the loader is last on `sys.meta_path`. A sibling
  `json.tl` never shadows the stdlib from `.tl` code.
- Parent directories import as inert namespace packages (bare
  directories also resolve without `sys.path` cooperation, via
  synthesized package shells); `__init__` files are never executed.
- Missing modules → `TL0310`; missing symbols → `TL0311`; relative
  imports → `TL0314` (use absolute dotted paths).

## Execution and caching

- Each module compiles through the existing frontend — identical
  declaration scope checks and M1 `TL0305` effect validation.
- Each module gets an isolated namespace; only imported names enter
  the importer. Top-level runtime statements execute on import (imports
  are not side-effect-free by claim).
- One execution = one `LoaderSession`: repeated imports reuse the
  initialized module and its first-class operator objects (uid-scoped
  `{X}` parameters untouched — no filename-based scoping).
- Independent `run_file()` calls are isolated sessions (modules
  re-initialize; numerical caches were never shared either). The REPL
  keeps one session until `:reset`.
- Failures are never cached: a failed import can be retried.

## Circular imports

Rejected entirely (`TL0312`) with the import chain, detected by static
pre-scan before anything executes:

```
main.tl:1
  imports cycle_a
cycle_a.tl:1
  imports cycle_b
cycle_b.tl:1
  imports cycle_a
TL0312: circular TensorLang import
```

Python-style partially-initialized circular modules are unsupported.

## Interaction with `:=` (M1 preserved)

Imported names are ordinary bindings, not trusted callables:

- `pipeline := calibrate >> normalize` — fine (no calls).
- `result = calibrate(data)` — fine (`=` runtime context).
- `r := calibrate(data)` — `TL0305` (call through an imported name,
  same as any unclassified callable; use `=`).
- `x := evil.grab()` smuggled through a `.tl` re-export is still
  `TL0305` with no effect: the *imported module's own* declarations
  were validated at its compile time.

## CLI

`graph`/`explain`/`profile` preflight walks the transitive `.tl`
graph statically: dependency effects and cycles refuse introspection
unless `--allow-effects` (cycles always refuse). `--allow-effects`
never bypasses `TL0305`.

## Unsupported (explicit)

Relative imports, packages with `__init__` logic, hot reloading,
cross-execution global caches, function-level `.tl` imports (they
resolve against the caller, not the defining file), `.tl` imports
from plain Python (`import calibration` in a `.py` driver stays
`ModuleNotFoundError` — use `run_file`), shadowed stdlib directory
names (e.g. an `io/` directory loses to stdlib `io`; name it
`inputs/`).

## Namespace injection (`extra=`)

`run_source(src, extra={...})` injects values into the entry module
only. `=` rebinding keeps module precedence; `:=` over an injected
name is `TL0203`; dunder names are rejected; imported modules never
inherit injections. Injected callables stay untrusted under `:=`
(`TL0305`) unless they are identity-verified TensorLang public
objects. See `docs/EFFECTS.md`.

## Concurrency

Loader sessions use process-global hooks (`sys.meta_path`,
`sys.modules`, `sys.path`, one current-file pointer) and are **not
thread-safe**: overlapping independent sessions from different threads
are rejected fail-fast with `TL0315` (verified: exactly one session
proceeds, the other is rejected — never silent contamination).
Same-thread nesting joins the active session; sequential sessions in
any thread work normally. Parallel execution must use subprocesses —
consistent with the subprocess-isolated benchmark methodology.
