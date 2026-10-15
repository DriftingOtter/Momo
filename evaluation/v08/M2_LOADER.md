# v0.8 M2 — native file-based module loading

## Implementation (narrow frontend extension)

`src/tensorlang/frontend/modules.py` (~600 lines): a last-on-`sys.meta_path`
finder + `LoaderSession` + static graph tools. No new grammar, no new
execution engine, no package manager.

- **Resolution**: dotted name → relative path from the importing file
  (`operators.calibration` → `operators/calibration.tl`); submodule
  requests resolve the trailing component under the parent `__path__`.
  Python finders run first (stdlib `json` beats sibling `json.tl`).
- **Execution**: `compile_source` (lexer → parser with M1 rules) + exec
  in an isolated namespace with `__file__` set; module dir on `sys.path`
  during its init (same as `run_file`).
- **Sessions**: outermost `run_source` opens/closes; nested joins; REPL
  holds one until `:reset`. Teardown removes owned `.tl` modules and
  namespace-package parents; third-party imports are kept (unloading
  them broke scipy extension re-import — found by testing).
- **Requester attribution**: the finder serves only the innermost
  below-machinery frame when it is the executing `.tl` module — lazy
  backend imports (dask→cytoolz) and `.py` helpers keep ordinary
  `ModuleNotFoundError` behavior (found by testing).
- **Cycles**: static pre-scan (`plan_cycle`) rejects `TL0312` with
  requester legs before execution; a runtime loading-set backstops
  dynamic imports.
- **Diagnostics**: `TL0310` (missing), `TL0311` (missing symbol, via
  fromlist `ImportError` conversion), `TL0312` (cycle), `TL0313`
  (init failure with chain + cause), `TL0314` (relative/ambiguous);
  `diagnose` hints for `TL0310`/`TL0312`; catalogue regenerated.
- **CLI**: `graph`/`explain`/`profile` preflight walks the transitive
  graph statically (effects attributed per file, cycles refuse).

## Verification

- 31 new tests (`tests/test_modules.py`): sibling/from/alias/dotted,
  precedence, missing/symbol/relative codes, single-init, shared
  identity across consumers, isolation, uid params, composition,
  fitted import, session isolation, retry-after-fix, direct+indirect
  cycles + recovery, effect gate through chains (`TL0313` caused by
  `TL0305`, sentinel file untouched), `:=`-through-import `TL0305`,
  cwd/space paths, REPL, CLI incl. transitive preflight, equivalent
  spellings, bad extension.
- Full suite: **514 green** (483 + 31); 11 M0 app tests green.

## Migrated application (`evaluation/v08/app/`)

`io/` → `inputs/` (stdlib `io` wins dotted resolution — documented
precedence rule in action), plus `pipelines/` stages and
`main_native.tl` (56 lines) fanning in all four stage directories.
`driver.py` (71 + 94-line Structure B) and `main.tl` retained for
comparison.

Numerics identical: coefficient **0.6311** (native `.tl`), 0.6311
(driver), 0.6311 (NumPy reference); 94976 rows, 50 experiments.

## Measurements (100K rows)

| Metric | Native `.tl` | Structure-B driver |
|---|---|---|
| Full run, fresh process | 166 ms | ~same class |
| Peak RSS | 106 MB | 171 MB |
| Entry LOC | 56 | 94 (71 driver + stages) |
| Stage import (fresh) | ~2 ms/file | 1.4 ms total |
| Same-session re-import | 0.1 ms/file | n/a |
| Full same-session repeat | 19 ms (modules cached) | n/a |

Boilerplate removed: stage-loading/timing/wiring scaffolding lives in
the language now. Numerical stage costs unchanged.

## Design findings for the record

1. Cross-file `{M}` ownership needed no changes — uid mechanism held.
2. `from m import f` + `f(data)` under `:=` stays `TL0305` (imported
   names are ordinary bindings, per M1); `>>` composition is the
   declarative cross-module form.
3. Series/scalar mixed arithmetic (`series - dense scalar`) hit an
   unrelated `TL0499` limit during migration — kept out of the app,
   noted, not fixed (not an M2 concern).
4. Dynamic-`__import__` cycles and `.tl` imports from plain Python
   remain out of scope by design.

## M3 recommendation

M3 is warranted but small: `extra=` shadowing semantics (document or
diagnose), structured/dense NaN comparison documentation, import-error
message polish. No package manager, no loader redesign. Then release
validation.
