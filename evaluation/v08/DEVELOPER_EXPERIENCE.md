# v0.8 M0 — developer experience

Evaluated with the installed CLI (`tensorlang run/check/diagnose/
explain/graph/profile/doctor`) and the REPL, against
`evaluation/v08/app/`.

## CLI assessment

| Command | Result |
|---|---|
| `run main.tl` | Works from any CWD; prints results |
| `check main.tl` | `OK` — parse+validate without executing |
| `diagnose main.tl` | Runs with enriched diagnostics; lists bound names |
| `graph` / `explain` / `profile` | Work on bindings (`valid` explains as SCAN→PROJECT→SELECTION with row estimates) |
| `doctor` | Reports version, platform, CPU, dependency versions |
| `errors` | Browsable error-code catalogue |
| `bench` | Existing benchmark suite untouched |

## Error attribution (controlled failures)

- **Missing module** (`import no_such_mod_xyz`): `error[PY0000]
  (ModuleNotFoundError)` with file context via `diagnose`.
- **Declaration misuse** (`x := T([1.0]) + 1`): `TLDeclarationError
  [TL0301]` naming the file, line (`/tmp/m0_probe2.tl:2`), the binding,
  and the fix (`use '=' instead`).
- **Bad matmul shapes**: `TLShapeError[TL0502]` with both shapes.
- **Bad reduction axis**: `TLShapeError[TL0511]` with rank/shape.
- **Bad `to_dense` column**: `TLTypeError[TL0418]` naming the column and
  suggesting numeric selection.
- **Wrong arity**: `TLShapeError[TL0503]` with expected count and names.
- **Admission failure**: `TLUnsupportedError[TL1024]` with MB/budget
  numbers and a streaming suggestion.

No case was found where tooling reported only the entry-point file for
an error inside a reusable operator — largely because cross-file
composition happens in Python, so tracebacks point at the driver line
plus the coded TL error. This is honest but thin: there is no
`.tl`-level stack across the Python/TL boundary.

## Gaps

1. **No `.tl` import means no `.tl` stack**: multi-file failures
   surface at the `run_file`/driver line, never as "imported from"
   chains. Acceptable now, but a future loader needs chained
   attribution from day one.
2. **REPL**: usable for inspection (`tensorlang repl`) but evaluates in
   a single namespace; no module-reload concept.
3. **Effect gates preserved**: CLI `run` enforces the same `:=`/`=`
   discipline as the library; the P1 bypass (MODULES.md) is a semantic
   hole, not a CLI hole — it reproduces identically under `run` and
   `run_source`.
4. **`explain`/`profile` scope**: per-binding only; no whole-application
   view. The driver calls `explain()` on one aggregate as a smoke
   signal, not a full plan review.

## Testing

`tests/test_application.py`: 11 tests, plain pytest, ~79 non-comment
lines. Covers reusable operators, cross-file composition, overlapping
params, structured transforms, dense output, fitting + frozen reuse,
laziness (`_has_cached`), coded errors, import independence, repetition,
determinism. Boilerplate per test: fixture `run_file` + `T(...)`
construction + `.collect()` + NumPy assertion — no custom harness.
Suite runs in ~0.4 s.
