# v0.8 M1 — declarative effect enforcement

## Root cause

Two layers guarded `:=`, both allowlisted nothing:

1. The syntactic classifier (`frontend/effects.py`) was a denylist
   over *attribute names* (`collect`, `write_*`, `inplace`, `T.read`).
   Bare-name calls (`evil.grab()`, `f()`, `print()`) and foreign
   attribute calls (`os.path.join`) matched nothing and defaulted to
   *declarative*.
2. The runtime hook (`__tl_declare__`) only rejected `Lazy` results
   and I/O-provenance tensors. A helper returning a plain value
   (`42`) passed — after its I/O had already executed during RHS
   evaluation.

So `x := evil.grab()` wrote the file, bound `42`, and reported
success. Classification must be fail-closed because Python evaluates
the call before any value check can see it.

## Fix (no grammar/ontology change)

- New `TL0305` (registry + catalogue + `diagnose` hint).
- `effects.check_declaration_calls`: every `Call` and every
  attribute/subscript access in a `:=` RHS must resolve to a known
  capability (tl import, `tensorlang` module, `:=` name,
  identity-seeded namespace name, `__tl_param__`); otherwise `TL0305`
  before execution. Known `TL0302`/`TL0303`/`TL0301` paths take
  precedence (verified by existing tests pinning each code).
- `parser`: same-file import tracking with shadowing
  (`import evil as T`, `T = 5`, `def sin` fail-close later uses);
  `=`/for/with/except/walrus rebinding drops trust.
- `compiler`: `namespace_trusted_names` seeds REPL/`extra` names by
  object identity against `tensorlang.__all__` — never by name.
- `diagnose`: `TL0305` suggests `=`.

## Classification matrix (expected → observed after fix)

| Form | Expected | Observed |
|---|---|---|
| `T([...])`, `T.scan_*`, `T.arange` | allow | allow |
| `{X}**2`, `sin({X})`, `diff(f)`, `grad(f)` | allow | allow |
| `standardize()`, `join` (→ runtime Lazy check) | allow→`TL0301` on data | same |
| `f(d)` on declared op | `TL0301` at runtime | `TL0301` |
| `t.collect()/.fit/.write_*`, `T.read` | `TL0302/303` | same |
| `evil.grab()`, nested, `f = evil.grab; f()` | `TL0305`, no effect | same |
| `os.path.join`, `print`, bound/callable objects | `TL0305` | same |
| `evil.box.prop`, `evil[k]` | `TL0305` | same |
| `import evil as T` then `T(...)` | `TL0305` | same |
| `x = evil.grab()` (runtime) | executes | executes |

## Compatibility audit

- 463 baseline tests green; 11 M0 app tests green (coefficient still
  0.6311).
- 20 new tests (`tests/test_declaration_effects.py`): adversarial
  (incl. subprocess isolation proving no pre-diagnosis effect),
  positive, tooling (diagnose/REPL/run_file).
- Correctly-rejected-then-migrated: `os.path.join` nested in scan
  descriptors in 4 examples + 2 M0 app stages → precomputed `=` paths
  (the `release_check.tl` pattern). No test needed weakening.
- Validation overhead: +0.5 ms/file (3.02 vs 2.48 ms on
  `core_demo.tl`); numerical execution untouched.

## Residual limitations

Property-free bare references still bind (`x := evil_obj` binds
without executing — safe); operator dunders on foreign objects,
comprehension iteration, lambda defaults, and f-string formatting can
run user code (documented in `docs/EFFECTS.md`, not classified).

## M2 recommendation

Proceed with the narrow `.tl` loader: resolve, compile through this
frontend (same `TL0305` rules, no shortcut), execute in an isolated
cached namespace, reject circular imports with a coded diagnostic.
Keep uid-based parameter identity untouched.
