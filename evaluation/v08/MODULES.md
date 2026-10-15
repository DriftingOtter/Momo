# v0.8 M0 — module and namespace capabilities

How `.tl` execution actually works: each file is translated to Python
and `exec`'d in a fresh namespace (`__name__ == "__tl_main__"`,
`__file__` set). `run_file` temporarily prepends the file's directory
to `sys.path`; `run_source` accepts an `extra=` namespace for
injection. There is no `.tl` loader, cache, or package system.

## The ten questions

1. **Python modules from `.tl`**: YES (`import os`, `from tensorlang
   import T` work natively; sibling `.py` helpers work via `run_file`
   because the file's directory joins `sys.path` during execution).
2. **Another `.tl` file**: YES since M2 (narrow file-based loader;
   see `M2_LOADER.md` and `docs/MODULES.md`). At M0 time there was no
   mechanism: `import operators.calib` raised `ModuleNotFoundError`.
   Resolution is file-relative with Python-precedence; `import` Reproducers
   below describe M0 behavior.
3. **Operators across files**: YES via Python modules (plain `.py`
   defining operators imports fine) and via `run_file` namespaces.
4. **TL operator into Python**: YES — `run_file(...)["calibrate"]`
   returns a first-class `Operator`; applying to `T(...)` works;
   repeated application works.
5. **Relative/absolute imports**: absolute Python imports work;
   relative imports fail (`ImportError: attempted relative import with
   no known parent package` — `__name__` is never package-qualified).
6. **Package layouts**: `.tl` files run from packages only as
   standalone scripts; no package-relative resolution.
7. **Init/caching**: N/A (nothing to cache — each `run_file` executes
   fresh; 0.33 ms first load, 0.18 ms repeat of one stage file).
8. **Namespace isolation**: YES — separate `run_source`/`run_file`
   calls produce independent namespaces (verified: `[1.]` vs `[2.]`).
9. **Circular imports**: N/A (no loader).
10. **Arbitrary working directories**: YES — absolute paths +
    `__file__`-relative data resolution work from any CWD (verified
    from `/tmp` via installed CLI and venv Python).

## Cross-file operator scope (the key architectural question)

Operators composed across separately loaded files behave correctly,
including overlapping parameter names:

- `calibrate` (`{M},{G},{O}`, uid `op5`) and `zscore`
  (`{X},{M},{S}`, uid `op10`) from different files: distinct uids,
  separate application correct (`[102.5, 204.5]`).
- Nested application `zscore(calibrate(m,g,o), mm, s)` correct.
- Expression mixing via `extra=` injection correct (`[-0.95, 1.09]`).
- **Parameter identities are uid-scoped, not file-scoped**: shared
  `{M}` names across files never collide. Module separation does not
  change composition, lookup, graph construction, or snapshots.

## Limitations that matter

- **No module abstraction (closed in M2)**: at M0 time the unit of
  reuse was a Python-reachable
  namespace dict, not a language-level module. Composition logic had to
  live in Python (driver: 71 lines vs native 47).
  → M2 natively imports `.tl` stages (`main_native.tl`, 56 lines);
  see `M2_LOADER.md`.
- **`extra=` injection is shadowed by module-level assignment**:
  a file assigning `DATA_DIR = ...` overwrites the injected value
  (exec order). Injection works only for names the module does not
  rebind — verified reproducer in evaluation notes.
- **Effect gate bypass (P1 — closed in M1)**: `x := evil.grab()`
  executed arbitrary
  I/O under a declarative binding (verified: file written, value
  bound). The syntactic classifier only recognizes known effectful
  names; the runtime check only rejects Lazy/I-O-provenance values.
  Any Python helper returning a plain value bypasses the `:=`/`=`
  discipline entirely. → M1 enforces fail-closed `TL0305`; see
  `M1_EFFECTS.md`.

## Parser vs compiler vs runtime

- Parser: Python-like syntax parses (no module syntax exists to fail).
- Compiler: effect classification is syntactic and name-based
  (the bypass vector).
- Import resolution: pure Python semantics (nothing TensorLang-specific).
- Runtime: namespaces are plain dicts; operator objects transfer
  safely between them.
