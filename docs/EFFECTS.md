# TensorLang effect boundary (v0.8 M1)

## Guarantee

The compiler rejects unsupported effectful or unclassified operations
within TensorLang **declarative** (`:=`) expressions, before executing
anything:

- Known effectful operations (`T.read`, `write_*`, `.inplace()`) →
  `TL0302`.
- Known runtime operations (operator application, `.collect()`,
  `.to_numpy()`, `.fit()`) → `TL0301`/`TL0303`.
- Unknown/arbitrary Python calls (`evil.grab()`, aliases, bound
  methods, `os.path.join`, `print`, callable objects), including nested
  inside supported constructors → `TL0305`, with file:line and a `=`
  migration hint. No side effect occurs before diagnosis.

Admitted under `:=`: names imported from `tensorlang`, the
`tensorlang` module itself, `:=`-declared names, identity-verified
namespace bindings (REPL/extra), and the internal `{X}` hook.
Attribute/subscript roots resolve by the same rule, so property
getters and `__getitem__` on foreign objects are also rejected.
Method calls on literal receivers (`'hi'.upper()`) are admitted:
builtin methods on literals run no user code. Plain already-computed
values (Tensor aliases, tuples such as `x.shape`) are admissible; only
pending `Lazy` results and I/O provenance count as runtime.
Runtime (`=`) contexts retain full Python interoperability.

## Non-guarantees

This is **not** a sandbox:
- `import evil` and `x = evil.grab()` execute arbitrary code by design.
- Operator dunders (`evil + 1`), comprehension iteration, lambda
  defaults, and f-string `__format__` on foreign objects can run user
  code; only calls and attribute/subscript accesses are classified.
- Third-party native extensions are never analyzed.
- Rejected declarations are diagnosed statically; nothing is executed
  to discover effects, and no `pure=True` annotation is honored.

## Consequences for programs

Compute paths/arguments with `=` before declarative bindings:

```
p = os.path.join(here, "data", "observations.parquet")
obs := T.scan_parquet(p)
```

Shadowing a TensorLang name (`import evil as T`, `T = 5`,
`def sin(...)`) fail-closes that name for later `:=` uses.

## Python-injected bindings (`extra=`)

Injection enters the entry namespace only; imported `.tl` modules
never inherit it. Precedence is deterministic:

- `=` rebinding an injected name: allowed, module wins (assignment is
  explicit override intent).
- `:=` declaring over an injected name: `TL0203` conflict — rename the
  declaration or stop injecting the name.
- Dunder names (`__tl_*`, `__file__`, …) cannot be injected (`TL0203`).
- Injected callables are never implicitly trusted: `x := grab()` with
  an injected `grab` is `TL0305`. Only identity-verified TensorLang
  public objects (e.g. an injected `T`) seed declarative trust.
