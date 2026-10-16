# v0.9 M0 — developer experience

## Coding

Mathematical operators express naturally (`{A} * exp(-{K} * {T}) + {C}`);
`>>` covers unary chains; flat restatement covers the rest (F-A1).
Tensor/Operator behavior predictable once the application-vs-definition
distinction is internalized. Single-materialization and sequential
filters are learnable but cost debugging time (F-B3/F-B4).

## Testing

Plain pytest sufficed everywhere (15 app tests total). Pattern:
`run_file` → retrieve operator → apply to `T(...)` → `collect` →
NumPy/polars asserts; `pytest.raises` with `.code` for diagnostics.
No custom framework, no special setup beyond `pip install tensorlang`.
Probe files for same-session identity checks must live where their
imports resolve (learned via failure).

## Imports and organization

Sibling/dotted/alias/re-export layouts worked; root fan-in is the
working pattern (subdirs cannot cross-import). `__file__`-relative
data paths work from any CWD. Failed dual-spelling sessions can
poison later in-process runs — test suites should isolate via
subprocesses or fresh interpreters for negative import tests (F-A2).

## Diagnostics and tooling

`run/check/diagnose/graph/explain/profile/doctor` all exercised;
`check`/`diagnose`/`graph` give file:line codes with hints.
Controlled failures observed: `TL0310` (missing module),
`TL0311` (missing symbol), `TL0305` (evil helper, located),
`TL0313` (init chains), `TL0502`/`TL0418`/`TL1020` (modeling errors).
`diagnose` executes the program (documented). REPL not stressed
beyond M2 coverage.

## Interoperability

NumPy↔`T` (views), Polars↔`T` (frames/series), SciPy (solve/linalg),
sklearn (reference parity) all worked with explicit conversions.
`run_file` namespaces supply operators to Python; errors propagate
with codes. No internal APIs were needed by any application.
