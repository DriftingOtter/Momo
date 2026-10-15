# v0.8 M0 — Python interoperability

Tested in both directions against the v0.7.0 baseline (Python 3.14,
NumPy 2.5.4, Polars 2.0.0, SciPy 1.18.1).

## TensorLang using Python

`.tl` files are translated Python, so the full Python import system is
available: `import os`, `import numpy as np`, `from tensorlang import
T` all work. Verified crossing values:

| Value | Result |
|---|---|
| NumPy array → `T(...)` | Zero-copy view semantics; dtype/shape preserved (`float64`, `(4, 2)` asserted in driver) |
| Polars DataFrame/Series → `T(...)` | Accepted; frame ops available |
| Python scalars → `T(3.0)` | Accepted |
| `numpy` ops inside `.tl` | Work (`np.arange` probed) |
| Sibling `.py` helpers | Importable via `run_file` (file dir joins `sys.path` during exec) |

No implicit conversions change mathematical meaning; unsupported
objects fail loudly at the boundary (e.g. `to_dense` on a `String`
column raises coded `TL0418`).

## Python using TensorLang

The public surface is `run_file` / `run_source` returning a plain
namespace dict:

- Load stage: `run_file("operators/calibration.tl")["calibrate"]`
  returns a first-class `Operator`.
- Apply to `T(...)` inputs, `.collect()` results — repeatedly (verified
  in-process reuse, no state leakage).
- Structured exceptions propagate with `.code` attributes
  (`TL0502` matmul, `TL0418` bad column, `TL0301` declaration misuse),
  catchable with ordinary `try/except` + `pytest.raises`.
- No custom test framework needed: `tests/test_application.py` (11
  tests) uses plain pytest; boilerplate per test is one `run_file`
  fixture plus `T(...)` construction.

## Guarantees observed

- **Dtype/shape**: preserved across the boundary; driver asserts
  `float64` / 2-column dense output.
- **Ownership/copying**: NumPy inputs are viewed, not copied, unless an
  op requires it; no aliasing hazards observed in repeated application.
- **Lazy vs eager**: graphs stay lazy until `.collect()`; `_has_cached`
  flag confirms laziness in tests.
- **Mutation isolation**: separate `run_file` calls give independent
  namespaces and distinct operator uids.
- **Exception translation**: backend `ValueError`s are wrapped in coded
  `TL*` errors at `collect()` time (matmul mismatch → `TL0502`).
- **Scope isolation**: `{X}` params are uid-scoped; cross-file reuse
  with overlapping names verified collision-free.
- **Resource cleanup**: no handles leak across `run_file` calls in the
  evaluation workload; chunk-policy state is explicit
  (`set/clear_chunk_policy`).

## Limitations

- Callbacks: arbitrary Python callables cannot be embedded in lazy
  graphs (purity/memory-planning risk) — and should not be; the driver
  keeps orchestration in Python and math in operators.
- No packaging story for `.tl` stages: distribution is "ship the `.tl`
  files + a Python driver", with `__file__`-relative paths doing the
  work. No entry-point or installed-data support.
- `extra=` precedence is deterministic since M3 (`:=` → `TL0203`,
  `=` keeps module precedence; see
  MODULES.md) — drivers must treat `.tl` files as closed namespaces,
  not parameterized templates.
