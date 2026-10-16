# Getting started with TensorLang

This guide takes you from installation to a tested multi-file
TensorLang application. Every command below was executed verbatim
against the current release during M3 validation.

Prerequisites: Python 3.12–3.14 and [`uv`](https://docs.astral.sh/uv/)
on `PATH`. `tl` delegates all dependency management to uv and never
falls back to pip or the global Python.

## 1. Install TensorLang

From a published package index (once available):

```bash
python -m pip install tensorlang
```

Or from a local source checkout or wheel (until the first index
release, this is the expected path):

```bash
python -m pip install /path/to/TensorLang
```

Verify both entry points:

```bash
tensorlang doctor   # runtime environment report
tl --help           # project manager help
```

`tensorlang doctor` reports the runtime (TensorLang version, Python,
backends). `tl doctor` is different: it diagnoses one *project*
(manifest, lockfile, freshness, environment).

## 2. Create a project

```bash
tl new falling --tensorlang /path/to/TensorLang
cd falling
```

`--tensorlang` points at a local checkout until TensorLang is
published to an index; without it, the first `tl sync` fails because
the `tensorlang` dependency is unresolvable. Project names are ASCII
(`tl new` rejects spaces and non-ASCII names); working directories
may contain spaces and Unicode.

Generated layout:

```
pyproject.toml          # [project] deps + [tool.tensorlang] entry
src/main.tl             # application entry point
src/operators/         # your reusable .tl modules live here
tests/                  # ordinary pytest tests
```

Bring up the environment (creates `.venv` and `uv.lock`):

```bash
tl sync
```

## 3. Write a reusable operator

`src/operators/transforms.tl` — `{X}` marks the operator parameter:

```tl
# height(t) = h0 - g t^2 / 2, speed(t) = g t.
height := 100.0 - 0.5 * 9.81 * {X} * {X}
speed := 9.81 * {X}
```

`:=` declares a declarative expression (no execution).
`=` binds a runtime result. Arbitrary Python calls are rejected
under `:=` (`TL0305`) — compute paths with `=` first.

## 4. Import and apply it

`src/main.tl`:

```tl
from tensorlang import T

from operators.transforms import height, speed

times := T([0.0, 1.0, 2.0, 3.0])
hs = height(times)
vs = speed(times)
print(hs.collect())
print(vs.collect())
```

Imports are file-relative (resolved from the importing file, never
`sys.path`); if a Python package of the same name exists, Python
wins. Application is explicit — `height(times)` never mutates
`times` — and `.collect()` materializes the lazy result.

```bash
tl check   # src/main.tl: OK — validates without executing
tl run     # [100. 95.095 80.38 55.855] / [0. 9.81 19.62 29.43]
```

## 5. Add dependencies

```bash
tl add packaging        # runtime dependency
tl add --dev pytest     # test dependency (tests/ needs this; pytest
                        # is NOT preinstalled in new projects)
tl sync
```

`pyproject.toml [project] dependencies` is the source of truth;
`uv.lock` is the resolution record. Project environments contain no
`pip` binary — out-of-band installs use `uv pip --project <root>`.

## 6. Test the application

When you change `src/main.tl`, update the shipped
`tests/test_application.py` — it asserts the template's original
output and will fail otherwise (replace it with assertions about
your program):

```python
"""Falling-body entry-point test."""
import os
import numpy as np
from tensorlang import T, run_file

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_main_result():
    ns = run_file(os.path.join(APP, "src", "main.tl"))
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(ns["hs"].collect()),
                       100.0 - 0.5 * 9.81 * t ** 2)
```

```bash
tl test   # 3 passed
```

## 7. Diagnose and lock

```bash
tl doctor  # manifest ok, lockfile present, freshness, environment
```

If `pyproject.toml` changed since the last sync, `tl run` warns
(`environment may be stale`) and keeps using the existing
environment — it never silently upgrades. `tl sync` refreshes;
`tl sync --locked` is the strict CI path (fails on missing/stale
lockfiles instead of resolving).

## 8. Pass arguments and reproduce elsewhere

Arguments after `--` reach your program (`tl run -- hello` makes
`sys.argv[3:] == ['hello']`; the `--` itself is consumed as the
separator — note `tensorlang run file.tl -- args` instead passes
`--` through verbatim). To reproduce in a clean location,
copy the project *without* `.venv` and re-sync:

```bash
cp -r falling /tmp/repro && rm -rf /tmp/repro/.venv
cd /tmp/repro
tl sync --locked
tl run    # identical output
tl test   # identical results
```

A project that depends on a local checkout or wheel reproduces only
together with that source (copy it to the same relative location).
`tl sync --locked` additionally requires the recorded relative
layout to still resolve: after moving a project to a different
depth, run `tl sync` once to refresh, then `tl sync --locked` to
confirm the refreshed lockfile.

## 9. Consume a reusable library (optional)

Libraries are ordinary Python distributions containing `.tl` files:

```bash
# in the library directory (hatchling or setuptools)
python -m build --wheel          # verify .tl files are inside the wheel:
                                 # unzip -l dist/*.whl
# in the consumer project
tl add /path/to/mylib_tl-0.1.0-py3-none-any.whl
tl sync
```

Then import the dotted path and use the operators like local ones
(distribution names use `-`, imports use `_`: `mylib-tl` → `mylib_tl`):

```tl
from mylib_tl.calibration import calibrate, gain
result = gain(calibrate(data))
```

Clean-location reproduction must include the wheel file at the same
project-relative location (a missing artifact fails `tl sync` with
the file named). Index-hosted distributions work the same way once
published — no TensorLang-specific registry step exists.

## Next steps

- Language rules: `LANGUAGE_SPEC.md` (§3 declarations, §5–§8
  operators and execution).
- Project reference: `docs/PROJECTS.md`.
- When things break: `docs/TROUBLESHOOTING.md`.
- Stability promises: `docs/API_STABILITY.md` (66 stable
  candidates, 16 provisional) and `docs/COMPATIBILITY.md`
  (proposed policy, pending v1.0 approval).
