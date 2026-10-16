# TensorLang troubleshooting

Each entry names the failing surface, the diagnostic to expect, and
the fix. TensorLang ends where external tools begin: language errors
carry `TLxxxx` codes; project errors carry `tl:` messages with hints;
`uv` owns resolution output (credentials redacted).

## Language errors (`tensorlang run/check/diagnose`)

| Symptom | Diagnostic | Fix |
|---|---|---|
| Runtime call under `:=`, e.g. `r := f(data)` | `TL0301` | bind with `=`, then `collect()` |
| Arbitrary Python under `:=`, e.g. `x := os.getcwd()` | `TL0305` | compute with `=` first (`p = os.getcwd()`), declare from `p` |
| Missing `.tl` module / symbol | `TL0310` / `TL0311` with importing-file dir | check spelling, sibling path (Python packages take precedence) |
| Circular `.tl` imports | `TL0312` + full chain | restructure one-way; re-export through a third module |
| Relative import (`from .x import`) | `TL0314` | top-level absolute imports only |
| Wrong operator arity / operator-to-operator | `TL0503` / `TL0499` | call with the declared `{X}` inputs |
| Unfitted `standardize()` etc. applied | `TL0419` at application | `fit()` first, then apply the fitted operator |
| Masks from different row domains with `&`/`\|` | `TL0512` | filter sequentially or derive both predicates from one source |
| Bad reduction axis | `TL0511` with rank/shape | reduce over an existing axis |
| Missing/unreadable entry `.tl` file | `TL0103` naming the path, exit 1 | check the path; nothing is auto-substituted |
| Threaded overlap of loader sessions | `TL0315` | run sequentially or in separate processes |

## Project errors (`tl ...`)

| Symptom | Diagnostic | Fix |
|---|---|---|
| `tl` in a directory without a project | `no TensorLang project found` | `tl new NAME` or `tl init` |
| Corrupt `pyproject.toml` | `invalid manifest <path>: <toml location>` (never "no project") | fix the TOML; `tl doctor` reports without changing files |
| `tl sync` fails as unsatisfiable | `uv sync failed (exit N)` + trailing context | with `--tensorlang` templates: pass `--tensorlang PATH` (pre-index); otherwise check the named dependency |
| `tl test` complains about pytest | `pytest is not installed` + hint | `tl add --dev pytest` then `tl sync` |
| Template test fails after editing `main.tl` | assertion on old `result` binding | update `tests/test_application.py` to your program |
| Stale warning on `run` | `environment may be stale (... changed)` | `tl sync` to refresh, `tl sync --locked` for strict CI |
| Local wheel missing on reproduce | `uv sync failed` naming the artifact | copy the wheel to the same project-relative location |
| `tl` says `uv` not found | actionable `uv`-missing error, never pip fallback | install uv, keep it on `PATH` |
| Resource rejection | `TL1020`–`TL1024` with estimate vs budget | raise the budget, narrow columns, or accept chunked execution |
| Stale-environment confusion | `fresh` only means manifest+lockfile match the last sync record — not installed-package integrity | re-sync when in doubt; use `--locked` in CI |

## Interop notes

- Third-party Python runs under `=` by design (effectful, not
  declarative, not sandboxed).
- `uv` isolation is dependency isolation, not a security boundary.
- Budgets are modeled working-memory estimates, not hard RSS caps.
- Staged publication is per-rename POSIX atomic with recoverable
  multi-step overwrite — not one continuous transaction and not a
  power-loss durability promise.
