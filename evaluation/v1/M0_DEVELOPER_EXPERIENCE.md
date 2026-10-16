# M0 — Developer experience (onboarding, diagnostics, study protocol)

Docs-only walk (README + PROJECTS + MODULES + ERROR_CODES); experiments in `/tmp`.

## 1. Onboarding (9 steps) — undocumented assumptions

1. Install: README `pip install -e .` vs PROJECTS `tl new --tensorlang PATH` (required until index release) — connection easy to miss; first sync without flag fails with raw uv output.
2. `tl test` needs `tl add --dev pytest` first — template ships `tests/` without the dep; README quickstart omits (PROJECTS has it).
3. `sys.argv[3:]` offset documented but surprising (runner entries).
4. Managed venv has no `pip` — external installs need `uv pip --project`; undocumented.
5. `tl new` name rules (ASCII, no spaces/paths) vs `tl init` dirname rules — discoverable only by failing.
6. `sync --locked` CI path vs stale-run warning distinction documented; truncated uv tail hurts first-run diagnosis.
7. `tensorlang doctor` (env report) vs `tl doctor` (project diagnostics) — similar names, no contrast in docs.
Fresh falling-body project (operators/physics.tl + main.tl + pytest) built from public clone in `~/Downloads/test_tensorlang` — full `tl` flow green, proving the documented path works once assumptions are known.

## 2. Diagnostics

`tensorlang run`/`diagnose`: exit 1, `Class[CODE]: loc: message`, human `error[TL0305]…` + `hint:`; `diagnose --format json` nests code at `error.code` (no top-level `code` — schema undiscoverable, P2). `tl run/check` propagate rendering; `tl test` forwards pytest; app-code passthrough; argparse 1/2.
Controlled failures: bad decl TL0305 + suggestions; dims/axes/masks TL0301/0501/0511; unfitted TL0419 at apply; missing module/symbol TL0310/311 + chain; circular TL0312 + chain (static); relative TL0314; TL0315 exactly-one-winner; missing chunked source TL1003; admission family TL1020-24 present — all good.
Uncoded escape: `tensorlang run /nonexistent.tl` → exit 2 raw `FileNotFoundError` traceback, not `TensorLangError` (P1 — fix to coded error). `cli._introspect`/`_summary`/`_binding` defensive `except Exception` pragma-covered, no other escapes observed.

## 3. External-user study protocol (SIMULATED — no humans run)

45-min think-aloud, `/tmp` sandbox: T1 install → T2 new/add/sync/run → T3 operator+import → T4 test (pytest-missing discovery) → T5 fix seeded TL0305 + stale warning → T6 copy + `sync --locked` reproduce.
Observation sheet: task, time, docs consulted, assumption hit (§1), code seen, hint usefulness 1–5, exit correctness, recovery success, verbatim confusion quote. Success: unaided T2+T6; hint comprehension ≥4/5 on TL0305/0310/0419. Label SIMULATED (author-executed); human recruitment not required for M0.

## 4. M0 verdict

Workflow learnable but 7 assumptions need docs; diagnostics strong except nonexistent-file escape. M1/M3: fix escape (P1), document assumptions (P2), run real study in M3.
