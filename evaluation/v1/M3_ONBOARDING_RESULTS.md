# M3 — Onboarding results (executed + simulated)

## A. Executed clean-environment validation (`/tmp/m3val`)

Empty venv (Python 3.14) + `pip install /path/to/TensorLang` →
`tensorlang 0.9.0`; both CLIs verified. In
`/tmp/m3val/my projects ünï/falling` (spaces + Unicode):

1. `tl new falling --tensorlang <repo>` → created.
2. `tl sync` → synchronized (venv + lockfile).
3. Reusable `{X}` operators written; imported from `main.tl`.
4. `tl check` → OK; `tl run` → `[100. 95.095 80.38 55.855]`,
   `[0. 9.81 19.62 29.43]` (h(1) = 95.095 exact).
5. `tl add packaging`, `tl add --dev pytest`, `tl sync`.
6. Tests written; **DX-01 observed** (stale template test) → test
   updated per guide → `3 passed`.
7. `tl doctor` → healthy; stale warning reproduced by touching the
   manifest (verbatim warning recorded); `tl sync --locked` green.
8. Argv: `tensorlang run src/args.tl -- hello "a b" héllo` →
   `['--', 'hello', 'a b', 'héllo']`; entry via `tl run --` →
   `sys.argv[3:] == ['hello', 'a b']` (DX-06 recorded).
9. Relocation (`cp -r` minus `.venv`) → `tl sync --locked` →
   identical numerics + `3 passed`.

Library loop (`/tmp/m3lib` → `/tmp/m3clean`): `python -m build
--wheel` (hatchling; `.tl` inside); `tl add <wheel>` (uv recorded
`../numerics-lib/dist/...`); dotted import + compose → `[4. 6.]`;
locked clean-location reproduce green; missing wheel fails with
the artifact named.

Error battery (`/tmp/m3err`, 14 cases): TL0301/0305/0310/0312/
0419/0503/0511/0512/0103 all structured with file:line + hints,
exit 1; `invalid manifest` (not "no project"); pytest/uv
actionable errors, exit 1; stale warning advisory only.

**Result: the published guide flow executes end-to-end (steps
1–14) with only documented behaviors encountered.**

## B. External-consumer study — SIMULATED (no humans available)

Protocol (45 min, think-aloud, `/tmp` sandbox, docs only):
T1 install → T2 `new/add/sync/run` → T3 operator + import →
T4 `test` (expect DX-03 discovery) → T5 fix seeded TL0305 +
stale warning → T6 copy + `sync --locked` reproduce.
Observation sheet: task, time, docs consulted, assumption hit
(DX-01…DX-08), code seen, hint usefulness 1–5, exit correctness,
recovery success, verbatim confusion.

Simulated walk (author acting strictly from the new docs):
T1–T3 unaided; T4 hits DX-03 then DX-01, both resolved by guide
§§5–6 within minutes; T5 TL0305 hint sufficient (5/5), stale
warning understood via §7 (4/5 — "advisory" needed one re-read);
T6 unaided. No task abandoned.

**These are documentation-driven simulation results, NOT human
evidence.** Human-study recruitment did not occur in M3; completion
rates/times above describe the simulation only. The protocol and
observation sheet are ready for real participants in M4 or later.
