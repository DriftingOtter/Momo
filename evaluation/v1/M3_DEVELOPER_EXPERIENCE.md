# M3 — Developer experience findings

Validation environments: `/tmp/m3val` (clean install from source,
spaced + Unicode paths), `/tmp/m3lib` + `/tmp/m3clean`
(library-consumer loop), `/tmp/m3err` (14-case error battery).
No human participants were available; §6 is a labeled simulation
(see `M3_ONBOARDING_RESULTS.md`). No production code changed in M3.

## Findings catalogue

| ID | Workflow | Expected | Actual | Kind | Sev | Status |
|---|---|---|---|---|---|---|
| DX-01 | edit `main.tl`, run `tl test` | tests track the program | shipped `tests/test_application.py` asserts template output (`result` binding) and fails | docs | P2 | addressed: guide §6 instructs updating the test |
| DX-02 | `tl run -- <file> args` | run that file | `tl run` takes no file operand; extras go to the entry program's argv | docs | P2 | addressed: guide §8 + `tl run --help` (no file arg exists) |
| DX-03 | first `tl test` | tests run | needs `tl add --dev pytest` + `tl sync` first | docs | P2 | addressed: guide §5, troubleshooting row |
| DX-04 | first `tl sync` (pre-index) | resolves | `tensorlang` unresolvable without `--tensorlang PATH` | docs | P2 | addressed: guide §§1–2, README snippet |
| DX-05 | relocate a wheel-dependent project | portable copy | uv may record an absolute source path (M1 e2e) or a relative one (M3 CLI flow) — portable only with the artifact copied along | docs | P2 | addressed: guide §§8–9 copy-together rule |
| DX-06 | `--` separator handling | uniform | `tl run --` strips `--`; `tensorlang run file --` passes it through | docs | P2 | addressed: guide §8 note |
| DX-07 | library naming | one name | distribution `mylib-tl` vs import `mylib_tl` (PEP 503 normalization) | docs | P2 | addressed: guide §9 |
| DX-08 | `tl doctor` vs `tensorlang doctor` | distinct scopes obvious | similar names, different concerns | docs | P2 | addressed: guide §1, README, PROJECTS rules |

No P0/P1 defects found. The M1 contract decisions held under
docs-driven use; every battery error produced the documented coded
diagnostic (TL0301/0305/0310/0312/0419/0503/0511/0512/0103,
`invalid manifest`, pytest/uv hints, stale warning, TL1020-family
covered by M1 suite).

## Seven M0 assumptions — resolution map

1. Install source (index vs local) → guide §1, PROJECTS sources.
2. `tl add --dev pytest` → guide §5, troubleshooting.
3. `sys.argv[3:]` forwarding → guide §8, PROJECTS rules.
4. uv delegation (no pip; `uv pip --project`) → guide prereq + §5.
5. ASCII project names vs free working dirs → guide §2, PROJECTS.
6. Changed manifest/lockfile vs existing env; `--locked` for CI →
   guide §7, PROJECTS freshness.
7. `tensorlang doctor` (runtime) vs `tl doctor` (project) →
   guide §1, README, PROJECTS.

## API documentation audit (66 stable / 16 provisional)

- Stable table signatures re-verified against `inspect.signature`
  (unchanged by M3 — no production changes); `check_source(source,
  filename)` stable form + deprecated plumbing documented in
  `API_STABILITY.md`, `COMPATIBILITY.md` §6, and the `DeprecationWarning`.
- Provisional boundary explicit (NaN divergence, order sensitivity,
  window semantics, carrier types); no doc implies a universal
  guarantee (policy header: "No v1.0 guarantee is active").
- Composition-order claims re-executed (`square >> shift` → 19,
  `f >> g` → 12); NaN/join-order/fitted/budget/publication examples
  match M1 wording; no example relies on accidental ordering or
  unsupported effects.

## Troubleshooting + structure

- New `docs/TROUBLESHOOTING.md`: 11 language + 10 project rows,
  each with diagnostic + fix; sandbox/budget/atomicity limits
  stated as limits.
- One contract per topic: guide links to authoritative docs
  (PROJECTS, LANGUAGE_SPEC, TROUBLESHOOTING, API_STABILITY,
  COMPATIBILITY); all relative links checked (no broken targets).
- Performance: no engine changes; CLI/discovery edits are M1's and
  off the numerical path (full suite 633 green, apps 15/15).

## Recommendation for M4

M3 acceptance criteria met (guide executes cleanly; assumptions
addressed; library loop documented; boundaries clear; no P0/P1).
M4 should: portability lint, dependency-floor installs, release
script, 66-contract CI matrix validation, and a real-human study
using the M3 protocol if participants become available.
