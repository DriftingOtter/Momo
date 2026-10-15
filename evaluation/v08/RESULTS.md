# v0.8 M0 — results and findings catalogue

## Numerical equivalence

Temperature coefficient (standardized): **0.6311 TL vs 0.6311 native**
(agreement to 10+ digits: `0.6310866430046445` vs `...6388`). Row count
94976 after null filtering, deterministic across runs. Join order
differs TL-vs-native (both self-consistent, backend-defined) — the app
derives the calibrated column inside the lazy graph, so order never
affects correctness.

## Code complexity

- `.tl` stages: 17 non-comment lines total (`calibration.tl`,
  `transforms.tl`, `sources.tl`, `main.tl`).
- Driver 71 lines vs native 47 — the delta is stage loading, per-phase
  timing, and diagnostics scaffolding, i.e. the cost of "no module
  system" paid once in the host, not per operator.
- Tests: 11 tests, ~79 lines, ~0.4 s, plain pytest.

## Performance (100K rows)

See APPLICATION.md table. Headline: stage loading 1.4 ms total; grouped
aggregates and conversion within ~1.3x of native; `solve` shows a
first-call SciPy-init cost (~126 ms, then <1 ms); cold CLI 0.33 s /
147 MB; driver peak 179 MB. Nothing here blocks modular applications —
startup and import overhead are negligible next to data costs.

## Findings catalogue

### F1 — No `.tl` import loader (M1, severity: high)
Reproducer: `import operators.calib` → `ModuleNotFoundError`.
Expected: some module mechanism. Actual: none; composition must live in
Python. Workaround: Python-hosted Structure B (this evaluation).
Roadmap: narrow loader (P1 below).

### F2 — `extra=` injection shadowing (closed in M3, was low)
Reproducer: `run_source(src, extra={"DATA_DIR": X})` where src assigns
`DATA_DIR` → injected value lost (exec order). Expected: documented
parameterization. Actual: silent shadowing. Workaround: treat `.tl`
files as closed namespaces. Roadmap: loader with explicit parameters
or documented "no parameterization" rule.

### F3 — Effect-gate bypass via plain-valued helpers (M2, severity: P1)
Reproducer: `x := evil.grab()` writes a file and binds (MODULES.md).
Expected: `:=` rejects effectful application. Actual: syntactic
classifier + Lazy/I-O-provenance runtime check both miss plain-valued
Python helpers. Workaround: none (audit helpers). Roadmap: runtime
provenance or allowlist (P0 below) — the only correctness hole found.

### F4 — `==` NaN divergence by representation (M3, severity: medium)
Dense `NaN == NaN` is False (NumPy); series/frame `NaN == NaN` is True
(Polars). Silent. Workaround: `not_null()` + true nulls (app does
this). Roadmap: document + lint (P2).

### F5 — Join/group row order backend-defined (M3, severity: low)
TL join order differs from native polars; both self-consistent.
Workaround: derive inside the graph; sort before comparison.
Roadmap: document (P2). Not a bug.

### F6 — No `.tl`-level import stack (M4, severity: low)
Multi-file failures attribute to the driver line + coded error, never
an import chain. No loader exists to produce chains. Roadmap: chained
attribution with the loader (P1).

### F7 — No packaging story for `.tl` stages (M5, severity: medium)
`__file__`-relative paths work from any CWD and installed CLI, but
distribution is ad-hoc (ship files + driver). No entry points, no
installed-data support. Roadmap: packaging convention (P3).

### F8 — Per-binding `explain`/`profile` only (M4, severity: low)
No whole-application plan view. Workaround: per-stage `explain()`
calls. Roadmap: driver-level plan summary (P3, optional).

No M6 (startup overhead: negligible) or M7 (public API `run_file` /
`run_source` / `T` / operators proved stable under this workload)
findings.

## What worked / workarounds / failed

1. **Worked unmodified**: reusable cross-file operators with correct
   uid-scoped params; bidirectional Python interop; lazy execution;
   fitted-operator reuse; coded diagnostics; CWD independence; plain
   pytest testing; installed-CLI execution.
2. **Required workarounds**: Python-hosted composition (no loader);
   true nulls + `not_null()` (NaN divergence); in-graph derivation
   (join order); `__file__`-relative paths (no package resolution).
3. **Failed**: `.tl`-to-`.tl` imports (no mechanism); `extra=`
   parameterization (shadowing); effect-gate enforcement for
   plain-valued helpers (P1 bypass).
4. **Material restrictions**: F1 (no loader) and F3 (effect bypass)
   are the two items that restrict larger applications; F7 matters at
   distribution time.
