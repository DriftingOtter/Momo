# v0.9 M0 — findings catalogue

## F-A1 — operators inapplicable to symbolic args (F4; medium)

Applying an `Operator` to `Operator` args raises `TL0499` (same-file
and cross-file alike). Composite operators must be written flat or via
`>>`/calculus combinators; model equations duplicate across modules.
Reproducer: `residual := predict({A}, {K}, {T}, {C}) - {Y}`.
Workaround: flat restatement (used in App A). Action: document the
composition model; consider (not commit to) application-to-symbolic
construction reuse.

## F-A2 — dual-spelling imports fail + poison later runs (F4/F7; high)

**M1 status: FIXED.** Cache keyed by `(path, name)` (each spelling independent, CPython semantics); each spelling
initializes independently (CPython semantics); no poisoning. The
M0 xfail pin is now a passing test. Original report preserved below.

Importing one file under two spellings in a session (bare from a
sibling, dotted from the root — unavoidable in hierarchical layouts)
fails (`TL0311` on a present symbol), and the failed session leaks a
partial `sys.modules` entry that breaks later independent in-process
runs. Reproducer: `evaluation/v09` App A/B layouts (pre-workaround);
pinned by `xfail` in `tests/test_modules.py`.
Workaround: exactly one spelling per file app-wide + chained
re-exports; subprocess isolation for negative import tests.
Proposed correction (validated during investigation, NOT applied to
baseline): key the session cache by `(path, name)` so each spelling
initializes independently, matching CPython dual-spelling semantics.
Action: v0.9 M1 must fix + regression-test + verify no leak.

## F-B1 — method-named fields need `["..."]` (F1/F6; low)

`{D}.count` resolves to a method; comparison raises raw `TypeError`.
Workaround `({D}["count"] > 5)` works (matches `"T"` precedent).
Action: coded diagnostic suggesting indexing.

## F-B2 — aggregated columns keep backend names (F2; low)

Grouped `sum`/`count` yield `amount`/`count`, not user labels.
Workaround: write predicates against actual names (verified via error
listing). Action: document.

## F-B3 — separate collects are not row-aligned (F3; medium)

Two `collect()`s of one lazy frame may order rows differently; pairing
them positionally (features vs labels) silently corrupts. Workaround:
single materialization, split after. Action: prominent documentation +
examples; consider a `sort`-by-key recipe.

## F-B4 — `&`-combined masks misalign silently (F7; high)

**M1 status: FIXED.** Same-domain `and`/`or` fuse into one query (plus `GetItem`-str row expressions and top-level collect fusion);
provably different domains raise `TL0512`. App B uses the combined
predicate (stable 1774). Original report preserved below.

`(A > x) & (B > y)` over joined data evaluates sides in
backend-defined order and combines positionally: counts vary run to
run (1715–1721 vs true 1774). Minimal reproducer: 6-row join gives
{0,1,2} vs truth 2 across 10 runs. Sequential single-predicate
filters are exact (1774 stable). `|` in filter position was exact.
Workaround used. Action: v0.9 M1 must align mask combination or
reject unalignable combinations with a coded diagnostic — silent
wrong results are unacceptable.

## F-C1 — razor-edge admission is correct but startling (F8/F2; low)

32.00 MB budget vs 32.07 MB model rejects a valid streaming write
(`TL1020`). Behavior correct; message actionable. No action beyond
documentation (64 MB used).

## F-D1 — unfitted application defers error to collect (F1; low)

`standardize()(data)` builds `Lazy`, fails at `collect()` with generic
`TL0499`. Workaround: none needed (message names `.fit`). Action:
raise at application with a precise code.

## F-D2 — `check_source` leaks internal params (F9; low)

`trusted_names`/`injected_names` in a public signature. Action: keep
but document as advanced, or namespace under a private wrapper.

No findings in F5 (distribution worked) or F9 beyond F-D2.
Performance (F8): no regressions observed.
