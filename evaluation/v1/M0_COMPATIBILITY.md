# M0 — Compatibility (v0.7/v0.8/v0.9 → current) + post-v1.0 policy proposal

## 1. Results

- Preserved examples (`examples/*.tl`, 10 files): 10/10 exit 0 on current impl.
- Apps (`evaluation/v09/apps`, built for 0.8.0): 15/15 pytest pass after generating gitignored artifacts (initial 11 failures = missing `obs.npz/parquet/zarr`, not regressions).
- `__all__`/exports: none changed v0.8→v0.9 (version string only).
- Deltas v0.8→v0.9 (`git diff 763d478..HEAD`): (2) new `TL0419` (unfitted at apply) + `TL0512` (mask mismatch) + `_try_lowered` re-raise + same-domain fusion; (3) `(path,name)` loader identity + no-poisoning; (4) `cli run` argv forwarding (additive); (5) `tl` project manager (~643 lines, additive); (6) `TL0203` wording tweak.
- Classification: intentional-documented (TL0512 fail-closed ex-silent-wrong F-B4; dual-spelling independence ex-F-A2; argv; `tl`; TL0419 timing); correctness-fix (TL0512 propagation; poisoning removal); accidental-regression: none observed; unsupported-detail: wording/plans/RSS/task counts/ulp order/join order.

Migration notes (v0.8→v0.9): `&`/`|` cross-domain → `TL0512` (filter sequentially / single source); unfitted → `TL0419` at call (fit first; catch TL0419 not TL0499); dual spellings → independent (standardize spelling); `tensorlang run file.tl args` now forwards argv; `tl` + `[tool.tensorlang]` purely additive.

## 2. Proposed post-v1.0 compatibility policy (for review, NOT enacted)

Promised (breaking ⇒ major): `.tl` syntax (`:=`/`=`, `{X}`, def/apply/`>>`, explicit `.inplace()`, resolving import spellings); operator semantics (broadcast, `collect()`+snapshot/ownership, `to_dense(nulls=)`, join-key contract, `fit`-returns-new + `TL0419`, 1st-order AD + `TL08xx`, `TL0512` verdict); frozen-A signatures + `run_source/run_file/diagnose_*` shapes; diagnostic code identities + triggers (wording NOT frozen); manifest additive-only + `uv.lock` honored by `--locked`; `tl run/check/test/doctor` exit contract; import resolution/precedence/identity + `TL0310–315`; persistence no-silent-format-change (round-trip tests + migration tool).
Internal (minor/patch): IR shapes, backend plans/lowering, dep floors in range, task counts, RSS/estimates, wording/hints, REPL history, benchmarks, `trusted/injected_names`, chunked-AD deferrals, perf.
Rules: additive exports in minor; corrections ship code+migration note; codes never renumbered without major+alias; Python floor 3.12–3.14 raised only in minor with notice.

## 3. M0 verdict

No accidental regressions; intentional fixes documented with migration notes. M1: adopt policy + publish migration guide; freeze code identities.
