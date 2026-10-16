# M0 — Numerical reliability (structured + AD + fitted ops)

Env: numpy 2.5.4, scipy 1.18.1, polars 2.0.0. No code changes.

## 1. Structured / mask domains

- Six-row fused mask: selected identity exactly `k=[2], a=b=10.0`, True-count 1, stable ×5 collects; OR/NOT/reversed/nested/reordered all correct; `explain()` lowered (no fallback) — pass.
- Fail-closed `TL0512`: cross-source equal-length, `where`-built (unanalyzable) domains — all raise, never positional zip — pass. Suite `test_mask_domains` + `test_missing_values`: 23 passed.
- Preserved: dense positional `&`/scalar broadcast; chunked `scan_zarr` path (`_NotLowerable`, fusion never touches); scalar reductions contribute no domain (`_row_roots(Reduce,by=None)→[]`); Series(Boolean) result types — pass.
- Nulls: Kleene `[False,None,False]` through fusion; NaN==NaN dense False vs structured True (total-order), null→None; `is_null` dense-NaN True vs structured-NaN False/null True; `== None` rejected both (`TL0499`) — by design, keep `is_null` guidance.
- `to_dense`: nullable-int → float64 NaN (`nulls="nan"`); `nulls="error"` → `TL0418`; grouped mean skips nulls — pass.
- App B: 7/7 pass; enriched 21954, feat 2000, clean 1950 (50 null dropped), **big_ticket 1774** ×3 stable, mse 116777.27197540613 vs ref 116777.2719754062 (rel ~1e-9) — pass.

## 2. Numerical / AD

- Arith/matmul/transpose/broadcast/reductions exact vs NumPy (max abs err 0.0); `solve` vs `np.linalg.solve`, `svd` vs scipy — allclose pass.
- Finite differences (central 1e-6, rtol 1e-4): x³, sin·exp(−x²), norm pass; repo 30+ op sweep passes.
- Reverse-mode: multi-wrt order, broadcast/matmul grads, solve+norm analytics, VJP seeds + shape rejection, pipeline grad 12.0, structured grads keep labels, `value_and_grad` single forward — pass.
- Fitted: `standardize` grad/VJP = 1/scale; unfitted → `TL0419` at apply; PCA VJP shape ok — pass.
- Clean rejections: 2nd-order → `TLDiffError`; chunked-input grad → `TL0809`; comparisons/sort/unique/einsum, int/bool/complex, unfitted-in-graph — pass.
- Suites: autodiff+dense+linalg+structured+reductions+chunked+symbolic 171 passed; chunk_plan+chunked+symbolic+ml 74 passed. (Benign `RuntimeWarning: invalid divide` from intentional singular-matrix test.)

## 3. Silent-ambiguity risks

1. Join row order backend-defined, varies run-to-run; fusion guarantees within-evaluation correspondence only — keep appB materialize-together workaround (F-B3 load-bearing).
2. `x==x`/`x==NaN` missing-filter diverges by representation — keep gated behind `is_null`/`not_null`.
3. Tolerance policy: exact for integer/shape paths; allclose (rtol 1e-4/atol 1e-6) for float/AD; grouped-float last-ulp order noise ~1e-12 documented, not frozen.

## 4. M0 verdict

No silent incorrect numerics found; fail-closed verdicts hold. M1: lock in regression tests for TL0512/TL0419/1774 identities; publish tolerance policy; no AD expansion.
