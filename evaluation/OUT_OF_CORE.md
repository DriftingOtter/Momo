# Workload 3 — out-of-core z-score normalization (Dask/Zarr)

`evaluation/workload_ooc.py`, driver `evaluation/run_ooc.py`, results
`evaluation/results/out_of_core.json`. Two-pass (mean/std reductions,
then streaming staged write, then verification reductions); datasets
exceed budgets by design; native Dask with identical chunks and worker
caps (`compute(num_workers=…)`).

## Correctness — all equivalent

Mean/std statistics, post-write mean≈0/std≈1 (1e-9), and staged outputs
match native Dask arrays on every config. A too-tight budget records a
rejection (TL1024/TL1020 path, exercised in the M5 benchmark matrix),
never a silent fallback.

## Performance (medians)

| Config | TensorLang | Native Dask | Ratio | TL RSS | NP RSS | Admission |
|---|---|---|---|---|---|---|
| 64MB data, 2.56MB chunks, 64MB budget, w4 | 0.333 s | 0.291 s | 1.14x | 342 | 274 | admitted, w4 |
| 256MB data, 2.56MB chunks, 16MB budget, w4 | 1.751 s | 0.991 s | 1.77x | 333 | 277 | admitted, w4 |
| 256MB data, 0.5MB chunks, 16MB budget, w2 | 3.591 s | 3.390 s | **1.06x** | 172 | 100 | admitted, w2 |

(MB for RSS columns.) Smaller chunks + fewer workers reach near-parity
at lower peak memory — the chunk-size/throughput trade-off documented
since M2, now quantified end to end. The 1.77x middle row concentrates
in the two reduction passes plus validation; the streaming write itself
is at staged-overhead levels (~1–7%, see M5 benchmarks).

## Expressiveness (23 LOC TL vs 12 native)

Native Dask is terser here: TensorLang needs explicit budget setup,
separate reduction statements, and an explicit write call where Dask
chains lazily. The `explain()`/admission record compensates with
inspectability native lacks (recorded outcome + workers per run), but
line-for-line this workload favors the incumbent.

## Evaluation-found defect (fixed on this branch, uncommitted)

Re-reading own just-published output later in a long-lived process
tripped TL1030 on the mtime change from the workload's own write
(report-once would have demanded a pointless retry). Successful
publishes now establish the source baseline (`_baseline_own_output`);
external modifications after our publish are still detected. All 355
release tests still pass. Recommended for a v0.5.4 backport or v0.6.

## Verdict

Reliable and competitive: admission decisions correct on every config,
streaming outputs exceed budgets safely, staged publication intact.
v0.6 candidates: partitioned contractions and data-dependent indexing
remain correctly rejected (hypothesis to validate against heavier
algebra below); write-path verbosity (§7-F7).
