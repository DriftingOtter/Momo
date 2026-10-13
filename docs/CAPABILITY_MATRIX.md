# TensorLang v0.5 capability matrix

How to read this table: **in-memory** execution supports everything the
operator library defines. The **chunked** column lists what the
out-of-core (Zarr/Dask) path lowers without full materialization.
Anything else over chunked sources raises a coded diagnostic (`TL1001`,
`TL1002`/`TL1003` for storage conflicts, `TL1010`–`TL1013`,
`TL1020`–`TL1024`, `TL1030`–`TL1035`) instead of
silently materializing. Collect the inputs explicitly to use the full
in-memory library.

## Elementwise and scalar math — chunked: yes

Arithmetic (`+ - * / ** %`), comparisons, `neg`, `abs`, `sin cos tan exp
log sqrt`, `is_null`/`not_null`, `fill_null`, `where`, `clip`.
Broadcasting across differently chunked inputs aligns through explicit
union subdivision (sides only split, never shuffle).

## Shape operations — chunked: constrained subset

`transpose`/`moveaxis` (metadata-only), even merge/split `reshape`,
`flatten`, `concat`/`stack` with aligned non-axis chunks, integer/slice
indexing. Unsafe chunk-merging reshapes are rejected (`TL1010`).

## Reductions — chunked: whole-array only

`sum mean min max std var` over the whole array (tree-combined).
Grouped reductions (`by=`) require materialization (`TL1013`).

## Contractions — chunked: bounded matmul only

Rank-2 `A @ B` with an **unchunked contract axis** plus task/block
gates. Partitioned contracts (`TL1010`), N-D products, `einsum`,
`tensordot`, `svd`, `eigh` require materialization.

## Relational and windowed — chunked: no (materialize first)

`sort`, `unique`, `join`, `over` (grouped transform), `rolling`
(`TL1013`). Mask/boolean and fancy indexing are rejected (`TL1001`;
masks produce unknown chunk dimensions, repeated positions can explode
chunk sizes).

## ML and differentiation — chunked: no (materialize first)

Estimator fitting/application and `grad`/`value_and_grad`/`vjp` over
chunked data are deferred (`TL1001`/`TL0809`); `collect()` explicitly,
then use the in-memory paths.

## Sinks — chunked: yes

`collect()` (refused when the materialized output alone exceeds the
working-memory budget — `TL1024` suggests streaming), `write_zarr()`
with staged publication on local stores.

## Memory policy — estimates, not guarantees

`set_chunk_policy(memory_budget_bytes=..., max_workers=...)` gates
admission on modeled working memory (chunk buffers × workers + output +
graph overhead). Estimates exclude baseline interpreter/BLAS
allocations, fragmentation, and running native tasks; only the OS can
enforce hard process containment. See `LANGUAGE_SPEC.md` §16.

## Platform-dependent guarantees

| Guarantee | Linux / macOS | Windows | Remote stores |
|---|---|---|---|
| Chunked compute + budgets | supported | supported (best-effort) | n/a (compute is local) |
| Staged publication rename | atomic (POSIX) | works; atomicity POSIX-only claim | unsupported → direct |
| Overwrite replacement | recoverable, not continuously atomic | same, untested | n/a |
| SIGINT cleanup | tested | best-effort, untested | n/a |
| RSS measurement | high-water RSS | `resource` unavailable → `nan` fields | n/a |

Windows is Tier-2: functionality is expected to work but signal handling
and rename atomicity are not validated in CI. Power-loss durability is
not claimed on any platform (rename atomicity ≠ fsync durability).
