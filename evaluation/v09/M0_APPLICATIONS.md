# v0.9 M0 — applications

External-consumer-style evaluation against installed TensorLang 0.8.0
(isolated venv, public API only, workspace outside the source tree).
Sources: `evaluation/v09/apps/`.

## App A — scientific numerical modeling (`appA_scientific`)

Exponential-decay fit (`y = A·exp(−Kt) + C`) on 400 seeded observations.
`models/equations.tl` (model), `models/loss.tl` (MSE), root `main.tl`
fan-in with a 1500-step gradient-descent loop via `value_and_grad`
(multi-`wrt`), `analysis/estimation.tl` (step operator).

- TensorLang params `[2.49993, 0.79422, 0.29395]`, mse `0.006592318568654708`.
- SciPy `curve_fit`: same to ~1e-9; mse identical to 15 digits.
- 4 pytest tests green; runs from installed wheel in a spaced path.
- Entry 31 lines; cold RSS 108 MB.

## App B — structured analytics (`appB_analytics`)

2000 customers / 30000 transactions (true nulls); filter → per-customer
aggregates → join → null handling → single-materialization `to_dense`
→ `standardize` fit → ridge (unpenalized intercept) → validation MSE.

- TensorLang mse `116777.2719754062`, weights match sklearn to ~1e-12.
- Native reference: Polars + `StandardScaler` + `Ridge`.
- 7 pytest tests green (counts, nulls, isolation, determinism, reuse).
- Entry 72 lines; cold RSS 169 MB.

## App C — out-of-core (`appC_outofcore`)

512 MB float64 zarr (8000×8000, 8 MB chunks), 64 MB budget:
`sin(x)+x²` chunk transform → global mean → streaming `write_zarr`.

- Mean `0.9998090895278484` both implementations (16 digits).
- Peak RSS 697 MB vs dask reference 750 MB (same class; interpreter-
  dominated, estimates are not RSS caps per v0.5 docs).
- 4 pytest tests green (shape, values vs reference, repeat, `TL10xx`
  rejection of `sort`).
- Entry 18 lines.

## Practical limitations encountered

- F-A1: operators cannot apply to symbolic args (`TL0499`) — composites
  must be flat or `>>`; equations duplicated across modules.
- F-A2: one file under two import spellings fails (`TL0311`); fixed
  single-spelling layouts + chained re-exports used throughout.
- F-B3: separate `collect()`s of one frame are not row-aligned —
  single materialization required.
- F-B4: `&`-combined boolean masks evaluate sides in backend-defined
  order (counts vary, silently wrong); sequential filters used.
- F-B1: method-named fields need `["..."]` indexing.
- Razor-edge budget admission (`TL1020` at 32.00 vs 32.07 MB model) is
  correct behavior; 64 MB used.
