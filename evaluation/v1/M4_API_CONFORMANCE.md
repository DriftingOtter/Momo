# M4 — API conformance (66 stable, 16 provisional)

## Machine-readable gate

`tests/test_public_api.py` pins the full 82-export surface:

- `test_export_set_exact` — `__all__` equals the 81 pinned names +
  `__version__` (82 total); version is a string.
- `test_signatures_pinned` — `inspect.signature` rendering compared
  verbatim for all 81 callables (annotations quoted under
  `from __future__ import annotations`).
- `test_partition_stable_provisional` — 65 stable + 16 provisional
  sets exactly cover the inventory; every stable name is callable.

Any accidental rename, signature drift, addition, or removal fails
here and forces a conscious contract decision per
`docs/COMPATIBILITY.md` (additive in minor; breakage in major;
codes never renumbered without alias note).

## Stable subset (65 + `__version__`)

Language core (`T`, `Tensor`, `Lazy`, `Operator`, `P`);
math (`sin`–`sqrt`, `diff`, `integrate`, `simplify`);
shapes (`concat`, `stack`, `reshape`, `flatten`, `transpose`);
logic (`logical_and/or/not`); relational (`join`, `sort`, `where`,
`clip`); fitted factories + carriers (`standardize`, `minmax`,
`normalize`, `impute`, `pca`, `FittableOperator`,
`PipelineOperator`); first-order AD (`grad`, `value_and_grad`,
`vjp`); linalg (`svd`, `eigh`, `solve`, `norm`); stats (`cov`,
`corr`, `quantile`); nulls (`is_null`, `not_null`, `fill_null`,
`drop_null`); entry points (`run_source`, `run_file`,
`check_source` 2-arg stable + deprecated plumbing,
`diagnose_source`, `diagnose_file`); policy
(`set/clear/current_policy`); 10 error classes.

Testing basis: the pinned signatures plus the existing suites that
exercise each export's contract (core semantics, fusion/TL0512,
TL0419, module identity, bridge/TL0418, admission/TL1020,
persistence recovery, `tl` project flow). No stable candidate was
found with an unverified core behavior.

## Provisional (16, explicitly not frozen)

`equal`, `not_equal` (dense-vs-structured NaN total-order
divergence); `unique` (order semantics); `over`, `rolling`
(window/grouped edge semantics; `rolling` additionally floor-gated
to `polars>=1.21` for `min_samples`); `einsum`, `tensordot`,
`moveaxis` (advanced algebra); `onehot`, `polynomial` (expansion
conventions); `GradOperator`, `ValueGradOperator`, `VjpOperator`,
`ValueGradResult`, `SVDResult`, `EighResult` (carrier types —
fields not frozen). None removed; none promoted in M4.

## Result

Gate green (3 passed) on the release candidate tree; full suite
639 passed with the gate included.
