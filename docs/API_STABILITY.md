# TensorLang public API and compatibility status (v1.0 M1)

Intended v1.0 contract proposal. No v1.0 guarantee is active until the
release is approved; categories below are the M1 stabilization
proposal based on v0.9 M0–M1 evidence.

Stability categories — **Stable** (intended for compatibility
guarantees beginning with v1.0), **Provisional** (public and usable,
not in the full stable guarantee), **Experimental** (subject to
change), **Internal** (no public promise).

Signatures verified against the implementation (`inspect.signature`,
v1.0 M1 tree). Stable contracts cover names, signatures, return
kinds, mutation/laziness, and diagnostic codes — never internal IR
classes, backend plans, message wording, or filenames.

## Stable (66 items: 65 functional names + `__version__` release metadata)

Counting convention (locked): 65 functional stable names + 16
provisional names + `__version__` (stable release metadata) = all 82
exports. `test_public_api.py` enforces exactly this partition.

Language core: `:=` declarative vs `=` runtime assignment; `{X}`
file-static checking, file-flat trust, root-based attribute rule,
literal-receiver exemption, plain-value admission, open-by-identity
declarative set, `x = f` aliasing (see `LANGUAGE_SPEC.md` §3);
`TL0305` fail-closed validation.

| name | signature | return / behavior | errors |
|---|---|---|---|
| `T` | `T(data=None, **fields)` | Tensor; copies at graph boundary | `TL0305`, `TL0418` |
| `Tensor` | `Tensor(data=None)` | Tensor (materialized) | `TL04xx` |
| `Lazy` | `Lazy(root, target=None)` | pending; `collect()` materializes + caches; `release()` drops | — |
| `Operator` | `Operator(params, root, uid)` | reusable; `f(data)` applies; `>>` composes; immutable | `TL0417`, `TL0499`, `TL0503` |
| `P` | `P(name) -> Operator` | single-param operator | — |
| `sin/cos/tan/exp/log/sqrt` | `(x)` | Lazy elementwise; declarative over `{X}` | `TL07xx` |
| `diff` | `diff(op)` | derivative operator (declarative) | `TL07xx` |
| `integrate` | `integrate(op, lo=None, hi=None)` | definite integral value/operator | `TL07xx` |
| `simplify` | `simplify(op)` | simplified operator | `TL07xx` |
| `concat/stack` | `(parts, axis=0)` | Lazy shape op, pure | `TL05xx` |
| `reshape` | `reshape(x, *shape)` | Lazy shape op, pure | `TL05xx` |
| `flatten/transpose` | `(x)` | Lazy shape op, pure | `TL05xx` |
| `logical_and/or` | `(a, b)` | Lazy Boolean; same-domain fusion, else `TL0512` | `TL0512` |
| `logical_not` | `(a)` | Lazy Boolean; order-preserving | — |
| `join` | `join(left, right, on=None, how='inner', suffix='_right')` | pure; explicit keys; order backend-defined (no order promise) | `TL0430/431/0510` |
| `sort` | `sort(x, by=None, descending=False)` | pure; deterministic given key | — |
| `where` | `where(cond, a, b)` | Lazy branch, pure | `TL0504/0505` |
| `clip` | `clip(x, lo=None, hi=None)` | Lazy clamp, pure | — |
| `standardize/minmax/normalize/impute/pca` | factories (see signatures) | fittable; `fit` returns new, never mutates | `TL0410-416`, `TL0419` |
| `FittableOperator` | `(kind, config, state=None, uid=None)` | unfitted until `fit`; apply-before-fit → `TL0419` | `TL0419` |
| `PipelineOperator` | `(stages, uid=None)` | ordered stages; fit in order | `TL0419` |
| `grad/value_and_grad/vjp` | `(loss, wrt=None)` | first-order only; VJP seeds; no 2nd-order/chunked | `TL0801-809` |
| `svd` | `svd(x, full_matrices=True)` | `SVDResult`-shaped values | `TL0506/0507` |
| `eigh` | `eigh(x)` | symmetric eigendecomposition | `TL0506/0507` |
| `solve` | `solve(a, b)` | linear solve | `TL0506/0507` |
| `norm` | `norm(x, ord=None, axis=None)` | scalar/vector norm | — |
| `cov/corr` | `(x, rowvar=None)` | reductions, pure | `TL0511` |
| `quantile` | `quantile(x, q=0.5)` | reduction, pure | `TL0511` |
| `is_null/not_null` | `(x)` | null predicates (NaN rules per `docs/COMPARISONS.md`) | — |
| `fill_null/drop_null` | `(x, value)` / `(x)` | explicit missing handling, pure | — |
| `run_source` | `(source, filename='<tl>', extra=None)` | isolated namespace dict; `extra=` entry-only, `=` keeps module precedence, `:=` over injected → `TL0203` | `TL01xx-03xx`, `TL0310-315` |
| `run_file` | `(path)` | as `run_source`; missing/unreadable → `TL0103` | `TL0103` |
| `check_source` | `(source, filename='<tl>')` | `ast.Module`, pure; `trusted_names=`/`injected_names=` deprecated (warn, still honored) | `TL01xx-03xx` |
| `diagnose_source/file` | `(source, filename)` / `(path)` | report dict (`ok`/`error.code`/`suggestions`/`names`/`path`) | coded reports |
| `set/clear/current_policy` | keyword-only policy dict | process-global explicit policy mutation | `TL10xx` |
| 10 error classes | `(message, loc=None, code=None, details=None)` | `ClassName[CODE]: loc: message`; codes stable, wording not | registry |
| `__version__` | `str` | release version (`0.9.0`) | — |

## Provisional (16)

Public and tested, but contract details remain open: `equal`,
`not_equal` (dense-vs-structured NaN total-order divergence);
`unique` (order semantics); `over`, `rolling` (window/grouped edge
semantics); `einsum`, `tensordot`, `moveaxis` (advanced algebra);
`onehot`, `polynomial` (expansion conventions); `GradOperator`,
`ValueGradOperator`, `VjpOperator`, `ValueGradResult`, `SVDResult`,
`EighResult` (result/operator carrier types — fields not frozen).

## Experimental / Internal (no public promise)

`tl build`/`publish` (nonexistent by design); chunked-AD/diff
deferrals; REPL history; benchmark harness; `trusted_names=`
plumbing (deprecated, warns).

## Remaining B/C risks (must clear before any freeze)

Aggregate column naming; grouped-float last-ulp order noise (~1e-12,
tolerance-documented not frozen); `==`-of-masks unfused by design;
method-named-field indexing `TypeError`; REPL `:reset` retention
subtlety; `extra=` `=`-override silent-but-documented;
`to_dense` default null policy; intercept-penalty convention.
C items (F-B4/`TL0512`, F-A2 identity, `TL0419` timing) are resolved;
risk is regression only.

## Policy

Intended v1.0 policy lives in `docs/COMPATIBILITY.md` (proposed,
pending release validation). Additive exports allowed in minor;
behavioral corrections ship with code + migration note; codes never
renumbered without a major + alias note; Python floor follows the CI
matrix (3.12–3.14).
