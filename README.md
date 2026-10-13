# TensorLang v0.5.3

TensorLang is a tensor-oriented programming language for general numerical
computing, large-scale data processing, structured datasets, and symbolic
mathematics. It is founded on two computational entities:

- **Tensor** — the universal representation of data: scalars, vectors,
  matrices, N-dimensional arrays, and heterogeneous structured datasets.
- **Operator** — a first-class, reusable computational transformation that
  is defined independently of data, applied explicitly, and composed with
  other operators.

```python
from tensorlang import T

vector := T([1, 5, 12, 20, 30])

square   := {X}**2
shift    := {X} + 5
pipeline := square >> shift      # pipeline(x) = shift(square(x))

result = pipeline(vector)        # explicit application, non-mutating
print(result.collect())          # [149 405 ...] per element

pipeline(vector).inplace()       # explicit in-place execution
print(vector.collect())
```

See [`examples/core_demo.tl`](examples/core_demo.tl) for the full
demonstration program.

## Installation

Requires Python ≥ 3.12.

```bash
pip install -e .          # from the repository root (editable install)
pytest                    # runs the test suite
```

Dependencies (installed automatically): NumPy (dense backend), Polars
(structured backend), SymPy (symbolic engine), SciPy (numerical
routines), Dask + Zarr (out-of-core chunked execution, v0.5).
Development extras (`pip install -e ".[dev]"`): pytest, scikit-learn
(used by ML tests).

v0.2 adds: lazy dataset scans, joins/sorting/unique/conditional/window
operators, single-query lowering into Polars lazy plans, `explain()` /
`profile()` introspection, streaming writes, and a benchmark suite
(`benchmarks/`).

v0.3 adds: first-class fittable operators (`standardize`, `minmax`,
`normalize`, `impute`, `pca`) with sequential pipeline fitting, numerical
linear algebra (`svd`, `eigh`, `solve`, `norm`), statistics
(`cov`, `corr`, `quantile`), tensor algebra (`einsum`, `tensordot`,
`moveaxis`), feature engineering (`onehot`, `polynomial`), and the
versioned language specification (`LANGUAGE_SPEC.md`).

v0.4 adds: reverse-mode automatic differentiation (`grad`,
`value_and_grad`, `vjp`) over existing operator graphs, fitted-affine
differentiation, structured gradients with label preservation, operator
reduction methods, and phase-split profiling.

## Running programs

TensorLang programs live in `.tl` files — Python-compatible syntax plus
the `:=` and `{X}` constructs below:

```bash
tensorlang run examples/core_demo.tl
tensorlang check examples/core_demo.tl    # parse + validate only
python -m tensorlang run examples/core_demo.tl
```

You can also embed the language in Python:

```python
from tensorlang import T, P, run_source

X = P("X")                 # plain-Python spelling of {X}
f = X**2 + 1
print(f(T(4)).collect())   # 17

ns = run_source("from tensorlang import T\ny := T([1, 2])\n")
```

## Language syntax

### Declarative definition: `:=`

`:=` introduces a name whose computational structure is established
declaratively — literal tensor constructors, symbolic operator
expressions, compositions of declared operators, and pure symbolic
constructors:

```python
data     := T([1, 2, 3])
f        := {X}**2 + 1
mask     := ({X} > 10) & ({X} < 100)
pipeline := f >> g
```

TensorLang classifies every right-hand side as **declarative**, **runtime**,
or **effectful** (see `src/tensorlang/frontend/effects.py`):

| Class | Meaning | Examples | Under `:=` |
|---|---|---|---|
| declarative | describes without executing | `T(...)`, `{X}**2`, `f >> g`, `diff(f)`, `T.scan_parquet(...)` | allowed |
| runtime | needs execution | `f(data)`, `t.collect()`, `sin(data)` | rejected, use `=` |
| effectful | I/O or mutation | `T.read(...)`, `t.write_parquet(...)`, `f(x).inplace()` | rejected, use `=` |

Two layers enforce this: a syntactic AST check (exact source locations,
no execution) plus a runtime value check (pending `Lazy` computations and
I/O-provenance tensors are rejected). A lazy scan is a pure source
*descriptor* — declaring one reads nothing:

```python
sales := T.scan_parquet("data/sales/*.parquet")   # declarative, reads nothing
data   = T.read("dataset.parquet")                # effectful, use =
```

A name may only be declared once per lexical scope; use `=` to reassign.
`:=` inside a loop body runs each iteration but is still checked once,
statically.

### Runtime assignment: `=`

`=` binds or rebinds names to runtime values and never applies an
operator by itself:

```python
result = f(data)
data   = T.read("dataset.parquet")
sales["net"] = net(sales)
x = f            # binds the operator itself; NOT x = f(x)
```

### Local operator parameters: `{X}`

Brace-enclosed names are lexically scoped symbolic parameters owned by
their operator definition:

```python
f := {X}**2        # X owned by f
g := {X} + 1       # a different X, owned by g
h := {X} + {Y}     # two-parameter operator: h(a, b)
```

Repeated `{X}` within one definition is one parameter; identical names in
different definitions never collide (internally `(owner, name)` pairs,
shown in diagnostics as e.g. `f_x`). `{X}` is not a global variable or
global SymPy symbol.

### Explicit application and composition

```python
result   = f(data)        # function-call notation only; non-mutating
selected = data[mask(data)]
h := f >> g               # h(x) = g(f(x)), a new first-class operator
```

### Lazy results: `.collect()`

Every non-mutating operation — operator application, arithmetic,
indexing, reductions — builds a pending computation over **snapshots** of
its inputs. `.collect()` executes it:

```python
result = f(data)     # constructs the graph; data untouched
result.collect()     # NumPy array / Polars frame / exact symbolic value
```

Because graphs hold snapshots, a later `.inplace()` on the source tensor
never changes an already constructed lazy expression.

### Explicit in-place execution: `.inplace()`

```python
f(data).inplace()
```

This validates the application (unambiguous single-input operator over a
mutable `Tensor`), evaluates it fully, then **atomically** replaces the
target's logical contents — shape, dtype, or schema may change; on
failure the input is left unchanged. All aliases of the same `Tensor`
object observe the update; rebinding a name does not mutate. Multi-input
applications and plain tensor expressions reject `.inplace()` with a
descriptive error.

## The `T(...)` constructor

| Input | Tensor |
|---|---|
| `T(5)`, `T(3.14)` | rank-zero scalar |
| `T([1, 2, 3])` | vector |
| `T([[1, 2], [3, 4]])` | matrix / N-D array |
| `T({"a": [...], "b": [...]})`, `T(amount=[...])` | structured dataset |
| `T(numpy_array)`, `T(polars_frame)` | wrapped backend data |

Helpers: `T.linspace`, `T.arange`, `T.zeros`, `T.ones`,
`T.read("f.parquet" | "f.csv" | "f.json")` (effectful),
`T.scan_parquet(...)` / `T.scan_csv(...)` (lazy descriptors; globs allowed).

Lazy scans expose `.schema`/`.fields` from file metadata without reading
row data; `.shape`/`.rank`/`len()` are unknown until an explicit execution
boundary (`.collect()` / `.write_*` / `.profile()`). Every operation on a
scan stays lazy — nothing is read until the boundary.

Metadata: `.kind` (`dense`/`frame`/`series`/`scan`), `.shape`, `.rank`,
`.dtype`/`.schema`, `.fields`. Conversion: `.to_numpy()`, `.to_polars()`.
Indexing: slices, integers, Boolean masks, `t.field`, `t["field"]`,
`t[["f1", "f2"]]`. Heterogeneous frames are never silently converted to
homogeneous arrays.

## Operator library

- **Arithmetic**: `+ - * / ** %`, unary `-`, `abs` (NumPy broadcasting).
- **Comparison/logic**: `== != > < >= <= & | ~` (+ `equal`, `logical_and`, …).
- **Selection**: indexing, slicing, Boolean masking, field selection, row filtering.
- **Manipulation**: `.T`/`.transpose()`, `.reshape()`, `.flatten()`,
  `@` (matmul), `concat`, `stack`.
- **Reductions**: `.sum()/.mean()/.min()/.max()/.std()/.var()`, with
  `by=` grouped aggregation for structured data.
- **Mathematics**: `sin cos tan exp log sqrt`.
- **Calculus**: `diff(f)` (symbolic derivative), `simplify(f)`,
  `integrate(f)` / `integrate(f, a, b)` (exact; 0-param constant operator).
- **Data science**: `is_null`, `not_null`, `fill_null`, `drop_null`,
  derived fields (`t["net"] = …`), grouped sums/means.
- **Relational (v0.2)**: `join(left, right, on=, how=)` with explicit keys
  (inner/left/right/full/outer/cross/semi/anti), `sort(x, by=)`,
  `unique(x)`.
- **Row transforms (v0.2)**: `where(cond, a, b)`, `clip(x, lo, hi)`,
  `over(x, by=, op=)` (per-group aggregate broadcast to rows),
  `rolling(x, window, op=)` (sliding windows over 1-D data/series).

## Lazy pipelines, lowering, and introspection (v0.2)

Compatible structured subgraphs — scan → filter → derive → sort → join →
aggregate — lower into **one Polars `LazyFrame` query**, so predicate
pushdown, projection pushdown, and subplan elimination apply across the
whole computation instead of per operator. Anything unlowerable falls back
to node-by-node evaluation with identical results; `explain()` always says
which path applies.

```python
sales := T.scan_parquet("data/sales/*.parquet")

valid := (({D}.amount > 100) & ({D}.status == "completed"))
net   := {D}.amount - {D}.fee

filtered = sales[valid(sales)]     # lazy; nothing read
filtered["net"] = net(filtered)    # lazy WithField; nothing read
summary = filtered.net.sum(by=filtered.region)

print(summary.explain())           # logical IR + physical Polars plan
print(summary.profile())           # measured time/memory/backend
summary.write_parquet("out/revenue.parquet")   # streams; never materializes
```

`explain()` shows the TensorLang logical graph, the physical plan (lowered
Polars plan text or the fallback reason), and node/backend counts — without
executing. `profile()` executes and reports measured wall time, allocator
and RSS memory, backend, result shape, node count, and scan executions.
`Tensor`/`Operator` also expose `explain()`; `Lazy` adds `release()` to
drop cached results.

Ordering: lazy/streaming plans preserve no incidental row order. Order is
guaranteed only where `sort` is applied explicitly; join tests and
benchmarks compare order-insensitively unless sorted.

## Machine learning and numerics (v0.3)

Fittable operators learn parameters from data while staying first-class
operators — same `f(data)` application, `>>` composition, `.collect()`,
`.inplace()`, and `explain()`:

```python
preprocess := impute("mean") >> standardize() >> pca(2)

fitted = preprocess.fit(train)   # runtime boundary; uses `=`, never `:=`
train_result = fitted(train)     # (n_train, 2) frame with pc0/pc1
test_result = fitted(test)       # train statistics reused; never refit
```

Invariants: `.fit` returns a new operator (the original is unchanged);
already-fitted stages are reused without refitting; applying an unfitted
operator is an error naming `.fit`; learned state is immutable and visible
via `.parameters`. Shape-preserving transforms on mixed frames transform
numeric fields and preserve the rest; PCA requires all-numeric input and
returns a `pc0…` frame. Estimators use NumPy/SciPy only.

Numerical library: `svd` (`U, S, Vh = svd(A)`, one shared computation),
`eigh` (square symmetric only), `solve` (square non-singular),
`norm`, `cov`/`corr` (labeled frames for structured input),
`quantile`, `einsum`, `tensordot`, `moveaxis`, `onehot`, `polynomial`.
See `examples/ml_pipeline.tl` and `LANGUAGE_SPEC.md` §10–§12.

## Developer tooling

Errors carry stable codes (`error[TL0301]: …`, catalogued in
`docs/ERROR_CODES.md` and generated from one registry). `tensorlang
diagnose FILE` re-runs a program with fix suggestions (did-you-mean
fields, `:=`/`=` confusion, missing imports) in human or JSON form;
`tensorlang errors [CODE]` browses the catalogue. `Operator.abstract()`,
`Tensor.abstract()`, and `Lazy.abstract()` return JSON-serializable
summaries (kind, params, graph size, backends, learned state, schema)
for debuggers and editor tooling — one introspection API, not three
formatters.

```bash
tensorlang repl                         # interactive playground (:help inside)
tensorlang graph FILE [--var V] [--dot] # expression graphs (runs FILE)
tensorlang explain FILE VAR             # logical + physical plan
tensorlang profile FILE VAR             # measured costs incl. AD phases
tensorlang diagnose FILE [--format json]# enriched error analysis
tensorlang errors [CODE]                # error-code catalogue
tensorlang doctor                       # environment report
tensorlang bench [--sizes small]        # benchmark suite
```

## Out-of-core tensors (v0.5)

Chunking is a physical execution property, not a new operator kind —
operators see one logical tensor regardless of partitioning:

```python
data := T.scan_zarr("large.zarr")       # declarative; reads no data
f := sin({X}) + {X}**2                  # ordinary operator, unchanged meaning
result = f(data)                        # lazy dask graph, chunk-local
average = result.mean()                 # tree reduction, bounded memory
print(average.collect())                # explicit boundary
result.write_zarr("out.zarr")           # streaming write, never materialized
```

`T.scan_zarr` exposes logical shape/dtype plus storage metadata (chunks,
partitions, bytes) via `abstract()` without reading data. Peak working
memory tracks chunk size, not data size (in-flight chunk buffers ×
worker threads) — size chunks to fit the budget; `set_chunk_policy()`
gates admission on modeled working memory. Supported over
chunks: elementwise math, comparisons, where/clip, shape ops, concat/
stack, slices, whole-array reductions, and bounded rank-2 matmul with an
unchunked contract axis. Anything else (sorting, joins, grouping,
fitting, differentiation, masks, partitioned contractions) raises a
coded diagnostic with the reason instead of silently materializing.
`Tensor`/`Lazy` gain `write_zarr()` (staged publication on local
paths); `inplace()` into chunked sources is rejected;
fitting/differentiation over chunked data is deferred. See
`examples/chunked_demo.tl`.

Milestone 2 adds chunk-aware planning: the planner classifies each
operation (chunk-local, tree-reducible, alignment-, rechunk-,
neighborhood-dependent, global), aligns mismatched layouts through
explicit union subdivision (recorded with movement estimates — sides
only ever split, never shuffle), gates plans against conservative task
and block-byte limits, and supports metadata-only transpose/moveaxis,
even merge/split reshapes, and bounded matmuls with an unchunked
contract axis. Anything else raises coded diagnostics (`TL1010`
incompatible layouts, `TL1011` unsafe resources, `TL1012` graph
expansion, `TL1013` global operations) instead of materializing.
`explain()` shows input/output layouts, plan steps, rechunks, and
estimates; peak memory tracks chunk size. See
`examples/chunk_planning.tl`.

Milestone 3 makes memory a planning concern: `set_chunk_policy(
memory_budget_bytes=..., max_workers=...)` configures an estimated
incremental working-memory budget (chunk buffers × workers + output +
graph overhead). Plans admit, admit with reduced concurrency
(workers → 1), or reject with coded diagnostics (`TL1020` budget
failure, `TL1021` unbounded estimates, `TL1022` execution-time pressure,
`TL1023` single-block-too-large, `TL1024` collect-too-large — the last
suggests streaming `write_zarr`, whose outputs may exceed the budget).
`explain()` shows the admission decision, `abstract()` carries the
policy without reading data, and `profile()` reports baseline/peak RSS
alongside the modeled estimate — labeled separately, since estimates
are heuristics, not hard process guarantees. See
`examples/memory_budget.tl`.

Milestone 4 makes execution reliable: every chunked run moves through
an internal lifecycle (planned → admitted → running → completed/failed/
cancelled, with a committing stage for writes), visible via
`execution.last_record()`. Local `write_zarr()` defaults to staged
publication — complete and validate (chunk census + metadata, data
never read) in a staging directory, then an atomic same-filesystem
rename — so failed runs never publish partial outputs and never destroy
a preexisting valid destination (`overwrite=True` replaces only after
its replacement validates; `staged=False` keeps direct streaming with
its documented weaker guarantees). Sources are consistency-checked
against their planned baseline before execution (`TL1030`, report-once
semantics); interruption reports `TL1032`; `TL1031`/`TL1033`–`TL1035`
cover staging, validation, stale-artifact, and cleanup failures with
recovery paths. See `examples/reliable_execution.tl`.

The REPL compiles every line through the real frontend, so `:=`, `{X}`,
and effect rules hold interactively; expression statements display
collected values (side effects execute exactly once). `graph`/`explain`/
`profile` execute the file, then introspect the named binding. Because
execution can perform writes, `.inplace()` mutation, or I/O, these three
commands refuse files with statically detected effectful constructs
unless passed `--allow-effects` (editor integrations must not mutate by
accident).

## Automatic differentiation (v0.4)

Reverse-mode gradients over ordinary operator graphs — no new syntax:

```python
loss := (({P} - {Y})**2).mean()

d_loss := grad(loss, wrt="P")        # declarative operator construction
gradient = d_loss(pred, target)      # lazy; [0.667, 0.667, 2.0] on collect

both := value_and_grad(loss, wrt="P")
value, gradient = both(pred, target) # one shared forward evaluation

v := vjp(loss, wrt="P")
g = v(pred, target, seed)            # explicit cotangent for any output
```

Rules cover arithmetic with broadcast adjoints, math functions,
sum/mean/min/max/std/var, matmul, shape ops, scatter indexing,
where/clip masks, solves, norms, and fitted-affine transforms
(standardize/minmax/normalize/impute/PCA with learned constants).
Sorting, comparisons, grouping, SVD, contractions, unfitted estimators,
and second-order attempts raise precise `TLDiffError`s. Gradients need
scalar losses and floating-point inputs; structured gradients preserve
labels where feasible (all-numeric frames/series). See
`examples/autodiff.tl`, `examples/gradient_descent.tl`, and
`LANGUAGE_SPEC.md` §14.

## Dense vs structured semantics

The same operator means the same thing on both representations:

- Arithmetic on frames transforms applicable **numerical** fields,
  preserving field names, order, and non-numeric fields.
- Field comparisons (`{D}.amount > 100`) yield per-record Boolean masks;
  `sales[mask]` filters rows.
- Whole-frame comparisons yield Boolean frames; combining incompatible
  Boolean results, or using a multi-column Boolean frame as a row mask,
  is an error — combine field-level predicates explicitly.
- Reductions over frames preserve labels (one-row frame); with `by=`
  they group explicitly.
- `@`, `.T`, `.reshape()` on heterogeneous frames are rejected as
  genuinely undefined.

## Symbolic mathematics

`f := sin({X}) * exp(-{X}**2)` keeps a symbolic representation:
`diff(f)` differentiates exactly, `integrate({X}**2, 0, 5).collect()`
returns the exact `125/3` (a SymPy `Rational`, never a silent float).
Applying mathematical operators to numeric data lowers to vectorized
NumPy evaluation. Integrals with no exact closed form raise instead of
approximating. SymPy handles mathematical subexpressions only — never
whole datasets.

## Architecture

```
src/tensorlang/
├── frontend/   lexer.py (token rewrite, no regex) · parser.py (AST + scope checks)
│               scope.py · compiler.py (compile/run .tl source)
├── core/       tensor.py (Tensor, Lazy, T) · operator.py (Operator, P)
│               expression.py (immutable IR) · errors.py
├── runtime/    executor.py (evaluation) · mutation.py (inplace)
│               planner.py (backend selection, constant folding)
│               chunk_plan.py (out-of-core planning, budgets, publication)
│               execution.py (per-run lifecycle records)
├── operators/  arithmetic, logical, indexing, linear_algebra (elementary),
│               elementary, statistics, calculus, data_science
├── backends/   numpy_backend, polars_backend, sympy_backend, dask_backend
└── cli.py      `tensorlang run|check|diagnose|errors|repl|graph|explain|profile|doctor|bench`
```

## Testing

```bash
pytest            # 220+ tests: frontend, effects, dense, operators, mutation,
                  # structured, processing, planning, ML, numerics, autodiff,
                  # symbolic, end-to-end — each checked against NumPy /
                  # Polars / SymPy / SciPy / scikit-learn ground truth
```

## Benchmarks (v0.2)

```bash
.venv/bin/python benchmarks/run_all.py --sizes small medium
```

Compares TensorLang vs native NumPy/Polars on dense chains, operator
application, matmuls, multi-predicate filters, filter+derive, grouped
aggregation, and a full scan→write pipeline, plus snapshot-memory behavior.
See `benchmarks/README.md` for methodology and `benchmarks/results/`
for recorded numbers. Headline results on the reference machine
(Apple Silicon, 8 CPUs, Polars 2.0): dense chains at ~1.2x native at 1M
values (operator application ~1.4x), matmul at parity, structured
pipelines within ~1.2x of native Polars as single lowered queries with one
scan execution, and ten lazy branches over 80MB of dense data adding 0.0MB
(copy-on-write snapshots).

## Snapshots and memory

Lazy graphs snapshot dense inputs as **read-only views** (zero-copy) under
a never-mutate-snapshots discipline; external arrays are copied once at
the boundary so caller-side mutation cannot leak in. Polars frames use
Polars' own cheap clones. `Lazy.collect()` caches; call `.release()` to
drop retained results in long pipelines.

## Behavioral changes since v0.1 (no syntax changes)

- v0.4: `Operator` gains `sum/mean/min/max/std/var` methods for reductions
  inside operator definitions (a field literally named e.g. `sum` needs
  `{D}["sum"]`, same rule as tensors); `grad`/`value_and_grad`/`vjp`
  construct declarative gradient operators; `norm`/`solve` accept operator
  inputs.

- v0.3: `Operator.fit` exists only meaningfully on fittables (helpful
  error otherwise); `.fit(...)` calls are classified runtime (rejected
  under `:=`); `>>` with a fittable side builds a fittable pipeline.
- v0.2: `T.read(...)` results carry I/O provenance: `d := T.read(...)` is now a
  static error (effectful), not a convention. Use `d = T.read(...)`.
- `.collect()` / `.to_numpy()` / `.to_polars()` / `.write_*` /
  `.inplace()` calls are classified as runtime/effectful: they are
  rejected under `:=` at parse time with source locations.
- `Lazy` field assignment (`filtered["net"] = …`) is now a lazy
  `WithField` graph node instead of an eager materialization, so
  scan→filter→derive pipelines lower into one backend query.
- Dense snapshots are zero-copy read-only views (v0.1 copied); observable
  semantics are unchanged.
- `join()` takes `on=None` (required except for `how="cross"`).

## Known limitations (v0.4)

- Reductions `by=` accept a field name, a field series, or a per-row key
  array; multi-key grouping needs explicit frames.
- `stack`/`concat` require uniform representations; mixed dense/frame
  inputs must be converted explicitly.
- A structured field whose name collides with a method (`sum`, `collect`,
  …) is only reachable via `t["name"]`.
- Single-identifier set literals (`{X}`) are always parameters in `.tl`
  files; write `{x,}` or use a list for one-element Python sets.
- Dense evaluation is node-by-node (one temporary per operator, like
  eager NumPy); there is no dense loop fusion yet.
- Float aggregates may differ from native plans in the last bit
  (engine-defined summation order); benchmarks compare those approximately
  after proving identical row membership.
- Whole-frame binary arithmetic over scans is not lowered (it falls back
  to a full read); filter/derive/aggregate pipelines lower fully.
- Autodiff covers the documented rule set; Jacobians, forward-mode JVP,
  higher-order derivatives, SVD/einsum rules, and differentiation through
  fitting are deferred. Integer intermediates compute in float64.
- No GPU, distributed, supervised ML, or ML training loops yet —
  those are future milestones (out-of-core next). The planner
  does backend selection, single-query lowering, and safe constant folding
  only.
