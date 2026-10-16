# TensorLang Language Specification — v0.7.0

This document defines TensorLang independently of its backing libraries
(NumPy, Polars, SymPy, SciPy). Implementations must honor the observable
semantics below; backend choice, plan shape, and performance may vary.

Normative keywords: **must**, **must not**, **should** (recommended),
**may** (permitted).

## 1. Program structure

A TensorLang program is UTF-8 source in a `.tl` file using Python-compatible
syntax plus the constructs in §3–§5. The frontend translates `.tl` source to
an executable form via token-stream rewriting (no regular-expression
substitution), then validates declarations on the abstract syntax tree with
exact source locations. Ordinary Python control flow (`for`, `if`, `while`,
`def`, `import`, `with`) retains Python semantics.

`{X}` braces containing exactly one bare identifier denote a symbolic
parameter (§5). All other brace uses (dict/set displays, comprehensions,
f-string fields) retain Python meaning.

## 2. Values

- **Tensor**: an eager, mutable handle to data of kind `dense` (N-D array),
  `frame` (heterogeneous named fields), `series` (single named field), or
  `scan` (lazy source descriptor, §9).
- **Operator**: a reusable transformation with named parameters and an
  immutable expression graph. Subtypes: stateless operators,
  **fittable** operators (unfitted estimators), **fitted** operators
  (immutable learned state), and **pipelines** (ordered stage lists, §10).
- **Lazy**: a pending computation over snapshotted inputs (§7).
- Plain Python values (numbers, strings, lists, …) behave as in Python.

## 3. Declarative definition (`:=`)

`NAME := expr` introduces `NAME` in the current lexical scope
(module/function/class; control-flow blocks share the enclosing scope).
Redeclaring in the same scope is an error; shadowing an outer scope is
allowed; `=` rebinding afterwards is allowed. Declaration checking is
static: a `:=` inside a loop body is one syntactic declaration, so it
may execute once per iteration without violating the redeclaration rule
(the rule forbids two `:=` bindings for one name in one scope, not
repeated execution of one binding).

`expr` is classified syntactically **and** by value. The trusted-name set
is file-flat: any name introduced by `:=` anywhere in the same file
authorizes later `:=` uses of that name in the static check (uses from
unrelated scopes still fail at runtime with `NameError` if the name is
not actually bound there).

| Class | Meaning | Examples |
|---|---|---|
| declarative | describes without executing | `T(...)`, `{X}**2`, `f >> g`, `diff(f)`, `T.scan_*` |
| runtime | needs execution | `f(data)`, `t.collect()`, `op.fit(data)` |
| effectful | I/O or mutation | `T.read(...)`, `t.write_*`, `o(x).inplace()` |
| plain value | already-computed data | `y := x` (Tensor alias), `y := x.shape` (tuple) |

`:=` **must** reject runtime and effectful expressions; `=` **must** be used
for them. Plain (non-`Lazy`, non-I/O-provenance) values are admissible:
only pending computations and I/O provenance count as "runtime" for the
value check. `x = f` binds the operator itself by alias (operators are
immutable, so aliasing is indistinguishable from copying) and **must
not** apply it.

Calls are fail-closed: every call (and every attribute/subscript
access) in a `:=` right-hand side **must** resolve to a known declarative
capability — a name imported from `tensorlang`, the `tensorlang` module
itself, a name introduced by `:=` in the same file, or the internal
parameter hook. The check is root-based: any attribute/subscript chain
whose roots are all trusted (declared, imported, or literal) is admitted
statically, and the value check then decides (with `x := T([1,2])`,
`y := x.shape` admits a tuple while `y := x[0]` fails `TL0301` at
runtime; with `x = T([1,2])`, both are `TL0305` because the root itself
is a runtime binding). Method calls on literal receivers
(`'hi'.upper()`) are admitted because builtin methods on literals run no
user code. Arbitrary Python calls (`evil.grab()`,
aliases, bound methods, `os.path.join(...)`, builtins such as
`print`/`len`) are rejected
with `TL0305` **without executing**, including when nested inside a
supported constructor: `T.scan_parquet(os.path.join(...))` **must** be
written with a precomputed runtime path (`p = os.path.join(...); s :=
T.scan_parquet(p)`). The declarative set is open by identity, not closed
by example: any
name imported from `tensorlang` (including `grad`, `P`, `T`) is trusted;
data-dependent calls whose class is statically
unknowable (e.g.
`sin(x)`: declarative for operator input, runtime for tensor input) are
decided by the value check: pending `Lazy` results and I/O-provenance
tensors are rejected. A lazy scan descriptor **must** be accepted (it reads
nothing).

Python-injected bindings (`run_source(..., extra={...})`) enter the
entry namespace only. Precedence is deterministic: `=` rebinding keeps
module precedence; `:=` declaring over an injected name is a `TL0203`
conflict; dunder names are rejected outright. Injected callables are
never implicitly trusted under `:=` (identity-based M1 seeding
excepted). Imported `.tl` modules never inherit `extra`.
This is effect enforcement, not sandboxing (see `docs/EFFECTS.md` for the
exact boundary and residual risks).

## 4. Runtime assignment (`=`)

`=` binds or rebinds names to runtime values. Assignment itself **must
never** imply operator application. `t["field"] = v` on a materialized
tensor assigns eagerly; on a pending computation it extends the graph
lazily (§7).

## 5. Parameters (`{X}`)

Brace parameters are lexically scoped to their operator definition.
Identity is the pair `(owner, name)` where `owner` is unique per
definition occurrence: repeated `{X}` in one definition is one parameter;
identical names in different definitions never collide. Diagnostics render
`f_x`-style names; identity **must not** rely on rendered names. `{X}`
**must not** be a global variable or global symbolic name. Single-parameter
operators are primary; multi-parameter operators bind positionally.

## 6. Application and composition

- Operators apply only via call notation `f(data)`. Arity mismatches and
  operator-to-operator application are errors.
- `f >> g` composes left-to-right (`h(x) = g(f(x))`) for unary-compatible
  operators, preserving independent parameter scopes in a new operator.
  Composing with a non-operator, or a non-unary right side, is an error.
- `f(data)` **must not** mutate its input and **may** return a pending
  computation.

## 7. Laziness (`.collect()`)

Every non-mutating operation **should** return a pending graph over
immutable snapshots of its inputs; `.collect()` executes it. Results:
dense arrays (rank-zero becomes a scalar), structured frames/series,
scalars, and exact symbolic values as-is. Graphs built before a mutation
**must not** change when the source is later mutated (snapshot semantics).
`Lazy.collect()` **may** cache; `.release()` drops the cache.

## 8. In-place execution (`.inplace()`)

`f(data).inplace()` is allowed only for an unambiguous single-input
operator application over a mutable `Tensor` (scan descriptors excluded).
The implementation **must** evaluate fully before touching the target,
then replace the target's logical contents atomically (shape, dtype, and
schema **may** change); on failure the input **must** be unchanged.
Aliasing is by object identity: all references observe the update;
rebinding never mutates. Multi-input or non-application graphs **must**
reject `.inplace()` with a descriptive error.

## 9. Tensors

- `T(...)` constructs from scalars, sequences, mappings (→ structured),
  backend arrays/frames, or keyword fields. Rank-zero scalars, vectors,
  matrices, N-D arrays, and heterogeneous frames are supported.
- Helpers: `linspace`, `arange`, `zeros`, `ones`; `read` (effectful);
  `scan_parquet`/`scan_csv` (declarative descriptors; globs and backend
  options allowed).
- Metadata: `kind`, `shape`, `rank`, `dtype`/`schema`, `fields`.
  Scans expose schema/fields from metadata only; `shape`/`len` are unknown
  until an explicit execution boundary (`collect`, `write_*`, `profile`).
- Indexing: integers, slices, Boolean masks (length-checked), field names,
  field lists. Heterogeneous frames **must not** be implicitly converted
  to homogeneous arrays. Conversions (`to_numpy`, `to_polars`) are explicit
  and **must** reject genuinely undefined cases.
- Reductions (`sum mean min max std var`) accept `axis=` (None, int,
  negative int, or tuple) and `keepdims=` with NumPy shape semantics;
  invalid axes report `TL0511` with operator, axes, rank, and shape.
  `axis=` over frames is rejected as ambiguous; over series it is
  restricted to full reductions.
- Transpose: `.T` works on operators (`{X}.T`), dense tensors, and lazy
  values; a structured field literally named `"T"` stays reachable via
  explicit string indexing (`X["T"]`). Transposing heterogeneous frames
  is rejected (`TL0901`).
- Structured-to-dense bridge: `frame.to_dense(columns=[...],
  nulls="nan"|"error")` builds a lazy dense representation without
  executing. Column order preserved; unknown/duplicate columns,
  non-numeric dtypes, and unknown policies raise `TL0418`; `nulls="error"`
  fails on any missing value while `"nan"` promotes per NumPy rules
  (nullable-int to float64). Row count stays unknown until execution.
  Memory admission covers conversion working memory only, after the
  upstream structured operand materializes: the bridge first evaluates
  its upstream frame, then admits the dense-conversion estimate (never
  estimated from unknown cardinality) at the execution boundary
  (`TL1020`). The conversion budget is therefore not a bound on
  upstream structured execution, nor a hard process-RSS cap (RSS
  baseline, BLAS threads, and fragmentation are outside the model).
  Differentiating through a conversion is deferred with a diagnostic.
- Dense snapshots are zero-copy read-only views; external arrays are
  copied once at the graph boundary. Frames use cheap clones.

## 10. Fittable operators and pipelines

- `standardize()`, `minmax(lo, hi)`, `normalize(norm)`,
  `impute(strategy[, fill_value])`, `pca(n_components[, whiten])` construct
  **unfitted** unary operators (invalid configuration fails at
  construction).
- `.fit(data)` is an explicit runtime boundary: it materializes `data`,
  learns parameters, and returns a **new** operator; the original is
  unchanged. Already-fitted stages are reused without refitting, so test
  data can never overwrite training statistics.
- Applying an unfitted operator is an error naming `.fit`. Learned state
  is immutable and exposed via `.parameters`; application input is never a
  statistics source.
- `>>` involving fittables builds a **pipeline** (ordered unary stages,
  plain or fittable). `pipeline.fit(data)` fits unfitted stages in order,
  threading each stage's transformed output into the next. Applying a
  fitted pipeline reuses learned state without refitting.
- Estimators consume numeric feature matrices: dense 1-D/2-D, single
  series, or all-numeric frames/selections. Shape-preserving transforms on
  mixed frames transform applicable numeric fields and preserve the rest
  (names, order); dimensionality-changing transforms (PCA) require
  all-numeric input. Missing fitted features are errors; extra columns
  pass through untransformed. Dense width mismatches are errors.
- PCA returns a rank-two frame with `pc0…` coordinates. Fitted application
  is non-mutating and lazy-compatible; `.inplace()` follows §8 (output
  kind may differ from input kind).
- Fit data with missing/infinite values is rejected (except `impute`
  itself); constant columns have defined outputs (standardize → 0,
  min-max → range lower bound); zero-norm rows stay zero; PCA requires
  `n_components ≤ min(n_samples, n_features)` and `n_samples ≥ 2`.
  Statistics use population conventions (variance `ddof=0`); PCA variance
  uses `n − 1`.

## 11. Structured operator semantics

The same operator has the same logical meaning across representations:

- Arithmetic/broadcasting on frames transforms applicable numeric fields,
  preserving names, order, and unrelated fields.
- Field comparisons yield per-record Boolean masks; `t[mask]` filters rows.
  Whole-frame comparisons yield Boolean frames, which **must not** be
  silently merged into record conditions.
- Boolean combination (`&`, `|`) over one logical row domain fuses into
  a single relational query (deterministic, order-independent);
  masks from provably different row domains are rejected (`TL0512`) —
  equal length is not proof of equal row identity. `~` preserves order.
  Separately materialized results have no proven correspondence: filter
  sequentially or derive predicates from one source.
- Joins require explicit keys (`cross` excepted); row order without an
  explicit `sort` is backend-defined. Relational row ordering is never
  guaranteed unless an operation with documented ordering semantics
  (`sort`, single-row reductions) establishes it. Row identity, row
  ordering, and logical relational equivalence are distinct: two
  evaluations may return the same logical rows in different positions,
  and independently materialized results (even from the same logical
  query) have no proven positional correspondence — never zip them with
  dense positional alignment. Supported patterns: derive related
  features in one logical query; materialize related columns together;
  `sort` by a deterministic key when an ordered result is required;
  align separate results through explicit stable identifiers.
- Reductions preserve labels (one-row frames); `by=` groups explicitly.
  `over()` broadcasts per-group aggregates back to row shape (unlike
  reductions).
- Counting: `frame.count(by=key)` gives per-group row counts (null rows
  included, null keys form their own group) in a `count` column of the
  backend's integer dtype; `series.count(by=key)` gives per-group
  non-null value counts. The two are never conflated. Row and value
  counts agree only when no nulls are present.
- `@`, transpose, reshape on heterogeneous frames are rejected as
  genuinely undefined. Ambiguous operations raise descriptive errors;
  nothing is silently discarded, reordered, or invented.
- Missing values: NaN and null are distinct (dense IEEE vs structured
  total-order equality; see `docs/COMPARISONS.md`). Filters **must**
  use `is_null()`/`not_null()`; `==` self-comparisons are not missing-
  value predicates. `to_dense(nulls="nan")` promotes null to NaN;
  `nulls="error"` rejects missing values (`TL0418`).

## 12. Mathematics

- Elementwise: `sin cos tan exp log sqrt`; reductions; `norm`, `cov`,
  `corr`, `quantile`; tensor algebra `einsum`, `tensordot`, `moveaxis`;
  features `onehot`, `polynomial`; linear algebra `svd`, `eigh`, `solve`.
- Symbolic calculus (SymPy): `diff`, `simplify`, exact `integrate`.
  Definite integrals with no exact closed form raise; numerical fallback
  is never silent. SymPy handles mathematical subexpressions only.
- Numerical linear algebra uses stable LAPACK-backed routines. `svd`
  returns a `(U, S, Vh)` triple sharing one computation; `eigh` requires a
  square symmetric matrix (ascending values) and rejects general matrices
  toward `svd`; `solve` requires square non-singular systems. Sign
  ambiguity of singular/eigenvectors is inherent: compare reconstructions,
  variances, or subspaces — never raw signs.
- `cov`/`corr` default to NumPy `rowvar` for dense data; frames always use
  columns-as-variables with labeled output.

## 13. Execution model

Compatible structured subgraphs lower into a single backend lazy query
(predicate/projection pushdown delegated); anything else evaluates
node-by-node with identical results. `explain()` shows the logical graph
plus the physical plan or fallback reason without executing; `profile()`
reports measured time, memory, backend, and shape — never estimates.
For gradient graphs, `profile()` additionally reports measured forward
and backward phase times. Fitting and I/O are explicit execution
boundaries. Multi-output nodes share one computation. Dense evaluation
memoizes shared subgraphs within one boundary execution (structurally
identical subtrees evaluate once) and releases entries at last use;
nested gradient control nodes and scans always evaluate directly, so
snapshot, scan-count, and tape semantics are unchanged. Reverse-mode
releases each cotangent after propagation (readout entries survive);
forward values stay pinned for repeated application. No loop fusion is
assumed.

## 14. Automatic differentiation (v0.4)

`grad(loss, wrt)` and `value_and_grad(loss, wrt)` construct first-class
operators; `vjp(loss, wrt)` constructs a vector-Jacobian operator applied
as `df(x, seed)`. Construction is declarative; execution is runtime.
`wrt` names an existing parameter (default: the sole parameter);
multi-parameter `wrt` returns gradients in order.

- The loss must be scalar-valued (dense rank-zero); vector outputs are
  rejected (use `vjp` with an explicit cotangent). Seeds must match the
  output shape/kind exactly.
- Differentiable inputs must be floating-point (dense float, float
  series, all-float frames); integer, Boolean, complex, string, and
  categorical inputs are rejected, never silently converted.
- Supported rules: elementwise arithmetic with broadcast reduction of
  adjoints, negation, abs (0 at 0), sin/cos/tan/exp/log/sqrt,
  sum/mean/min/max/std/var over all or selected axes (axis cotangents
  broadcast back; ties split evenly within each output element;
  zero-variance std gives 0),
  matmul (1-D/2-D/batched), transpose/reshape/flatten, concat/stack/
  moveaxis, indexing with scatter (slices, integers, masks), where/clip/
  fill_null branch masks, linear solves (implicit), the vector 2-norm,
  and fitted affine transforms (standardize/minmax/l2-normalize/impute/
  PCA) with learned parameters held constant.
- Sorting, comparisons, logic, grouping, joins, row filtering of frames,
  SVD/eigendecomposition, tensor contractions, and unfitted estimators
  are rejected with precise diagnostics. Differentiation through `.fit()`
  is deferred; `grad()` of a gradient operator is rejected (no
  second-order); gradient values reused as data are constants.
- Application binds `DiffVar` input leaves holding snapshots, so later
  `.inplace()` mutations cannot alter constructed gradient computations.
  Unused parameters yield zeros. `value_and_grad` shares one forward
  evaluation between value and gradient. Integer intermediates compute in
  float64; results are cast to input dtypes.
- Structured gradients preserve labels: all-numeric frames yield frames
  with identical columns; series yield series with identical names.
  Heterogeneous frames on differentiable paths are rejected (select
  numeric fields explicitly).

## 15. Errors

Typed errors with source locations where applicable: syntax, scope,
declaration/effect, type, shape, mutation, symbolic, differentiation
(`TLDiffError`), unsupported. Messages **must** name the offending
operation, the representations involved, and the corrective action
(explicit selection, `.fit` first, `=` vs `:=`).

Every error carries a stable machine-readable code (`TL0000`–`TL9999`,
rendered `ClassName[CODE]: loc: message`) plus optional structured
details (available names, expected/got shapes). Codes are catalogued in
`docs/ERROR_CODES.md` (generated from the registry in `core/errors.py`)
and are stable API: tools and programmers may rely on them across message
rewording. Ranges mirror the categories above; `x99` in each range is the
generic class fallback for sites not yet individually coded.

## 16. Chunked tensors (v0.5)

Chunking is a physical execution property, not a new operator kind.
`T.scan_zarr(path)` describes a persistent array without reading data
(declarative, legal under `:=`); only store metadata (logical shape,
dtype, chunk shape, partition count, byte counts) is inspected, lazily.
Operators observe one logical tensor regardless of partitioning.

Supported over chunks: elementwise arithmetic/comparisons, the six
elementary functions, where/clip, null masks, transpose/reshape/flatten,
concat/stack, integer/slice indexing, whole-array reductions, and
axis/keepdims reductions (tree-combined per axis, bounded memory).
Anything else raises `TL1001` naming the
operation instead of silently materializing. `write_zarr(path)` streams
chunk-by-chunk (`TL1002` unless `overwrite=True`); `inplace()` into
chunked sources is rejected; fitting and differentiation over chunked
data are deferred (`TL1001` / `TL0809`).

`abstract()` reports logical shape, dtype, storage, chunks, partitions,
byte counts, and materialized status without reading data; unknown
dimensions would stay explicitly unknown (zarr metadata always provides
them). Peak working memory tracks chunk size rather than data size
(in-flight chunk buffers times worker threads): size chunks to fit the
budget. `explain()` shows the dask plan (partitions, tasks, output
shape); `profile()` reports the `dask-chunked` backend with chunk
execution counts.

Milestone 2 classifies each operation by physical requirements —
chunk-local, tree-reducible, alignment-dependent, rechunk-dependent,
neighborhood, global — and records the classification in the plan.
Multi-input operations align to the finest common boundary subdivision
(sides only split, never shuffle); broadcast axes need no movement.
Admission is gated by conservative task-count and block-byte limits
(heuristics, not guarantees; adjustable for tests and power users).
Transpose/moveaxis are metadata-only; reshape admits even merge/split
layouts and rejects unsafe merges; matmul admits rank-2 products with an
unchunked contract axis; masks, fancy indexing, sorting, joins, grouping,
contractions, fitting, and differentiation stay rejected with specific
codes. Mathematical validity (shapes, dtypes) is always checked
separately from physical feasibility, with distinct error categories.

Milestone 3 adds an explicit memory-budget contract
(`set_chunk_policy(memory_budget_bytes=..., max_workers=...)`; no
language-grammar change). The budget covers *estimated incremental
working memory*: per-task working sets (operator-specific models from
chunk layouts × dtypes: inputs, outputs, temporaries, broadcast/
rechunk/reduction/matmul intermediates) times active workers, plus
output materialization for `collect()` (one output block for streaming
sinks) plus a linear task-graph overhead term. Included and excluded
allocations are documented in `runtime/chunk_plan.py`: baseline
interpreter/BLAS/thread-pool allocations, allocator fragmentation, and
already-running native operations are NOT modeled, and Dask's own
worker limits are best-effort (its pause mechanism cannot stop running
tasks) — so estimates are planning heuristics, and only the operating
system can enforce hard process containment. Unknown requirements are
never treated as zero (`TL1021`).

Decisions are inspectable per purpose (`collect` vs `stream`): admitted,
admitted with reduced concurrency (workers → 1; the planner never
silently rechunks data to fit a budget), or rejected (`TL1020` modeled
peak over budget with a suggested budget, `TL1023` single-block over
budget needing smaller chunks, `TL1024` collect-output over budget
suggesting `write_zarr`, whose outputs may exceed the working budget).
Concurrency uses a per-call worker cap — never global Dask config, never
leaking between executions. `collect()` checks output size against the
budget; rejected computations never fall back to NumPy evaluation.
Runtime monitoring records baseline/peak high-water RSS and the delta
above baseline separately from the modeled estimate; far-above-budget
observations report `TL1022` pressure (already-running native operations
may transiently overshoot). `explain()` shows the admission decision and
its assumptions, `abstract()` carries the policy without reading data,
and `profile()` reports baseline/peak/estimated/budget/workers/tasks
with estimates, measurements, and unavailable values clearly labeled.
Writes are incremental, never claimed atomic.

Milestone 4 adds execution reliability without new syntax. Each chunked
run follows an internal lifecycle — planned, admitted, running,
committing (writes), completed, failed, or cancelled — recorded per run
(no user-managed state object, no leakage across runs) and summarized
for `profile()` tooling. Persistent output has two modes on one surface
(`write_zarr(path, overwrite=False, staged=None)`): direct streaming
(lower storage overhead; partial outputs possible after failure) and
staged publication (default for local stores; complete and validate in
a same-filesystem staging directory — chunk census and metadata only,
never data reads — then publish via rename). Each individual rename is
atomic on POSIX, but backup-based overwrite is a recoverable multi-step
replacement (backup rename, then publish rename), not one continuously
atomic overwrite: between the two renames the destination is briefly
absent, and power-loss durability is not claimed (see
docs/RECOVERY.md). Staged runs never publish a
partial dataset as success and never destroy a preexisting valid
destination before its replacement validates (`overwrite=True` swaps via
a backup round-trip); concurrent publications are detected at publish
time (loser keeps its validated staging for recovery). Non-local stores
keep direct streaming with a documented warning — no distributed
transactions are introduced.

Failures preserve valid destinations, release run resources, clean
staging (validated-but-unpublished staging is retained and reported,
never silently deleted), and keep source-level plus backend error
information; interrupted runs (`KeyboardInterrupt`/cancellation) stop
scheduling new work while recognizing that running native tasks may
finish, then clean up once workers stop. Abrupt process termination
cannot run cleanup: stale staging is identifiable and removable via
`tensorlang.runtime.chunk_plan.list_stale_staging()` /
`recover_stale_staging()` (see `docs/RECOVERY.md`). External sources are
checked against their planned baseline before execution (shape/dtype/
chunk changes always caught; content-only changes best-effort via
filesystem stamps, with report-once resynchronization); in-memory
snapshot semantics are unchanged, and no full-dataset reads are used
for validation. Disk-space (storage) and RAM-budget requirements are
reported separately; no rechunk-to-fit is introduced. New diagnostics
`TL1030`–`TL1035` flow through the existing registry, and `write_zarr`
is classified effectful so `:=` cannot bypass I/O protection
(`describe_write()` is read-only introspection).

## 18. Modules (v0.8 M2)

`.tl` files import each other with existing syntax (`import m`,
`from m import x`, aliases, dotted paths). Dotted names map to
relative paths (`a.b` → `a/b.tl`) resolved against the importing
file; Python imports take precedence; parents are inert namespace
packages; relative imports are rejected (`TL0314`).

Every module compiles through the existing frontend (same scope and
M1 effect validation), executes in an isolated namespace, and is
cached per loader session: the same file requested twice under one
spelling reuses its initialized module and operator identities, while
different spellings initialize independently (as in CPython);
independent executions stay isolated.
Circular imports are rejected (`TL0312`) by static pre-scan before
execution; missing modules/symbols report chains (`TL0310`/`TL0311`);
initialization failures chain their cause (`TL0313`).

Imported names are ordinary bindings: `pipeline := f >> g` and
`result = f(data)` work; `r := f(data)` through an imported name is
`TL0305` (use `=`). Parameter identity stays UID-based. CLI
graph/explain/profile preflight inspects the transitive `.tl` graph
without executing it. Bare directories import as inert package shells
(`__init__` never executes). Loader sessions are process-global and
not thread-safe: overlapping cross-thread sessions are rejected with
`TL0315`; parallel execution **must** use subprocesses. See `docs/MODULES.md`.

## 19. Version history

- **v0.1**: frontend, scoped operators, dense/structured/symbolic core,
  lazy execution, explicit mutation.
- **v0.2**: effect classification with provenance; scans; joins/sort/
  unique/conditional/window operators; single-query lowering with fallback;
  `explain`/`profile`; streaming writes; CoW snapshots; benchmarks.
- **v0.3**: fittable operators and sequential pipeline fitting with
  no-refit guarantees; ML preprocessing (standardize/minmax/normalize/
  impute/PCA); numerical linear algebra and tensor algebra; this
  specification.
- **v0.4**: reverse-mode automatic differentiation (`grad`,
  `value_and_grad`, `vjp`) over existing operator graphs with
  vector-Jacobian rules, fitted-affine differentiation, structured
  gradients with label preservation, and phase-split profiling.
- **v0.5 M1**: chunked tensor foundation — `T.scan_zarr` lazy persistent
  sources, dask-backed lowering for elementwise/shape/reduction graphs,
  streaming `write_zarr`, bounded-memory execution, `TL10xx` diagnostics,
  chunk metadata in `abstract()`/`explain()`/`profile()`.
- **v0.5 M2**: chunk-aware planning — operation capability classes, union
  subdivision alignment with recorded movement, policy gates (tasks, block
  bytes), bounded matmul, reshape/transpose planning, `TL1010`–`TL1013`
  diagnostics, plan introspection with estimates.
- **v0.5 M3**: memory budgeting and runtime resource control —
  working-memory budget with per-task resource models, admit /
  admit-reduced-concurrency / reject decisions per purpose, per-call
  worker caps, baseline-vs-estimate monitoring, streaming outputs larger
  than the budget, `TL1020`–`TL1024` diagnostics, estimate-vs-measured
  introspection.
- **v0.5 M4**: execution reliability and persistent data integrity —
  per-run lifecycle states, staged publication with POSIX-atomic renames
  (recoverable multi-step replacement on overwrite; see docs/RECOVERY.md),
  output validation without data reads, source-consistency checks with
  report-once resynchronization, interruption/cancellation handling,
  stale-staging recovery, `TL1030`–`TL1035` diagnostics.
